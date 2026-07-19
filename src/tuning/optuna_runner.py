"""
Optuna 하이퍼파라미터 최적화 공용 실행기.

목적함수: 5-Fold GroupKFold 평균 대분류 macro_F1 최대화.
중분류·카테고리태그는 보조 지표로만 로깅한다.

사용 예:
    runner = OptunaRunner(
        model_name="tfidf_lgbm",
        model_cls=TfidfLgbmClassifier,
        search_space=config["tfidf_lgbm"],
        folds=folds,
        df=df,
        n_trials=100,
        storage="sqlite:///results/optuna.db",
    )
    result = runner.run()
    print(result.best_params, result.best_score)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Type

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score

from ..models.base import BaseClassifier

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class StudyResult:
    """Optuna study 완료 결과."""
    model_name:   str
    best_score:   float                    # 대분류 macro_F1 (CV 평균)
    best_params:  dict = field(default_factory=dict)
    all_trials_df: pd.DataFrame | None = None   # trial별 상세 (HTML 리포트용)


class OptunaRunner:
    """
    모델 공통 Optuna HPO 실행기.

    Args:
        model_name:   결과 식별자 ("tfidf_lgbm" | "fasttext" | "koelectra")
        model_cls:    BaseClassifier 서브클래스
        search_space: experiment.yaml 해당 모델 섹션 딕셔너리
        folds:        make_folds() 반환 [(train_idx, val_idx), ...]
        df:           fit_transform() 완료된 DataFrame (refined_text, nouns_text, label_* 포함)
        n_trials:     탐색 trial 수
        sampler:      "TPE" | "Random" | "CmaEs"
        pruner:       "MedianPruner" | "NopPruner"
        n_startup_trials: TPE 초기 Random trial 수
        n_warmup_steps:   Pruner warm-up fold 수
        timeout:      최대 탐색 시간 (초, None=무제한)
        storage:      Optuna DB URI (None=메모리)
        study_name:   study 식별자 (None이면 "ref_{model_name}"으로 자동 생성)
                      지표 변경 시 suffix를 바꿔 격리 (예: "ref_tfidf_lgbm_v2")
    """

    def __init__(
        self,
        model_name: str,
        model_cls: Type[BaseClassifier],
        search_space: dict,
        folds: list[tuple[list[int], list[int]]],
        df: pd.DataFrame,
        n_trials: int = 50,
        sampler: str = "TPE",
        pruner: str = "MedianPruner",
        n_startup_trials: int = 10,
        n_warmup_steps: int = 2,
        timeout: float | None = 3600.0,
        storage: str | None = None,
        extra_kwargs: dict | None = None,
        study_name: str | None = None,      # ← 추가: 외부에서 study 이름 주입
        level: str = "joint",               # ★"joint"(기존) | "large" | "medium" — 2단계 HPO 모드
        frozen_params: dict | None = None,  # ★level="medium" 시 stage-1 확정 파라미터 (동결)
    ) -> None:
        self._model_name      = model_name
        self._model_cls       = model_cls
        self._search_space    = search_space
        self._folds           = folds
        self._df              = df
        self._n_trials        = n_trials
        self._timeout         = timeout
        self._storage         = storage
        self._extra_kwargs    = extra_kwargs or {}
        self._level           = level
        self._frozen_params   = dict(frozen_params or {})
        # level="medium" 전용: fold별 (피처+대분류) 고정 베이스 모델 캐시
        #   피처·대분류 파라미터가 동결 상태라 trial 간 재사용 가능 → 수 배 고속화
        self._base_models: dict[int, BaseClassifier] = {}

        # study_name: 외부 주입 우선, 없으면 기본값
        self._study_name = study_name or f"ref_{model_name}"

        # Sampler
        match sampler:
            case "TPE":
                self._sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
            case "Random":
                self._sampler = optuna.samplers.RandomSampler()
            case "CmaEs":
                self._sampler = optuna.samplers.CmaEsSampler()
            case _:
                self._sampler = optuna.samplers.TPESampler()

        # Pruner
        match pruner:
            case "MedianPruner":
                self._pruner = optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps)
            case "NopPruner":
                self._pruner = optuna.pruners.NopPruner()
            case _:
                self._pruner = optuna.pruners.MedianPruner()

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def run(self) -> StudyResult:
        """Optuna study를 실행하고 StudyResult를 반환한다."""
        study = optuna.create_study(
            study_name=self._study_name,     # ← 하드코딩 제거, 주입된 이름 사용
            direction="maximize",
            sampler=self._sampler,
            pruner=self._pruner,
            storage=self._storage,
            load_if_exists=True,
        )
        logger.info(
            "[%s] Optuna study 시작: %d trials (study_name=%s, level=%s)",
            self._model_name, self._n_trials, self._study_name, self._level,
        )

        study.optimize(
            self._objective,
            n_trials=self._n_trials,
            timeout=self._timeout,
            show_progress_bar=True,
            callbacks=[self._trial_callback],
        )

        best = study.best_trial
        logger.info(
            "[%s] 최적 trial #%d — large_F1=%.4f",
            self._model_name, best.number, best.value
        )

        trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

        return StudyResult(
            model_name=self._model_name,
            best_score=best.value,
            best_params=best.params,
            all_trials_df=trials_df,
        )

    # ──────────────────────────────────────────────────────────────
    # Objective
    # ──────────────────────────────────────────────────────────────

    def _trial_callback(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """trial 완료 시 결과를 INFO로 출력한다."""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            logger.info(
                "[%s] Trial %3d 완료 — large_F1=%.4f  (best=%.4f, trial #%d)",
                self._model_name,
                trial.number,
                trial.value,
                study.best_value,
                study.best_trial.number,
            )
        elif trial.state == optuna.trial.TrialState.PRUNED:
            logger.info("[%s] Trial %3d Pruned", self._model_name, trial.number)

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective 함수 — level 별 목적이 다르다 (2026-07 3차 개편).

          joint  (기존): 전체 학습 → 대분류 macro_F1 (하위호환)
          large  (1단계): 피처+대분류만 학습(fit_large_only) → 대분류 macro_F1
                          중분류 학습 생략으로 trial 당 시간 대폭 단축.
          medium (2단계): stage-1 동결 파라미터로 만든 fold별 베이스(피처+대분류)를
                          캐시에서 재사용, refit_medium 으로 중분류만 재학습
                          → 중분류 macro_F1. 빔 디코딩(decode/beam_*)도 이 단계에서
                          함께 탐색된다.
        """
        params = self._suggest_params(trial)
        fold_f1s: list[float] = []

        n_folds = len(self._folds)
        for fold_idx, (tr_idx, va_idx) in enumerate(self._folds):
            tr = self._df.iloc[tr_idx]
            va = self._df.iloc[va_idx]

            logger.info(
                "[%s] Trial %d | Fold %d/%d 시작 (train=%d, val=%d)",
                self._model_name, trial.number, fold_idx + 1, n_folds, len(tr_idx), len(va_idx),
            )
            try:
                f1 = self._eval_fold(trial, fold_idx, tr, va, params)
            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.warning("[%s] Trial %d, Fold %d 실패: %s", self._model_name, trial.number, fold_idx, e)
                f1 = 0.0

            fold_f1s.append(f1)

            # Pruning: warmup 이후 성능 부진 trial 조기 종료
            trial.report(float(np.mean(fold_f1s)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_f1s))

    def _eval_fold(self, trial, fold_idx: int, tr, va, params: dict) -> float:
        """단일 fold 학습·평가 — level 별 경로 분기."""
        if self._level == "medium":
            # ── 베이스 캐시: 동결 파라미터로 (피처+대분류)를 fold 당 1회만 학습 ──
            if fold_idx not in self._base_models:
                base = self._model_cls(**self._extra_kwargs)
                base.fit(
                    tr["refined_text"], tr["nouns_text"],
                    tr["label_large"].values, tr["label_medium"].values,
                    tr["label_tag"].values,
                    fit_large_only=True, **self._frozen_params,
                )
                self._base_models[fold_idx] = base
                logger.info("[%s] Fold %d 베이스(피처+대분류) 캐시 생성", self._model_name, fold_idx + 1)
            model = self._base_models[fold_idx]
            model.refit_medium(
                tr["refined_text"], tr["nouns_text"],
                tr["label_large"].values, tr["label_medium"].values,
                tr["label_tag"].values,
                **params,
            )
            _, pred_medium, _ = model.predict(va["refined_text"], va["nouns_text"])
            return f1_score(va["label_medium"].values, pred_medium,
                            average="macro", zero_division=0)

        # ── joint / large: 매 trial 새 모델 ──
        model = self._model_cls(**self._extra_kwargs)
        model.fit(
            tr["refined_text"], tr["nouns_text"],
            tr["label_large"].values, tr["label_medium"].values,
            tr["label_tag"].values,
            fit_large_only=(self._level == "large"),
            **{**self._frozen_params, **params},
        )
        pred_large, _, _ = model.predict(va["refined_text"], va["nouns_text"])
        return f1_score(va["label_large"].values, pred_large,
                        average="macro", zero_division=0)

    # ──────────────────────────────────────────────────────────────
    # 파라미터 샘플링 (모델별 search_space → Optuna suggest_*)
    # ──────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────
    # 파라미터 샘플링 (2026-07 개편)
    #
    # 탐색 공간이 3개 그룹으로 확장됨:
    #   feature : char/word n-gram · head-noun 가중 (FeatureConfig 필드)
    #   clf     : ★모델 종류(model_type_large/medium) · C · class_weight
    #   lgbm    : LightGBM 전용 (어느 헤드든 lgbm/ensemble 을 고른 trial 만 샘플링
    #             → Optuna define-by-run 조건부 공간으로 낭비 차원 제거)
    #
    # FastText / KoELECTRA 전용 공간은 제거됨:
    #   - FastText 모델 자체가 삭제됨 (char n-gram 과 사실상 중복)
    #   - KoELECTRA 는 술 OOV 구제 후순위 옵션으로만 남아 HPO 대상에서 제외
    # ──────────────────────────────────────────────────────────────

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        """experiment.yaml search_space 딕셔너리를 Optuna suggest_* 호출로 변환한다."""
        if self._level == "medium":
            return self._suggest_medium_params(trial)   # ★2단계: 중분류 전용 공간
        params: dict = {}
        sp = self._search_space

        # ── 피처 공간 (feature) ──
        if "feature" in sp:
            fe = sp["feature"]
            if "char_ngram_range" in fe:
                # [[2,4],[2,5],[3,5]] 형태의 후보 목록에서 인덱스로 선택
                i = trial.suggest_categorical(
                    "char_ngram_idx", list(range(len(fe["char_ngram_range"])))
                )
                params["char_ngram_range"] = tuple(fe["char_ngram_range"][i])
            if "max_features" in fe:
                params["max_features"] = trial.suggest_int("max_features", *fe["max_features"], log=True)
            if "min_df" in fe:
                params["min_df"] = trial.suggest_int("min_df", *fe["min_df"])
            if "sublinear_tf" in fe:
                params["sublinear_tf"] = trial.suggest_categorical("sublinear_tf", fe["sublinear_tf"])
            if "word_ngram_range" in fe:
                i = trial.suggest_categorical(
                    "word_ngram_idx", list(range(len(fe["word_ngram_range"])))
                )
                params["word_ngram_range"] = tuple(fe["word_ngram_range"][i])
            if "word_max_features" in fe:
                params["word_max_features"] = trial.suggest_int(
                    "word_max_features", *fe["word_max_features"], log=True
                )
            if "head_weight" in fe:
                params["head_weight"] = trial.suggest_float("head_weight", *fe["head_weight"])

        # ── 분류기 선택 공간 (clf) — ★헤드별 베스트 모델 탐색 ──
        need_lgbm = False       # lgbm/ensemble 이 선택된 경우에만 LGBM 공간 샘플링
        need_c    = False       # 선형 계열이 선택된 경우에만 C 샘플링
        if "clf" in sp:
            cl = sp["clf"]
            if "model_type" in cl:
                # 대분류/중분류 헤드가 서로 다른 모델을 가질 수 있다
                params["model_type_large"] = trial.suggest_categorical(
                    "model_type_large", cl["model_type"]
                )
                if self._level == "large":
                    # ★1단계: 중분류 헤드는 학습하지 않으므로 샘플링 자체를 생략
                    #   (낭비 차원 제거 — TPE 가 대분류 공간에만 집중)
                    chosen = {params["model_type_large"]}
                else:
                    params["model_type_medium"] = trial.suggest_categorical(
                        "model_type_medium", cl["model_type"]
                    )
                    chosen = {params["model_type_large"], params["model_type_medium"]}
                need_lgbm = bool(chosen & {"lgbm", "ensemble"})
                need_c    = bool(chosen & {"linearsvc", "logreg", "ensemble"})
            if "C" in cl and need_c:
                params["C"] = trial.suggest_float("C", *cl["C"], log=True)
            if "class_weight" in cl:
                # YAML 의 null 은 None 으로 로드됨 → 그대로 categorical 후보로 사용
                params["class_weight"] = trial.suggest_categorical("class_weight", cl["class_weight"])

        # ── LightGBM 공간 (조건부) ──
        if "lgbm" in sp and need_lgbm:
            g = sp["lgbm"]
            if "n_estimators" in g:
                params["n_estimators"] = trial.suggest_int("n_estimators", *g["n_estimators"], log=True)
            if "num_leaves" in g:
                params["num_leaves"] = trial.suggest_int("num_leaves", *g["num_leaves"], log=True)
            if "max_depth" in g:
                params["max_depth"] = trial.suggest_int("max_depth", *g["max_depth"])
            if "learning_rate" in g:
                params["learning_rate"] = trial.suggest_float("learning_rate", *g["learning_rate"], log=True)
            if "min_child_samples" in g:
                params["min_child_samples"] = trial.suggest_int(
                    "min_child_samples", *g["min_child_samples"], log=True
                )
            if "colsample_bytree" in g:
                params["colsample_bytree"] = trial.suggest_float("colsample_bytree", *g["colsample_bytree"])
            if "subsample" in g:
                params["subsample"] = trial.suggest_float("subsample", *g["subsample"])
            if "reg_alpha" in g:
                params["reg_alpha"] = trial.suggest_float("reg_alpha", *g["reg_alpha"], log=True)
            if "reg_lambda" in g:
                params["reg_lambda"] = trial.suggest_float("reg_lambda", *g["reg_lambda"], log=True)

        return params

    def _suggest_medium_params(self, trial: optuna.Trial) -> dict:
        """
        ★2단계 HPO 전용 탐색 공간 — 중분류 헤드 + LCPN 디코딩 정책만 샘플링.

        피처·대분류 파라미터는 frozen_params 로 동결되어 여기서 건드리지 않는다.
        범위는 experiment.yaml 의 clf/lgbm 섹션을 재사용하고(없으면 기본값),
        빔 디코딩(k·margin·λ)은 중분류 F1 에 직접 작용하므로 함께 탐색한다.
        """
        params: dict = {}
        sp = self._search_space
        cl = sp.get("clf", {})
        g  = sp.get("lgbm", {})

        # ── 중분류 헤드 모델·정규화 (전용 키: *_medium → 대분류 헤드 불간섭) ──
        mt = trial.suggest_categorical(
            "model_type_medium", cl.get("model_type", ["linearsvc", "logreg", "lgbm", "ensemble"])
        )
        params["model_type_medium"] = mt
        if mt in ("linearsvc", "logreg", "ensemble"):
            c_lo, c_hi = cl.get("C", [1e-2, 30.0])
            params["C_medium"] = trial.suggest_float("C_medium", c_lo, c_hi, log=True)
        params["class_weight_medium"] = trial.suggest_categorical(
            "class_weight_medium", cl.get("class_weight", [None, "balanced"])
        )
        if mt in ("lgbm", "ensemble"):
            params["n_estimators_medium"] = trial.suggest_int(
                "n_estimators_medium", *g.get("n_estimators", [100, 800]), log=True)
            params["num_leaves_medium"] = trial.suggest_int(
                "num_leaves_medium", *g.get("num_leaves", [15, 255]), log=True)
            params["max_depth_medium"] = trial.suggest_int(
                "max_depth_medium", *g.get("max_depth", [3, 12]))
            params["learning_rate_medium"] = trial.suggest_float(
                "learning_rate_medium", *g.get("learning_rate", [0.01, 0.3]), log=True)
            params["min_child_samples_medium"] = trial.suggest_int(
                "min_child_samples_medium", *g.get("min_child_samples", [5, 60]), log=True)

        # ── LCPN 디코딩 정책 (greedy vs 빔 결합) ──
        decode = trial.suggest_categorical("decode", ["greedy", "beam"])
        params["decode"] = decode
        if decode == "beam":
            params["beam_size"]    = trial.suggest_int("beam_size", 2, 4)
            params["beam_margin"]  = trial.suggest_float("beam_margin", 0.05, 0.6)
            params["joint_lambda"] = trial.suggest_float("joint_lambda", 0.3, 2.5)

        return params
