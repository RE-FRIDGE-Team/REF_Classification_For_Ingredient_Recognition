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
        extra_kwargs: dict | None = None,   # 모델 생성자 추가 인수
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
        study_name = f"ref_{self._model_name}"
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=self._sampler,
            pruner=self._pruner,
            storage=self._storage,
            load_if_exists=True,         # 중단 후 재개 지원
        )
        logger.info("[%s] Optuna study 시작: %d trials", self._model_name, self._n_trials)

        study.optimize(
            self._objective,
            n_trials=self._n_trials,
            timeout=self._timeout,
            show_progress_bar=False,
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

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective 함수.

        trial 파라미터 샘플링 → 모델 생성 → 5-Fold CV → 대분류 macro_F1 반환
        """
        params = self._suggest_params(trial)
        fold_f1s: list[float] = []

        for fold_idx, (tr_idx, va_idx) in enumerate(self._folds):
            tr = self._df.iloc[tr_idx]
            va = self._df.iloc[va_idx]

            model = self._model_cls(
                **{k: v for k, v in self._extra_kwargs.items()},
            )
            try:
                model.fit(
                    tr["refined_text"], tr["nouns_text"],
                    tr["label_large"].values,
                    tr["label_medium"].values,
                    tr["label_tag"].values,
                    **params,
                )
                pred_large, _, _ = model.predict(va["refined_text"], va["nouns_text"])
                f1 = f1_score(va["label_large"].values, pred_large, average="macro", zero_division=0)
            except Exception as e:
                logger.warning("[%s] Trial %d, Fold %d 실패: %s", self._model_name, trial.number, fold_idx, e)
                f1 = 0.0

            fold_f1s.append(f1)

            # Pruning: warmup 이후 성능 부진 trial 조기 종료
            trial.report(float(np.mean(fold_f1s)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_f1s))

    # ──────────────────────────────────────────────────────────────
    # 파라미터 샘플링 (모델별 search_space → Optuna suggest_*)
    # ──────────────────────────────────────────────────────────────

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        """experiment.yaml search_space 딕셔너리를 Optuna API로 변환."""
        params: dict = {}
        sp = self._search_space

        # ── TF-IDF 공간 ──
        if "tfidf" in sp:
            t = sp["tfidf"]
            if "ngram_range" in t:
                choice = trial.suggest_categorical("ngram_range_idx", list(range(len(t["ngram_range"]))))
                params["ngram_range"] = tuple(t["ngram_range"][choice])
            if "max_features" in t:
                params["max_features"] = trial.suggest_int("max_features", *t["max_features"], log=True)
            if "min_df" in t:
                params["min_df"] = trial.suggest_int("min_df", *t["min_df"])
            if "sublinear_tf" in t:
                params["sublinear_tf"] = trial.suggest_categorical("sublinear_tf", t["sublinear_tf"])

        # ── LightGBM 공간 ──
        if "lgbm" in sp:
            g = sp["lgbm"]
            if "n_estimators" in g:
                params["n_estimators"]      = trial.suggest_int("n_estimators", *g["n_estimators"], log=True)
            if "num_leaves" in g:
                params["num_leaves"]        = trial.suggest_int("num_leaves", *g["num_leaves"], log=True)
            if "max_depth" in g:
                lo, hi = g["max_depth"]
                params["max_depth"]         = trial.suggest_int("max_depth", lo, hi)
            if "learning_rate" in g:
                params["learning_rate"]     = trial.suggest_float("learning_rate", *g["learning_rate"], log=True)
            if "min_child_samples" in g:
                params["min_child_samples"] = trial.suggest_int("min_child_samples", *g["min_child_samples"], log=True)
            if "colsample_bytree" in g:
                params["colsample_bytree"]  = trial.suggest_float("colsample_bytree", *g["colsample_bytree"])
            if "subsample" in g:
                params["subsample"]         = trial.suggest_float("subsample", *g["subsample"])
            if "reg_alpha" in g:
                params["reg_alpha"]         = trial.suggest_float("reg_alpha", *g["reg_alpha"], log=True)
            if "reg_lambda" in g:
                params["reg_lambda"]        = trial.suggest_float("reg_lambda", *g["reg_lambda"], log=True)

        # ── FastText 공간 ──
        if "fasttext" in sp:
            ft = sp["fasttext"]
            if "vector_size" in ft:
                params["vector_size"] = trial.suggest_int("vector_size", *ft["vector_size"])
            if "window" in ft:
                params["window"]      = trial.suggest_int("window", *ft["window"])
            if "min_count" in ft:
                params["min_count"]   = trial.suggest_int("min_count", *ft["min_count"])
            if "epochs" in ft:
                params["epochs"]      = trial.suggest_int("ft_epochs", *ft["epochs"])
            if "sg" in ft:
                params["sg"]          = trial.suggest_int("sg", *ft["sg"])
            if "negative" in ft:
                params["negative"]    = trial.suggest_int("negative", *ft["negative"])

        # ── FastText 다운스트림 분류기 ──
        if "classifier" in sp:
            clf_choices = sp["classifier"].get("type", ["logreg"])
            clf_type = trial.suggest_categorical("classifier_type", clf_choices)
            params["classifier_type"] = clf_type
            if clf_type == "logreg" and "logreg" in sp["classifier"]:
                lr_sp = sp["classifier"]["logreg"]
                if "C" in lr_sp:
                    params["C"] = trial.suggest_float("C", *lr_sp["C"], log=True)
                if "max_iter" in lr_sp:
                    params["max_iter"] = trial.suggest_int("max_iter", *lr_sp["max_iter"])

        # ── KoELECTRA 공간 ──
        if "training" in sp:
            tr = sp["training"]
            if "learning_rate" in tr:
                params["learning_rate"] = trial.suggest_float("lr", *tr["learning_rate"], log=True)
            if "batch_size" in tr:
                params["batch_size"]    = trial.suggest_categorical("batch_size", tr["batch_size"])
            if "epochs" in tr:
                params["epochs"]        = trial.suggest_int("epochs", *tr["epochs"])
            if "max_length" in tr:
                params["max_length"]    = trial.suggest_categorical("max_length", tr["max_length"])
            if "dropout" in tr:
                params["dropout"]       = trial.suggest_float("dropout", *tr["dropout"])
            if "freeze_layers" in tr:
                params["freeze_layers"] = trial.suggest_int("freeze_layers", *tr["freeze_layers"])
            if "warmup_ratio" in tr:
                params["warmup_ratio"]  = trial.suggest_float("warmup_ratio", *tr["warmup_ratio"])
            if "weight_decay" in tr:
                params["weight_decay"]  = trial.suggest_float("weight_decay", *tr["weight_decay"])

        if "loss_weights" in sp:
            lw = sp["loss_weights"]
            w_large  = trial.suggest_float("w_large",  *lw["large"])
            w_medium = trial.suggest_float("w_medium", *lw["medium"])
            w_tag    = trial.suggest_float("w_tag",    *lw["tag"])
            total = w_large + w_medium + w_tag
            params["w_large"]  = w_large  / total
            params["w_medium"] = w_medium / total
            params["w_tag"]    = w_tag    / total

        return params
