"""
통합 텍스트 분류기 — TF-IDF/BM25 FeatureUnion × {LightGBM, 선형, CNB, 앙상블}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[기존 tfidf_lgbm.py 대비 변경점]

  1. 피처: 이중 char TF-IDF hstack → src/features.py 의 FeatureBuilder 로 교체.
     (char + word FeatureUnion, head-noun 가중, 술 스타일어, BM25 토글)

  2. 분류기 선택: LightGBM 고정 → model_type 파라미터로 교체.
       "lgbm"      : LightGBM (기존)
       "linearsvc" : LinearSVC — 고차원 희소 TF-IDF 에서 강력·고속
       "logreg"    : LogisticRegression (multinomial, saga 미사용·lbfgs)
       "cnb"       : ComplementNB — 불균형 텍스트 분류 특화 NB 변형
       "ensemble"  : LogReg + LGBM soft-voting (predict_proba 평균)
     ★대분류/중분류 헤드가 서로 다른 model_type 을 가질 수 있다
       (model_type_large / model_type_medium) — Optuna 가 헤드별 베스트를 고른다.

  3. class_weight='balanced' 기본 — macro_F1 지표 하에서 희소 클래스(술 등) 보호.
     (ComplementNB 는 class_weight 미지원 → 자체 보정 로직이 그 역할을 대신함)

  4. 룰 레이어: predict 직후 HeadRuleEngine 으로 모호 헤드어 반례 오버라이드.
     (rule_engine 미주입 시 완전히 비활성 — 기존 동작과 동일)

  5. LCPN 구조 유지: 대분류 단일 분류기 → 대분류별 중분류 분류기 → 중분류→태그 맵.

[의존성 근거]
  scikit-learn>=1.4 : LinearSVC(dual="auto"), LogisticRegression 다중클래스 자동
                      처리(1.5+에서 multi_class 인자 deprecated → 인자 미지정으로
                      기본 multinomial 동작 사용, 버전 호환 안전).
  lightgbm>=4.3     : class_weight="balanced" 지원 (기존 코드와 동일).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False
    logger.error("lightgbm 미설치. pip install lightgbm")

from ..features import FeatureBuilder, FeatureConfig
from ..rules import HeadRuleEngine
from .base import BaseClassifier

# 헤드(대분류/중분류)별로 선택 가능한 분류기 종류
MODEL_TYPES = ("lgbm", "linearsvc", "logreg", "cnb", "ensemble")


class TextPipelineClassifier(BaseClassifier):
    """
    FeatureBuilder × 선택형 분류기 3-head(대/중/태그) LCPN 분류기.

    파라미터는 세 그룹으로 나뉜다 (Optuna trial 파라미터가 fit(**params)로 주입됨):
      - feature_* / char_* / word_* / head_weight / vectorizer : FeatureConfig
      - model_type_large / model_type_medium / C / class_weight : 분류기 선택
      - n_estimators / num_leaves / ... : LightGBM 전용 (lgbm/ensemble 선택 시)
    """

    def __init__(
        self,
        # ── 피처 토글 (CLI에서 고정 주입 — A/B 실험 축) ──
        use_char: bool = True,
        use_word: bool = True,
        use_head_noun: bool = True,
        use_gin_head: bool = True,
        use_alcohol_lexicon: bool = True,
        use_alcohol_brands: bool = True,
        vectorizer: str = "tfidf",
        gin_vocab: list[str] | None = None,
        # ── 룰 레이어 ──
        rule_engine: HeadRuleEngine | None = None,
        # ── 기본 하이퍼파라미터 (Optuna 가 fit 시 오버라이드) ──
        model_type_large: str = "linearsvc",
        model_type_medium: str = "linearsvc",
        C: float = 1.0,
        class_weight: str | None = "balanced",
        seed: int = 42,
    ) -> None:
        # 피처 설정 — 토글은 생성자에서, 세부 수치는 fit(**params)에서 갱신
        self._feature_cfg = FeatureConfig(
            use_char=use_char,
            use_word=use_word,
            use_head_noun=use_head_noun,
            use_gin_head=use_gin_head,
            use_alcohol_lexicon=use_alcohol_lexicon,
            use_alcohol_brands=use_alcohol_brands,
            vectorizer=vectorizer,
        )
        self._gin_vocab = gin_vocab       # GIN 핵어 분해 어휘 (PGIN 컬럼 유래)
        # 분류기 공통 설정
        self._clf_cfg: dict = dict(
            model_type_large=model_type_large,
            model_type_medium=model_type_medium,
            C=C,
            class_weight=class_weight,
            seed=seed,
        )
        # LightGBM 전용 설정 (기존 기본값 유지)
        self._lgbm_params: dict = dict(
            n_estimators=300, num_leaves=63, max_depth=-1, learning_rate=0.05,
            min_child_samples=20, colsample_bytree=0.8, subsample=0.8,
            reg_alpha=0.01, reg_lambda=0.01,
            class_weight=class_weight, random_state=seed, n_jobs=-1, verbose=-1,
        )

        self._rules = rule_engine                 # None 이면 룰 비활성

        # fit 후 채워지는 내부 상태
        self._features: FeatureBuilder | None = None
        self._clf_large = None                    # 대분류 분류기
        self._clf_medium_map: dict[int, object] = {}   # 대분류값 → 중분류 분류기
        self._medium_single_map: dict[int, int] = {}   # 중분류가 1종뿐인 대분류 → 그 중분류값
        self._tag_map: dict[int, int] = {}        # 중분류 → 최빈 태그
        self._last_texts: pd.Series | None = None # 룰 레이어용 원문 캐시

    # ──────────────────────────────────────────────────────────────
    # 분류기 팩토리
    # ──────────────────────────────────────────────────────────────

    def _make_clf(self, model_type: str):
        """model_type 문자열로 sklearn/LGBM 분류기 인스턴스를 생성한다."""
        cw   = self._clf_cfg["class_weight"]
        C    = self._clf_cfg["C"]
        seed = self._clf_cfg["seed"]

        if model_type == "lgbm":
            if not _LGB_AVAILABLE:
                raise ImportError("lightgbm 패키지가 필요합니다.")
            return lgb.LGBMClassifier(**self._lgbm_params)

        if model_type == "linearsvc":
            # dual="auto": n_samples > n_features 여부에 따라 자동 선택 (sklearn 1.3+)
            return LinearSVC(C=C, class_weight=cw, dual="auto", random_state=seed)

        if model_type == "logreg":
            # multi_class 인자는 sklearn 1.5+ deprecated → 미지정(기본 multinomial 동작)
            return LogisticRegression(C=C, class_weight=cw, max_iter=2000, random_state=seed)

        if model_type == "cnb":
            # ComplementNB: class_weight 미지원 — 불균형 보정이 알고리즘에 내장됨
            return ComplementNB()

        if model_type == "ensemble":
            # soft-voting은 fit/predict 를 직접 구현한 경량 래퍼 사용
            return _SoftVotingPair(
                LogisticRegression(C=C, class_weight=cw, max_iter=2000, random_state=seed),
                lgb.LGBMClassifier(**self._lgbm_params) if _LGB_AVAILABLE else None,
            )

        raise ValueError(f"지원하지 않는 model_type: {model_type} (선택지: {MODEL_TYPES})")

    # ──────────────────────────────────────────────────────────────
    # BaseClassifier 구현
    # ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_refined: pd.Series,
        X_nouns: pd.Series,
        y_large: np.ndarray,
        y_medium: np.ndarray,
        y_tag: np.ndarray,
        **params,
    ) -> None:
        # Optuna trial 파라미터 주입 (feature / clf / lgbm 각 그룹으로 라우팅)
        if params:
            self._update_params(params)

        # 1. 피처 행렬 생성 (char+word+head+lexicon)
        self._features = FeatureBuilder(self._feature_cfg, gin_vocab=self._gin_vocab)
        X = self._features.fit_transform(X_refined, X_nouns)
        logger.debug("피처 차원: %s", X.shape)

        # 2. 대분류 헤드 학습
        self._clf_large = self._make_clf(self._clf_cfg["model_type_large"])
        self._clf_large.fit(X, y_large)

        # 3. LCPN — 대분류별 중분류 헤드 학습
        label_df = pd.DataFrame({"large": y_large, "medium": y_medium, "tag": y_tag})
        self._clf_medium_map.clear()
        self._medium_single_map.clear()

        for large_val, group in label_df.groupby("large"):
            uniq = group["medium"].unique()
            if len(uniq) == 1:
                # 해당 대분류에 중분류가 1종뿐 → 분류기 불필요, 상수 매핑
                self._medium_single_map[int(large_val)] = int(uniq[0])
                continue
            X_sub = X[group.index.to_numpy()]            # 해당 대분류 행만 추출
            clf = self._make_clf(self._clf_cfg["model_type_medium"])
            clf.fit(X_sub, group["medium"].values)
            self._clf_medium_map[int(large_val)] = clf

        # 4. 중분류 → 최빈 태그 매핑 테이블 (태그는 중분류에 종속적)
        self._tag_map = (
            label_df.groupby("medium")["tag"]
            .agg(lambda x: int(x.value_counts().index[0]))
            .to_dict()
        )

    def predict(
        self,
        X_refined: pd.Series,
        X_nouns: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = self._transform(X_refined, X_nouns)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            # 1. 대분류 예측
            large_pred = np.asarray(self._clf_large.predict(X), dtype=np.int64)

            # 2. 대분류 라우팅 → 해당 중분류 헤드만 호출
            medium_pred = np.full(len(large_pred), -1, dtype=np.int64)
            for large_val, med_val in self._medium_single_map.items():
                medium_pred[large_pred == large_val] = med_val     # 단일 중분류 상수 매핑
            for large_val, clf in self._clf_medium_map.items():
                mask = large_pred == large_val
                if mask.any():
                    medium_pred[mask] = clf.predict(X[mask])

            # 3. ★룰 레이어 — 모호 헤드어 반례 오버라이드 (활성 시)
            if self._rules is not None:
                large_pred, medium_pred, _ = self._rules.apply(
                    X_refined, large_pred, medium_pred
                )

            # 4. 태그: 중분류 → 최빈 태그 조회 (미등록 중분류는 -1)
            tag_pred = np.array(
                [self._tag_map.get(int(m), -1) for m in medium_pred], dtype=np.int64
            )
            return large_pred, medium_pred, tag_pred

    def predict_proba(
        self,
        X_refined: pd.Series,
        X_nouns: pd.Series,
    ) -> Tuple[np.ndarray, list, np.ndarray]:
        """
        확률 반환. LinearSVC 는 decision_function 을 softmax 로 근사한다.
        (LCPN 라우팅에는 predict 만 필요하므로 proba 는 분석용 보조 기능)
        """
        X = self._transform(X_refined, X_nouns)
        large_proba = _proba_of(self._clf_large, X)
        large_pred = large_proba.argmax(axis=1)

        medium_proba: list = [None] * len(large_pred)
        for large_val, clf in self._clf_medium_map.items():
            mask = large_pred == large_val
            if not mask.any():
                continue
            proba_sub = _proba_of(clf, X[mask])
            for local_i, global_i in enumerate(np.where(mask)[0]):
                medium_proba[global_i] = proba_sub[local_i]

        # 태그는 결정적 매핑이라 확률 정의가 없음 → predict 결과를 그대로 반환
        _, _, tag_pred = self.predict(X_refined, X_nouns)
        return large_proba, medium_proba, tag_pred

    def get_params(self) -> dict:
        return {
            **self._clf_cfg,
            **{f"lgbm_{k}": v for k, v in self._lgbm_params.items()},
            "feature_cfg": vars(self._feature_cfg),
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "TextPipelineClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ──────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────

    def _transform(self, X_refined: pd.Series, X_nouns: pd.Series):
        if self._features is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")
        return self._features.transform(X_refined, X_nouns)

    def _update_params(self, params: dict) -> None:
        """
        Optuna trial 파라미터를 세 그룹(feature/clf/lgbm)으로 라우팅한다.
        알 수 없는 키는 조용히 무시하지 않고 경고 로그를 남긴다(디버깅 편의).
        """
        feature_keys = set(vars(FeatureConfig()).keys())
        clf_keys = set(self._clf_cfg.keys())
        lgbm_keys = set(self._lgbm_params.keys())

        for k, v in params.items():
            if k == "ngram_range":                    # 구버전 키 호환 → char n-gram 으로 매핑
                self._feature_cfg.char_ngram_range = tuple(v)
            elif k in feature_keys:
                setattr(self._feature_cfg, k, tuple(v) if "ngram_range" in k else v)
            elif k in clf_keys:
                self._clf_cfg[k] = v
            elif k in lgbm_keys:
                self._lgbm_params[k] = v
            else:
                logger.warning("알 수 없는 파라미터 무시: %s=%r", k, v)

        # class_weight 변경은 LGBM 파라미터에도 반영해야 일관됨
        self._lgbm_params["class_weight"] = self._clf_cfg["class_weight"]


# ══════════════════════════════════════════════════════════════════
# 보조 유틸
# ══════════════════════════════════════════════════════════════════

def _proba_of(clf, X) -> np.ndarray:
    """
    분류기의 확률 추정치를 반환한다.
    - predict_proba 지원 (LogReg/CNB/LGBM/앙상블): 그대로 사용
    - LinearSVC: decision_function 을 softmax 로 근사 (엄밀 확률 아님, 순위 보존)
    """
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    scores = clf.decision_function(X)
    if scores.ndim == 1:                       # 이진 분류 시 (n,) → (n, 2)
        scores = np.stack([-scores, scores], axis=1)
    return softmax(scores, axis=1)


class _SoftVotingPair:
    """
    LogReg + LightGBM soft-voting 경량 래퍼.

    sklearn VotingClassifier 대신 자체 구현한 이유:
      - 희소 행렬 + LGBM 조합에서 불필요한 재검증/복사 비용 회피
      - classes_ 정렬을 두 모델 공통 라벨 공간으로 명시적으로 맞춤
    """

    def __init__(self, logreg, lgbm) -> None:
        if lgbm is None:
            raise ImportError("ensemble 모드에는 lightgbm 이 필요합니다.")
        self._logreg = logreg
        self._lgbm = lgbm
        self.classes_: np.ndarray | None = None

    def fit(self, X, y):
        self._logreg.fit(X, y)
        self._lgbm.fit(X, y)
        # 두 모델 모두 정렬된 고유 라벨을 classes_ 로 갖지만 방어적으로 통일
        self.classes_ = np.asarray(sorted(set(np.asarray(y).tolist())))
        return self

    def predict_proba(self, X) -> np.ndarray:
        # 두 모델의 classes_ 가 동일 정렬(np.unique)이므로 단순 평균 가능
        return (self._logreg.predict_proba(X) + self._lgbm.predict_proba(X)) / 2.0

    def predict(self, X) -> np.ndarray:
        idx = self.predict_proba(X).argmax(axis=1)
        return self.classes_[idx]
