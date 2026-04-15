"""
모델 1: TF-IDF + LightGBM.

refined_text + nouns_text 이중 TF-IDF 피처를 결합하여
대/중분류, 카테고리태그를 각각 독립 LightGBM으로 분류한다.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False
    logger.error("lightgbm 미설치. pip install lightgbm")

from .base import BaseClassifier


class TfidfLgbmClassifier(BaseClassifier):
    """
    TF-IDF(refined + nouns) × LightGBM 3-head 분류기.

    각 타깃(대분류, 중분류, 카테고리태그)을 독립적인
    LightGBM 분류기로 학습한다.
    """

    def __init__(
        self,
        # TF-IDF
        ngram_range: tuple = (1, 2),
        max_features: int = 10000,
        min_df: int = 1,
        sublinear_tf: bool = True,
        # LightGBM
        n_estimators: int = 300,
        num_leaves: int = 63,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        min_child_samples: int = 20,
        colsample_bytree: float = 0.8,
        subsample: float = 0.8,
        reg_alpha: float = 0.01,
        reg_lambda: float = 0.01,
        seed: int = 42,
    ) -> None:
        if not _LGB_AVAILABLE:
            raise ImportError("lightgbm 패키지가 필요합니다.")

        self._tfidf_params = dict(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            analyzer="char_wb",   # 한국어는 char n-gram 효과적
        )
        self._lgbm_params = dict(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

        # 내부 컴포넌트 (fit 후 할당)
        self._vec_refined: TfidfVectorizer | None = None
        self._vec_nouns:   TfidfVectorizer | None = None
        self._clf_large:   lgb.LGBMClassifier | None = None
        self._clf_medium:  lgb.LGBMClassifier | None = None
        self._clf_tag:     lgb.LGBMClassifier | None = None

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
        # 런타임에 파라미터 오버라이드 (Optuna trial 파라미터 주입용)
        if params:
            self._update_params(params)

        self._vec_refined = TfidfVectorizer(**self._tfidf_params)
        self._vec_nouns   = TfidfVectorizer(**self._tfidf_params)

        X_r = self._vec_refined.fit_transform(X_refined.fillna(""))
        X_n = self._vec_nouns.fit_transform(X_nouns.fillna(""))
        X   = hstack([X_r, X_n])

        logger.debug("TF-IDF 피처 차원: %s", X.shape)

        self._clf_large  = lgb.LGBMClassifier(**self._lgbm_params)
        self._clf_medium = lgb.LGBMClassifier(**self._lgbm_params)
        self._clf_tag    = lgb.LGBMClassifier(**self._lgbm_params)

        self._clf_large.fit(X, y_large)
        self._clf_medium.fit(X, y_medium)
        self._clf_tag.fit(X, y_tag)

    def predict(
        self,
        X_refined: pd.Series,
        X_nouns: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = self._transform(X_refined, X_nouns)
        return (
            self._clf_large.predict(X),
            self._clf_medium.predict(X),
            self._clf_tag.predict(X),
        )

    def predict_proba(
        self,
        X_refined: pd.Series,
        X_nouns: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = self._transform(X_refined, X_nouns)
        return (
            self._clf_large.predict_proba(X),
            self._clf_medium.predict_proba(X),
            self._clf_tag.predict_proba(X),
        )

    def get_params(self) -> dict:
        return {**self._tfidf_params, **self._lgbm_params}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "TfidfLgbmClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    def _transform(self, X_refined: pd.Series, X_nouns: pd.Series):
        if self._vec_refined is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")
        X_r = self._vec_refined.transform(X_refined.fillna(""))
        X_n = self._vec_nouns.transform(X_nouns.fillna(""))
        return hstack([X_r, X_n])

    def _update_params(self, params: dict) -> None:
        """Optuna trial 파라미터를 내부 딕셔너리에 적용."""
        tfidf_keys = set(self._tfidf_params)
        lgbm_keys  = set(self._lgbm_params)
        for k, v in params.items():
            if k in tfidf_keys:
                self._tfidf_params[k] = v
            elif k in lgbm_keys:
                self._lgbm_params[k] = v
