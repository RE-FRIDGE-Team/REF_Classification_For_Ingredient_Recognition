"""
모델 2: FastText + KoNLPy + ML (LogReg / LightGBM).

Okt 명사 토큰화 → FastText 임베딩 학습 → 문장 벡터 (mean pooling)
→ 다운스트림 분류기 (LogReg 또는 LightGBM) × 3 타깃
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

try:
    from gensim.models import FastText
    _GENSIM_AVAILABLE = True
except ImportError:
    _GENSIM_AVAILABLE = False
    logger.error("gensim 미설치. pip install gensim")

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

from .base import BaseClassifier


class FastTextKonlpyClassifier(BaseClassifier):
    """
    FastText 임베딩 + 다운스트림 ML 분류기.

    nouns_text (Okt 명사)로 FastText를 학습하고
    문장 벡터 mean-pooling으로 LogReg / LightGBM 3-head 분류.
    """

    def __init__(
        self,
        # FastText
        vector_size: int = 100,
        window: int = 3,
        min_count: int = 1,
        epochs: int = 10,
        sg: int = 1,           # 0=CBOW, 1=SkipGram
        negative: int = 5,
        # 다운스트림 분류기
        classifier_type: Literal["logreg", "lgbm"] = "logreg",
        # LogReg
        C: float = 1.0,
        max_iter: int = 500,
        # LightGBM (classifier_type="lgbm" 시 사용)
        n_estimators: int = 300,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        # 공통
        seed: int = 42,
    ) -> None:
        if not _GENSIM_AVAILABLE:
            raise ImportError("gensim 패키지가 필요합니다. pip install gensim")

        self._ft_params = dict(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            sg=sg,
            negative=negative,
            seed=seed,
            workers=4,
        )
        self._clf_type = classifier_type
        self._logreg_params = dict(C=C, max_iter=max_iter, random_state=seed, n_jobs=-1)
        self._lgbm_params   = dict(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

        self._ft_model = None
        self._clf_large  = None
        self._clf_medium = None
        self._clf_tag    = None

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
        if params:
            self._update_params(params)

        # FastText 학습 (nouns_text 기준)
        corpus = [text.split() for text in X_nouns.fillna("")]
        # 너무 짧은 문장 보완: refined_text fallback
        for i, tokens in enumerate(corpus):
            if not tokens:
                corpus[i] = X_refined.iloc[i].split() if pd.notna(X_refined.iloc[i]) else [""]

        logger.debug("FastText 학습 시작: vocab 후보 %d문장", len(corpus))
        self._ft_model = FastText(**self._ft_params)
        self._ft_model.build_vocab(corpus_iterable=corpus)
        self._ft_model.train(
            corpus_iterable=corpus,
            total_examples=len(corpus),
            epochs=self._ft_params["epochs"],
        )
        logger.debug("FastText 완료: vocab_size=%d", len(self._ft_model.wv))

        # 문장 벡터 생성
        X_vec = np.vstack([self._sentence_vector(tokens) for tokens in corpus])

        # 다운스트림 분류기 학습
        self._clf_large  = self._make_clf()
        self._clf_medium = self._make_clf()
        self._clf_tag    = self._make_clf()

        self._clf_large.fit(X_vec, y_large)
        self._clf_medium.fit(X_vec, y_medium)
        self._clf_tag.fit(X_vec, y_tag)

    def predict(
        self,
        X_refined: pd.Series,
        X_nouns: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_vec = self._vectorize(X_refined, X_nouns)
        return (
            self._clf_large.predict(X_vec),
            self._clf_medium.predict(X_vec),
            self._clf_tag.predict(X_vec),
        )

    def predict_proba(
        self,
        X_refined: pd.Series,
        X_nouns: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_vec = self._vectorize(X_refined, X_nouns)
        return (
            self._clf_large.predict_proba(X_vec),
            self._clf_medium.predict_proba(X_vec),
            self._clf_tag.predict_proba(X_vec),
        )

    def get_params(self) -> dict:
        return {
            **self._ft_params,
            "classifier_type": self._clf_type,
            **self._logreg_params,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "FastTextKonlpyClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    def _sentence_vector(self, tokens: list[str]) -> np.ndarray:
        """토큰 목록 → 벡터 평균 (FastText OOV 강인)."""
        vecs = [self._ft_model.wv[t] for t in tokens if t in self._ft_model.wv]
        if vecs:
            return np.mean(vecs, axis=0)
        # 빈 문장: 영벡터
        return np.zeros(self._ft_params["vector_size"])

    def _vectorize(self, X_refined: pd.Series, X_nouns: pd.Series) -> np.ndarray:
        if self._ft_model is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")
        corpus = [text.split() for text in X_nouns.fillna("")]
        for i, tokens in enumerate(corpus):
            if not tokens:
                corpus[i] = X_refined.iloc[i].split() if pd.notna(X_refined.iloc[i]) else [""]
        return np.vstack([self._sentence_vector(tokens) for tokens in corpus])

    def _make_clf(self):
        if self._clf_type == "lgbm":
            if not _LGB_AVAILABLE:
                raise ImportError("lightgbm 패키지가 필요합니다.")
            return lgb.LGBMClassifier(**self._lgbm_params)
        return LogisticRegression(**self._logreg_params)

    def _update_params(self, params: dict) -> None:
        ft_keys    = set(self._ft_params)
        logreg_keys = set(self._logreg_params)
        lgbm_keys  = set(self._lgbm_params)
        for k, v in params.items():
            if k == "classifier_type":
                self._clf_type = v
            elif k in ft_keys:
                self._ft_params[k] = v
            elif k in logreg_keys:
                self._logreg_params[k] = v
            elif k in lgbm_keys:
                self._lgbm_params[k] = v
