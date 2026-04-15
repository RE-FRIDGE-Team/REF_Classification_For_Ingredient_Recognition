
"""
BaseClassifier ABC — 3개 모델의 공통 인터페이스.

모든 모델은 이 클래스를 상속하고 추상 메서드를 구현해야 한다.
타깃은 대분류(large) / 중분류(medium) / 카테고리태그(tag) 3개.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class CVResult:
    """단일 Fold 또는 최종 CV 평균 지표 컨테이너."""
    large_f1:   float = 0.0
    medium_f1:  float = 0.0
    tag_f1:     float = 0.0
    large_acc:  float = 0.0
    train_time: float = 0.0          # 학습 시간 (초)
    infer_time_ms: float = 0.0       # 단일 샘플 추론 시간 (ms)

    # 표준편차 (Fold 집계 시 사용)
    large_f1_std:  float = 0.0
    medium_f1_std: float = 0.0
    tag_f1_std:    float = 0.0

    # Fold별 상세 (집계 전 raw 데이터)
    fold_large_f1:  list[float] = field(default_factory=list)
    fold_medium_f1: list[float] = field(default_factory=list)
    fold_tag_f1:    list[float] = field(default_factory=list)

    def aggregate_folds(self) -> "CVResult":
        """fold_* 리스트로부터 mean/std를 계산하여 필드를 갱신한다."""
        if self.fold_large_f1:
            self.large_f1      = float(np.mean(self.fold_large_f1))
            self.large_f1_std  = float(np.std(self.fold_large_f1))
        if self.fold_medium_f1:
            self.medium_f1     = float(np.mean(self.fold_medium_f1))
            self.medium_f1_std = float(np.std(self.fold_medium_f1))
        if self.fold_tag_f1:
            self.tag_f1        = float(np.mean(self.fold_tag_f1))
            self.tag_f1_std    = float(np.std(self.fold_tag_f1))
        return self


class BaseClassifier(ABC):
    """
    RE:FRIDGE 분류기 공통 인터페이스.

    모든 구현체는 다음 추상 메서드를 구현해야 한다:
        fit()       — 학습
        predict()   — 예측 (3-tuple 반환)
        get_params() — 현재 하이퍼파라미터 딕셔너리 반환
    """

    # ──────────────────────────────────────────────────────────────
    # 추상 메서드
    # ──────────────────────────────────────────────────────────────

    @abstractmethod
    def fit(
        self,
        X_refined: pd.Series,
        X_nouns: pd.Series,
        y_large: np.ndarray,
        y_medium: np.ndarray,
        y_tag: np.ndarray,
        **params,
    ) -> None:
        """
        모델 학습.

        Args:
            X_refined: 정제된 제품명 시리즈
            X_nouns:   명사 추출 텍스트 시리즈
            y_large:   대분류 레이블 (int ndarray)
            y_medium:  중분류 레이블 (int ndarray)
            y_tag:     카테고리태그 레이블 (int ndarray)
        """

    @abstractmethod
    def predict(
        self,
        X_refined: pd.Series,
        X_nouns: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        예측 반환.

        Returns:
            (pred_large, pred_medium, pred_tag) — 각각 int ndarray
        """

    @abstractmethod
    def predict_proba(
        self,
        X_refined: pd.Series,
        X_nouns: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        확률 반환.

        Returns:
            (proba_large, proba_medium, proba_tag) — 각각 shape (n, n_classes)
        """

    @abstractmethod
    def get_params(self) -> dict:
        """현재 하이퍼파라미터 딕셔너리."""

    # ──────────────────────────────────────────────────────────────
    # 공통 유틸 (override 불필요)
    # ──────────────────────────────────────────────────────────────

    def timed_fit(
        self,
        X_refined: pd.Series,
        X_nouns: pd.Series,
        y_large: np.ndarray,
        y_medium: np.ndarray,
        y_tag: np.ndarray,
        **params,
    ) -> float:
        """fit()을 실행하고 소요 시간(초)을 반환한다."""
        t0 = time.perf_counter()
        self.fit(X_refined, X_nouns, y_large, y_medium, y_tag, **params)
        return time.perf_counter() - t0

    def timed_infer_ms(
        self,
        X_refined: pd.Series,
        X_nouns: pd.Series,
        n_warmup: int = 3,
    ) -> float:
        """단일 샘플 평균 추론 시간(ms)을 반환한다 (warmup 후 측정)."""
        single_r = X_refined.iloc[:1]
        single_n = X_nouns.iloc[:1]
        for _ in range(n_warmup):
            self.predict(single_r, single_n)
        t0 = time.perf_counter()
        self.predict(single_r, single_n)
        return (time.perf_counter() - t0) * 1000
