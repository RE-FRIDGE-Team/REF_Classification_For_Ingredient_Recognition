"""
모델 패키지 (2026-07 개편).

변경점:
  - TextPipelineClassifier 신규 (구 TfidfLgbmClassifier 대체 — 모델 선택형)
  - FastTextKonlpyClassifier 삭제 (char n-gram 과 사실상 중복, 학습시간 대비 마진 없음)
  - KoElectraMultiTaskClassifier 유지 — 술 OOV 고유명사 구제용 후순위 옵션.
    싼 개선(전처리/피처/모델선택)을 모두 짜내 재베이스라인한 뒤에도
    +2~3점이 더 필요할 때만 --models koelectra 로 붙인다.
"""

from .base import BaseClassifier, CVResult
from .koelectra_multitask import KoElectraMultiTaskClassifier
from .text_classifier import TextPipelineClassifier

__all__ = [
    "BaseClassifier",
    "CVResult",
    "TextPipelineClassifier",
    "KoElectraMultiTaskClassifier",
]
