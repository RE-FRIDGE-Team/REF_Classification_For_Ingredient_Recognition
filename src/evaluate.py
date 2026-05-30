"""
GroupKFold CV 평가 유틸리티.

최적 파라미터로 학습된 모델을 5-Fold GroupKFold로 평가하여
macro_F1, accuracy, 추론 시간, per-class F1을 집계한다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from .models.base import BaseClassifier, CVResult

logger = logging.getLogger(__name__)


@dataclass
class FoldDetail:
    """단일 Fold 상세 결과 (오류 분석용)."""
    fold_idx:       int
    val_indices:    list[int]
    pred_large:     np.ndarray
    pred_medium:    np.ndarray
    pred_tag:       np.ndarray
    true_large:     np.ndarray
    true_medium:    np.ndarray
    true_tag:       np.ndarray
    large_f1:       float
    medium_f1:      float
    tag_f1:         float


def cv_evaluate(
    model_cls,
    best_params: dict,
    extra_kwargs: dict,
    df: pd.DataFrame,
    folds: list[Tuple[list[int], list[int]]],
) -> Tuple[CVResult, list[FoldDetail]]:
    """
    최적 파라미터로 5-Fold GroupKFold CV를 실행한다.

    Args:
        model_cls:    BaseClassifier 서브클래스
        best_params:  Optuna best_params
        extra_kwargs: 모델 생성자 추가 인수 (n_large, n_medium 등)
        df:           fit_transform() 완료 DataFrame
        folds:        make_folds() 반환 리스트

    Returns:
        (CVResult, list[FoldDetail])
    """
    result  = CVResult()
    details: list[FoldDetail] = []

    train_times: list[float] = []
    infer_times: list[float] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        tr = df.iloc[tr_idx]
        va = df.iloc[va_idx]

        model: BaseClassifier = model_cls(**extra_kwargs)

        # 학습 시간 측정
        t0 = time.perf_counter()
        model.fit(
            tr["refined_text"], tr["nouns_text"],
            tr["label_large"].values,
            tr["label_medium"].values,
            tr["label_tag"].values,
            **best_params,
        )
        train_times.append(time.perf_counter() - t0)

        # 예측
        pred_l, pred_m, pred_t = model.predict(va["refined_text"], va["nouns_text"])
        true_l = va["label_large"].values
        true_m = va["label_medium"].values
        true_t = va["label_tag"].values

        print("- pred_m : ", pred_m[:5])
        print("- true_m : ", true_m[:5])
        print("- pred_type : ", pred_m.dtype)
        print("- true_type : ", true_m.dtype)

        # 단일 샘플 추론 시간
        t1 = time.perf_counter()
        model.predict(va["refined_text"].iloc[:1], va["nouns_text"].iloc[:1])
        infer_times.append((time.perf_counter() - t1) * 1000)

        # 지표
        large_f1  = f1_score(true_l, pred_l, average="macro", zero_division=0)
        medium_f1 = f1_score(true_m, pred_m, average="macro", zero_division=0)
        tag_f1    = f1_score(true_t, pred_t, average="macro", zero_division=0)

        result.fold_large_f1.append(large_f1)
        result.fold_medium_f1.append(medium_f1)
        result.fold_tag_f1.append(tag_f1)

        details.append(FoldDetail(
            fold_idx=fold_idx,
            val_indices=va_idx,
            pred_large=pred_l, pred_medium=pred_m, pred_tag=pred_t,
            true_large=true_l, true_medium=true_m, true_tag=true_t,
            large_f1=large_f1, medium_f1=medium_f1, tag_f1=tag_f1,
        ))

        logger.info(
            "Fold %d/%d — large_F1=%.4f, medium_F1=%.4f, tag_F1=%.4f",
            fold_idx + 1, len(folds), large_f1, medium_f1, tag_f1
        )

    # 집계
    result.aggregate_folds()
    result.train_time     = float(np.mean(train_times))
    result.infer_time_ms  = float(np.mean(infer_times))
    # 대분류 accuracy (마지막 fold 기준 근사 — 전체 CV 집계 시 정확도는 per-fold mean)
    large_accs = []
    for d in details:
        large_accs.append(accuracy_score(d.true_large, d.pred_large))
    result.large_acc = float(np.mean(large_accs))

    logger.info(
        "CV 완료 — large_F1=%.4f±%.4f, medium_F1=%.4f±%.4f, tag_F1=%.4f±%.4f, "
        "train=%.1fs, infer=%.2fms",
        result.large_f1, result.large_f1_std,
        result.medium_f1, result.medium_f1_std,
        result.tag_f1, result.tag_f1_std,
        result.train_time, result.infer_time_ms,
    )
    return result, details


def per_class_f1_report(
    details: list[FoldDetail],
    label_encoder,
    task: str = "large",
) -> pd.DataFrame:
    """
    Fold 0 기준 대분류별 per-class F1 DataFrame 반환 (히트맵용).

    Args:
        details:       cv_evaluate() 반환 FoldDetail 목록
        label_encoder: 해당 task의 LabelEncoder
        task:          "large" | "medium" | "tag"

    Returns:
        DataFrame with columns: [class_name, f1, support]
    """
    d = details[0]
    true = getattr(d, f"true_{task}")
    pred = getattr(d, f"pred_{task}")
    report = classification_report(
        true, pred,
        labels=list(range(len(label_encoder.classes_))),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for cls_name in label_encoder.classes_:
        if cls_name in report:
            rows.append({
                "class": cls_name,
                "f1": report[cls_name]["f1-score"],
                "precision": report[cls_name]["precision"],
                "recall": report[cls_name]["recall"],
                "support": report[cls_name]["support"],
            })
    return pd.DataFrame(rows).sort_values("f1", ascending=False)


def error_examples(
    details: list[FoldDetail],
    df: pd.DataFrame,
    label_encoder_large,
    n: int = 20,
) -> pd.DataFrame:
    """
    Fold 0 기준 대분류 오분류 예시 상위 n개 반환 (HTML 리포트용).

    Returns:
        DataFrame with columns: [product_name, true_large, pred_large, refined_text]
    """
    d = details[0]
    mask = d.true_large != d.pred_large
    err_indices = np.array(d.val_indices)[mask][:n]

    rows = []
    for i, idx in enumerate(err_indices):
        row = df.iloc[idx]
        rows.append({
            "product_name": row.get("product_name", ""),
            "refined_text": row.get("refined_text", ""),
            "true_large":   label_encoder_large.inverse_transform([d.true_large[mask][i]])[0],
            "pred_large":   label_encoder_large.inverse_transform([d.pred_large[mask][i]])[0],
        })
    return pd.DataFrame(rows)
