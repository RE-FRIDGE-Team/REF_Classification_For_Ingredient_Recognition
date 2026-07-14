"""
GroupKFold CV 평가 유틸리티.

변경점(2026-xx):
  1. 증강 데이터(is_augmented==1)는 train에만 사용, valid는 원본(==0)만.
     - folds 는 make_folds_orig_only() 로 만든 '원본 전용' fold 여야 한다.
     - valid 원본과 refined_text 가 겹치는 증강은 해당 fold train 에서 제외(누수 차단).
  2. 대/중/태그 3개 타깃에 대한 Confusion Matrix 저장 함수 추가
     (save_confusion_matrices).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .models.base import BaseClassifier, CVResult

logger = logging.getLogger(__name__)


@dataclass
class FoldDetail:
    """단일 Fold 상세 결과 (오류 분석·혼동행렬용)."""
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
    leak_guard: bool = True,
) -> Tuple[CVResult, list[FoldDetail]]:
    """
    최적 파라미터로 5-Fold CV를 실행한다.

    핵심: folds 는 '원본(is_augmented==0)만'으로 만들어진 것이어야 한다
          (make_folds_orig_only 사용). 증강은 아래에서 train 에만 합류한다.

    Args:
        model_cls:    BaseClassifier 서브클래스
        best_params:  Optuna best_params
        extra_kwargs: 모델 생성자 추가 인수 (n_large, n_medium 등)
        df:           fit_transform() 완료 DataFrame (is_augmented 컬럼 포함)
        folds:        make_folds_orig_only() 반환 리스트 (원본 인덱스)
        leak_guard:   valid 원본과 refined_text 가 겹치는 증강을 train 에서 제외

    Returns:
        (CVResult, list[FoldDetail])
    """
    result  = CVResult()
    details: list[FoldDetail] = []

    train_times: list[float] = []
    infer_times: list[float] = []

    # 증강/원본 분리 (is_augmented 없으면 전부 원본 취급 → 기존 동작과 동일)
    if "is_augmented" in df.columns:
        aug_mask = df["is_augmented"].astype(str) == "1"
        aug_df   = df[aug_mask]
    else:
        aug_df = df.iloc[0:0]  # 빈 프레임

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        tr_orig = df.iloc[tr_idx]      # 원본 train fold
        va      = df.iloc[va_idx]      # 원본 valid fold (증강 없음)

        # ── 증강을 train 에만 합류 (누수 차단) ──
        if len(aug_df) > 0:
            if leak_guard and "refined_text" in df.columns:
                va_refined = set(va["refined_text"])
                aug_train = aug_df[~aug_df["refined_text"].isin(va_refined)]
            else:
                aug_train = aug_df
            tr = pd.concat([tr_orig, aug_train], ignore_index=True)
        else:
            aug_train = aug_df
            tr = tr_orig

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

        # 예측 (valid = 원본만)
        pred_l, pred_m, pred_t = model.predict(va["refined_text"], va["nouns_text"])
        true_l = va["label_large"].values
        true_m = va["label_medium"].values
        true_t = va["label_tag"].values

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
            val_indices=list(va_idx),
            pred_large=pred_l, pred_medium=pred_m, pred_tag=pred_t,
            true_large=true_l, true_medium=true_m, true_tag=true_t,
            large_f1=large_f1, medium_f1=medium_f1, tag_f1=tag_f1,
        ))

        logger.info(
            "Fold %d/%d — train=%d(원본%d+증강%d), valid=%d(원본) | "
            "large_F1=%.4f, medium_F1=%.4f, tag_F1=%.4f",
            fold_idx + 1, len(folds),
            len(tr), len(tr_orig), len(aug_train), len(va),
            large_f1, medium_f1, tag_f1,
        )

    # 집계
    result.aggregate_folds()
    result.train_time     = float(np.mean(train_times))
    result.infer_time_ms  = float(np.mean(infer_times))
    large_accs = [accuracy_score(d.true_large, d.pred_large) for d in details]
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


# ──────────────────────────────────────────────────────────────
# Confusion Matrix (대/중/태그) — 전 fold 합산
# ──────────────────────────────────────────────────────────────

def _set_korean_font() -> None:
    """
    matplotlib 한글 폰트 설정.

    ★수정(2026-07): 기존 코드는 matplotlib.rc(font, family=...)가 폰트 부재 시
    예외를 던진다고 가정했지만, rc()는 검증 없이 항상 성공한다(경고는 렌더링
    시점에 발생). 그래서 폴백 루프가 무의미하게 첫 항목(Malgun Gothic —
    Windows 전용)에서 멈췄고, Linux 컨테이너에서 findfont WARNING 이 반복됐다.
    → font_manager 로 '실제 설치된' 폰트 목록을 조회해 존재하는 것만 채택한다.
    우선순위: NanumGothic(Docker 이미지 fonts-nanum) → Noto CJK →
              AppleGothic(Mac 로컬) → Malgun Gothic(Windows 로컬).
    """
    import matplotlib
    from matplotlib import font_manager as fm

    installed = {f.name for f in fm.fontManager.ttflist}
    for fam in ["NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP",
                "Noto Sans KR", "AppleGothic", "Malgun Gothic"]:
        if fam in installed:                     # 실제 존재하는 폰트만 채택
            matplotlib.rc("font", family=fam)
            break
    else:                                        # 한글 폰트가 전무한 환경
        logger.warning("한글 폰트 미발견 — 차트 한글이 깨질 수 있습니다. "
                       "(Docker: apt install fonts-nanum)")
    matplotlib.rcParams["axes.unicode_minus"] = False


def save_confusion_matrices(
    details: list[FoldDetail],
    label_encoders: dict,
    out_dir: str | Path,
    model_name: str = "model",
    normalize: bool = True,
) -> None:
    """
    대/중/태그 3개 타깃에 대해 전 fold 를 합산한 confusion matrix 를
    png + csv 로 out_dir 에 저장한다.

    Args:
        details:        cv_evaluate() 반환 FoldDetail 목록
        label_encoders: prep.label_encoders  ({"large":LE, "medium":LE, "tag":LE})
        out_dir:        저장 디렉토리
        model_name:     파일/타이틀용 모델명
        normalize:      True 면 행(실제 클래스) 기준 비율로 정규화하여 시각화
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _set_korean_font()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in ["large", "medium", "tag"]:
        le = label_encoders[task]
        classes = list(le.classes_)
        n = len(classes)
        labels = list(range(n))

        y_true = np.concatenate([getattr(d, f"true_{task}") for d in details])
        y_pred = np.concatenate([getattr(d, f"pred_{task}") for d in details])

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm = cm / row_sum

        # CSV (원시 카운트) — 클래스명 인덱스/컬럼
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(
            out_dir / f"confusion_{model_name}_{task}.csv", encoding="utf-8-sig"
        )

        # PNG heatmap
        cell = 0.42 if n > 20 else 0.6
        figsize = (max(6.0, n * cell + 3), max(5.0, n * cell + 2))
        fig, ax = plt.subplots(figsize=figsize)
        data = cm_norm if normalize else cm
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalize else None,
                       aspect="auto")

        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        fs = 7 if n > 20 else 9
        ax.set_xticklabels(classes, rotation=90, fontsize=fs)
        ax.set_yticklabels(classes, fontsize=fs)
        ax.set_xlabel("예측(pred)"); ax.set_ylabel("실제(true)")
        ax.set_title(f"{model_name} — {task} confusion matrix"
                     f"{' (row-normalized)' if normalize else ''}")

        # 클래스가 적을 때만 숫자 표기 (대분류·태그 등)
        if n <= 15:
            for i in range(n):
                for j in range(n):
                    v = cm[i, j]
                    if v > 0:
                        ax.text(j, i, int(v), ha="center", va="center",
                                fontsize=fs,
                                color="white" if cm_norm[i, j] > 0.5 else "black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(out_dir / f"confusion_{model_name}_{task}.png", dpi=130,
                    bbox_inches="tight")
        plt.close(fig)

    logger.info("Confusion matrix 저장 완료: %s (confusion_%s_[large|medium|tag].png/csv)",
                out_dir, model_name)


# ──────────────────────────────────────────────────────────────
# 기존 리포트 유틸 (변경 없음)
# ──────────────────────────────────────────────────────────────

def per_class_f1_report(
    details: list[FoldDetail],
    label_encoder,
    task: str = "large",
) -> pd.DataFrame:
    """Fold 0 기준 per-class F1 DataFrame 반환 (히트맵/HTML용)."""
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
    """Fold 0 기준 대분류 오분류 예시 상위 n개 반환 (HTML 리포트용)."""
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