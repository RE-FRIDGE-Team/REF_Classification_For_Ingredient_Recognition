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
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
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


def build_oof_table(
    details: list[FoldDetail],
    df: pd.DataFrame,
    label_encoders: dict,
) -> pd.DataFrame:
    """
    전 fold OOF(out-of-fold) 예측 결합 테이블 생성 — 오류 분석의 단일 소스.

    각 원본 샘플은 전체 CV에서 정확히 한 번 valid 에 등장하므로, 전 fold 를
    결합하면 데이터 전량에 대한 예측표가 완성되는 구조 (기존 fold 0 단독
    분석 대비 표본 5배 확보 + 표본 편향 제거).

    Returns:
        columns = [fold, product_name, refined_text,
                   true_large, pred_large, true_medium, pred_medium,
                   true_tag, pred_tag]  — 라벨은 전부 문자열 복원 상태.
    """
    le_l, le_m, le_t = label_encoders["large"], label_encoders["medium"], label_encoders["tag"]
    frames = []
    for d in details:
        idx = np.asarray(d.val_indices)
        sub = df.iloc[idx]
        frames.append(pd.DataFrame({
            "fold":         d.fold_idx,
            "product_name": sub.get("product_name", pd.Series([""] * len(idx))).values,
            "refined_text": sub.get("refined_text", pd.Series([""] * len(idx))).values,
            "true_large":   le_l.inverse_transform(d.true_large),
            "pred_large":   le_l.inverse_transform(d.pred_large),
            "true_medium":  le_m.inverse_transform(d.true_medium),
            "pred_medium":  le_m.inverse_transform(d.pred_medium),
            "true_tag":     le_t.inverse_transform(d.true_tag),
            "pred_tag":     le_t.inverse_transform(d.pred_tag),
        }))
    return pd.concat(frames, ignore_index=True)


def error_examples(
    details: list[FoldDetail],
    df: pd.DataFrame,
    label_encoders: dict,
    n: int = 5000,
) -> pd.DataFrame:
    """
    대분류 또는 중분류가 틀린 OOF 오류 테이블 반환 (HTML 리포트용).

    ★개편: fold 0 대분류 한정 → 전 fold OOF + 중분류 컬럼 + 오류 유형 라벨.
    error_type: 대분류오류 / 중분류오류(대분류정답) / 대분류오류·중분류정답
    """
    oof = build_oof_table(details, df, label_encoders)
    large_ok  = oof["true_large"] == oof["pred_large"]
    medium_ok = oof["true_medium"] == oof["pred_medium"]

    err = oof[~large_ok | ~medium_ok].copy()
    err_large_ok  = err["true_large"] == err["pred_large"]
    err_medium_ok = err["true_medium"] == err["pred_medium"]
    err["error_type"] = np.select(
        [~err_large_ok & ~err_medium_ok, err_large_ok & ~err_medium_ok],
        ["대분류오류→중분류오류", "중분류오류(대분류정답)"],
        default="대분류오류·중분류정답",
    )
    cols = ["product_name", "refined_text", "true_large", "pred_large",
            "true_medium", "pred_medium", "error_type", "fold"]
    return err[cols].head(n)


def _rank_misclassified(oof: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    task(large/medium) 기준 '가장 많이 오분류한 카테고리 순위' 테이블 생성.

    columns = [카테고리, 오분류수, 전체수, 오분류율, 최다혼동대상(→예측, 건수)]
    """
    t, p = f"true_{task}", f"pred_{task}"
    err = oof[oof[t] != oof[p]]
    if err.empty:
        return pd.DataFrame(columns=["카테고리", "오분류수", "전체수", "오분류율", "최다혼동대상"])

    total = oof.groupby(t).size()
    rows = []
    for cat, grp in err.groupby(t):
        top = grp[p].value_counts()
        top_target, top_cnt = top.index[0], int(top.iloc[0])
        rows.append({
            "카테고리":    cat,
            "오분류수":    len(grp),
            "전체수":      int(total[cat]),
            "오분류율":    round(len(grp) / total[cat], 4),
            "최다혼동대상": f"→ {top_target} ({top_cnt}건)",
        })
    return pd.DataFrame(rows).sort_values("오분류수", ascending=False).reset_index(drop=True)


def _task_metrics(y_true, y_pred) -> dict[str, float]:
    """단일 태스크의 F1 외 보조 분류 지표 묶음 계산."""
    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1":          f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1":       f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc":               matthews_corrcoef(y_true, y_pred),
        "cohen_kappa":       cohen_kappa_score(y_true, y_pred),
    }


def hierarchical_error_stats(oof: pd.DataFrame) -> dict:
    """
    LCPN 계층 관점의 오류 통계 일괄 계산 (HTML 'Other Statistics' 섹션 소스).

    Returns:
        {
          "rank_large":  대분류 오분류 순위 DataFrame,
          "rank_medium": 중분류 오분류 순위 DataFrame,
          "case_both_wrong":            대분류부터 틀려 중분류도 틀린 목록,
          "case_large_ok_medium_wrong": 대분류 정답·중분류 오답 목록 (최중요),
          "case_large_wrong_medium_ok": 대분류 오답·중분류 정답 목록,
          "tag_inconsistent":  중분류 정답인데 태그가 틀린 목록
                               (비어 있으면 태그 오류 = 순수 중분류 오류임이 입증),
          "metrics": {task: {지표: 값}},   # F1 외 보조 지표
          "hier_exact_large_medium": 대·중 동시 정답률,
          "hier_exact_all":          대·중·태그 동시 정답률,
        }
    """
    large_ok  = oof["true_large"]  == oof["pred_large"]
    medium_ok = oof["true_medium"] == oof["pred_medium"]
    tag_ok    = oof["true_tag"]    == oof["pred_tag"]

    case_cols = ["product_name", "refined_text",
                 "true_large", "pred_large", "true_medium", "pred_medium", "fold"]

    return {
        "rank_large":  _rank_misclassified(oof, "large"),
        "rank_medium": _rank_misclassified(oof, "medium"),
        "case_both_wrong":            oof[~large_ok & ~medium_ok][case_cols].reset_index(drop=True),
        "case_large_ok_medium_wrong": oof[large_ok & ~medium_ok][case_cols].reset_index(drop=True),
        "case_large_wrong_medium_ok": oof[~large_ok & medium_ok][case_cols].reset_index(drop=True),
        "tag_inconsistent": oof[medium_ok & ~tag_ok][
            case_cols[:2] + ["true_medium", "pred_medium", "true_tag", "pred_tag", "fold"]
        ].reset_index(drop=True),
        "metrics": {
            "large":  _task_metrics(oof["true_large"],  oof["pred_large"]),
            "medium": _task_metrics(oof["true_medium"], oof["pred_medium"]),
            "tag":    _task_metrics(oof["true_tag"],    oof["pred_tag"]),
        },
        "hier_exact_large_medium": float((large_ok & medium_ok).mean()),
        "hier_exact_all":          float((large_ok & medium_ok & tag_ok).mean()),
    }