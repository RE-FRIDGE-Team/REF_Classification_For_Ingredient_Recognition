"""
데이터 로드 및 KFold 분할 유틸리티.

CSV / XLSX 파일을 읽어 표준 스키마로 정규화하고,
experiment.yaml cv.strategy 에 따라 Fold 인덱스를 반환한다.

변경점(2026-xx):
  1. load_data 가 is_augmented 컬럼을 유지 (없으면 "0" 으로 채워 원본 취급).
  2. make_folds_orig_only 추가 — 원본(is_augmented==0)만으로 fold 생성.
     증강은 fold split 에 넣지 않고, cv_evaluate 에서 train 에만 합류시킨다.

지원 전략:
    GroupKFold           - 브랜드 기준 분리 (데이터 누수 방지)
    StratifiedGroupKFold - 브랜드 기준 분리 + 클래스 비율 유지
    StratifiedKFold      - 클래스 비율만 유지 (브랜드 분리 없음)
    KFold                - 순수 랜덤 (비교 베이스라인용)
"""

import logging
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

logger = logging.getLogger(__name__)

CVStrategy = Literal["GroupKFold", "StratifiedGroupKFold", "StratifiedKFold", "KFold"]


def load_data(
    input_path: str,
    col_map: dict[str, str],
    exclude_large: list[str] | None = None,
    exclude_tag: list[str] | None = None,
) -> pd.DataFrame:
    """
    CSV 또는 XLSX 파일을 읽어 표준 컬럼명으로 반환한다.

    Returns:
        DataFrame with columns:
            product_name, large_category, medium_category,
            category_tag, brand_name, is_augmented
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"입력 파일 없음: {input_path}")

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
        logger.info("XLSX 로드: %s (%d행)", path.name, len(df))
    elif suffix == ".csv":
        # utf-8-sig: BOM 이 있으면 제거하고, 없으면 일반 utf-8 과 동일하게 동작.
        # (증강 CSV 가 UTF-8 BOM 으로 저장되어 첫 컬럼명이 '\ufeffproduct_name'이
        #  되는 사고를 방지 — 기존 utf-8 지정의 상위 호환)
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        logger.info("CSV 로드: %s (%d행)", path.name, len(df))
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {suffix}. CSV 또는 XLSX만 허용.")

    # 컬럼 매핑: 내부 표준명 → 실제 파일 컬럼명
    reverse_map = {v: k for k, v in col_map.items()}
    missing = [v for v in col_map.values() if v not in df.columns]
    if missing:
        raise ValueError(f"파일에 없는 컬럼: {missing}\n실제 컬럼: {list(df.columns)}")

    df = df.rename(columns=reverse_map)

    # is_augmented 는 col_map 에 없어도 CSV 에 있으면 그대로 유지된다.
    # 없으면(구버전 CSV) 전부 원본("0")으로 채운다.
    if "is_augmented" not in df.columns:
        df["is_augmented"] = "0"
        logger.warning("is_augmented 컬럼이 없어 전부 원본(0)으로 처리합니다.")

    # 표준 컬럼만 선택 (is_augmented 포함)
    keep = ["product_name", "large_category", "medium_category",
            "category_tag", "brand_name", "is_augmented"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # 결측치 처리
    before = len(df)
    df = df.dropna(subset=["product_name", "large_category", "medium_category"])
    df["brand_name"]    = df["brand_name"].fillna("unknown")
    df["category_tag"]  = df["category_tag"].fillna("UNKNOWN")
    df["is_augmented"]  = df["is_augmented"].fillna("0").astype(str)
    logger.info("결측치 제거: %d행 → %d행", before, len(df))

    # 제외 카테고리 필터
    if exclude_large:
        df = df[~df["large_category"].isin(exclude_large)]
    if exclude_tag:
        df = df[~df["category_tag"].isin(exclude_tag)]

    df = df.reset_index(drop=True)

    n_orig = int((df["is_augmented"] == "0").sum())
    n_aug  = int((df["is_augmented"] == "1").sum())
    logger.info(
        "최종 학습 데이터: %d행 (원본 %d / 증강 %d), 대분류 %d종, 중분류 %d종, 태그 %d종",
        len(df), n_orig, n_aug,
        df["large_category"].nunique(),
        df["medium_category"].nunique(),
        df["category_tag"].nunique(),
    )
    return df


# ──────────────────────────────────────────────────────────────────
# CV 전략 팩토리
# ──────────────────────────────────────────────────────────────────

def make_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "brand_name",
    target_col: str = "large_category",
    strategy: CVStrategy = "GroupKFold",
    seed: int = 42,
) -> list[Tuple[list[int], list[int]]]:
    """
    (참고용/구버전) 전체 df 로 CV fold 를 만든다.
    증강 train-only 실험에는 make_folds_orig_only 를 사용할 것.
    """
    groups = df[group_col].values
    y      = df[target_col].values

    splitter = _make_splitter(strategy, n_splits, seed)
    split_kwargs = _make_split_kwargs(strategy, df, groups, y)

    folds: list[Tuple[list[int], list[int]]] = []
    for fold_idx, (tr, va) in enumerate(splitter.split(**split_kwargs)):
        folds.append((tr.tolist(), va.tolist()))
        logger.debug(
            "[%s] Fold %d — train: %d, val: %d, val 브랜드 수: %d, val 클래스 수: %d",
            strategy, fold_idx + 1, len(tr), len(va),
            len(set(groups[va])), len(set(y[va])),
        )

    _log_fold_summary(strategy, folds, groups, y)
    return folds


def make_folds_orig_only(
    df: pd.DataFrame,
    n_splits: int = 5,
    target_col: str = "large_category",
    group_col: str = "brand_name",
    strategy: CVStrategy = "StratifiedKFold",
    seed: int = 42,
) -> list[Tuple[list[int], list[int]]]:
    """
    원본(is_augmented=="0")만으로 fold 를 만든다. 증강은 split 에 넣지 않는다.
    반환하는 인덱스는 '전체 df 기준 위치 인덱스'라 df.iloc 에 그대로 쓸 수 있다.

    strategy:
      - "StratifiedKFold"     : 클래스 비율만 유지 (권장 — 원본만 나누므로 브랜드 그룹 불필요)
      - "GroupKFold"          : 브랜드로 분리 (원본만이라 unknown 붕괴 없음)
      - "StratifiedGroupKFold": 브랜드 분리 + 클래스 비율
      - "KFold"               : 순수 랜덤
    """
    if "is_augmented" not in df.columns:
        logger.warning("is_augmented 없음 → 전체 df 로 fold 생성(= make_folds 와 동일).")
        orig_pos = np.arange(len(df))
    else:
        orig_pos = np.where(df["is_augmented"].astype(str) == "0")[0]

    orig   = df.iloc[orig_pos]
    y      = orig[target_col].values
    groups = orig[group_col].values if group_col in orig.columns else None

    splitter = _make_splitter(strategy, n_splits, seed)

    folds: list[Tuple[list[int], list[int]]] = []
    if strategy == "GroupKFold":
        it = splitter.split(orig_pos, groups=groups)
    elif strategy == "StratifiedGroupKFold":
        it = splitter.split(orig_pos, y=y, groups=groups)
    elif strategy == "StratifiedKFold":
        it = splitter.split(orig_pos, y=y)
    else:  # KFold
        it = splitter.split(orig_pos)

    for fold_idx, (tr_local, va_local) in enumerate(it):
        # local(orig 내부) 인덱스 → 전체 df 위치 인덱스로 환원
        tr_global = orig_pos[tr_local].tolist()
        va_global = orig_pos[va_local].tolist()
        folds.append((tr_global, va_global))
        logger.debug(
            "[orig-only/%s] Fold %d — 원본 train: %d, valid: %d, valid 클래스: %d",
            strategy, fold_idx + 1, len(tr_global), len(va_global), len(set(y[va_local])),
        )

    n_orig = len(orig_pos)
    logger.info("[orig-only/%s] %d-Fold 구성 완료 (원본 %d행 기준, 증강은 train 에서만 합류)",
                strategy, n_splits, n_orig)
    return folds


def _make_splitter(strategy: CVStrategy, n_splits: int, seed: int):
    match strategy:
        case "GroupKFold":
            return GroupKFold(n_splits=n_splits)
        case "StratifiedGroupKFold":
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        case "StratifiedKFold":
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        case "KFold":
            return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        case _:
            raise ValueError(
                f"지원하지 않는 CV 전략: '{strategy}'. "
                f"선택 가능: GroupKFold, StratifiedGroupKFold, StratifiedKFold, KFold"
            )


def _make_split_kwargs(strategy: CVStrategy, df: pd.DataFrame, groups, y) -> dict:
    X = df
    match strategy:
        case "GroupKFold":
            return {"X": X, "groups": groups}
        case "StratifiedGroupKFold":
            return {"X": X, "y": y, "groups": groups}
        case "StratifiedKFold":
            return {"X": X, "y": y}
        case "KFold":
            return {"X": X}
        case _:
            return {"X": X}


def _log_fold_summary(strategy, folds, groups, y) -> None:
    val_sizes    = [len(va) for _, va in folds]
    val_brands   = [len(set(groups[va])) for _, va in folds]
    val_classes  = [len(set(y[va])) for _, va in folds]
    total_classes = len(set(y))

    logger.info(
        "[%s] %d-Fold 구성 완료\n  val 크기: %s (mean=%.0f)\n"
        "  val 브랜드 수: %s\n  val 클래스 수: %s / 전체 %d",
        strategy, len(folds), val_sizes, np.mean(val_sizes),
        val_brands, val_classes, total_classes,
    )
    if strategy == "KFold":
        logger.warning("[KFold] 브랜드 분리 없음 — 누수로 낙관적 측정 가능. 베이스라인 비교용만 권장.")
    if strategy == "StratifiedKFold":
        logger.warning("[StratifiedKFold] 클래스 비율 유지, 브랜드 누수 가능(원본만 나누면 영향 작음).")
    if strategy == "GroupKFold":
        missing_cls = [
            f"Fold {i+1}: {total_classes - c}개 누락"
            for i, (_, va) in enumerate(folds)
            if (c := len(set(y[va]))) < total_classes
        ]
        if missing_cls:
            logger.warning("[GroupKFold] val 클래스 누락 — %s. StratifiedGroupKFold 전환 권장.",
                           ", ".join(missing_cls))


def dedup_near_duplicates(
    df: pd.DataFrame,
    text_col: str = "refined_text",
) -> pd.DataFrame:
    """
    제품명 기준 근접중복 행을 제거한다 (★2026-07 신규).

    배경: 같은 제품이 용량/구성만 다르게 여러 번 수집되면 정제 후
          refined_text 가 사실상 동일해지고, CV에서 train/valid 양쪽에
          걸리면 점수가 낙관적으로 뻥튀기된다(누수의 한 형태).

    정규화 키: 소문자화 + 공백/특수문자 제거 → 완전 일치 행 중 첫 행만 유지.
    ※ 원본(is_augmented==0) 행끼리만 dedup 한다. 증강 행은 evaluate.py 의
      leak_guard 가 valid 원본과의 중복을 fold 별로 이미 차단하므로 유지.

    Args:
        df:       전처리 완료 DataFrame (refined_text 필요)
        text_col: 중복 판정에 사용할 텍스트 컬럼

    Returns:
        중복 제거된 DataFrame (index reset)
    """
    if text_col not in df.columns:
        logger.warning("dedup 스킵 — %s 컬럼 없음", text_col)
        return df

    # 정규화 키 생성: 대소문자·공백·기호 차이를 무시하는 canonical form
    norm_key = (
        df[text_col].fillna("")
        .str.lower()
        .str.replace(r"[\s\W_]+", "", regex=True)
    )

    is_orig = df.get("is_augmented", pd.Series(["0"] * len(df))).astype(str) == "0"

    # 원본 행에서만 첫 등장 이후의 중복을 표시 (증강 행은 대상 외)
    dup_mask = norm_key.duplicated(keep="first") & is_orig

    before = len(df)
    out = df[~dup_mask].reset_index(drop=True)
    logger.info("근접중복 제거: %d행 → %d행 (원본 중복 %d행 삭제)",
                before, len(out), int(dup_mask.sum()))
    return out


def data_summary(df: pd.DataFrame) -> dict:
    """학습 데이터 통계 요약 딕셔너리 반환 (HTML 리포트용)."""
    return {
        "n_samples": len(df),
        "n_orig":    int((df.get("is_augmented", pd.Series(["0"] * len(df))).astype(str) == "0").sum()),
        "n_aug":     int((df.get("is_augmented", pd.Series(["0"] * len(df))).astype(str) == "1").sum()),
        "n_large":   df["large_category"].nunique(),
        "n_medium":  df["medium_category"].nunique(),
        "n_tag":     df["category_tag"].nunique(),
        "n_brands":  df["brand_name"].nunique(),
        "large_dist": df["large_category"].value_counts().to_dict(),
        "tag_dist":   df["category_tag"].value_counts().to_dict(),
    }