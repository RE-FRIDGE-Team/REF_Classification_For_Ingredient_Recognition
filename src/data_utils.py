"""
데이터 로드 및 KFold 분할 유틸리티.

CSV / XLSX 파일을 읽어 표준 스키마로 정규화하고,
experiment.yaml cv.strategy에 따라 Fold 인덱스를 반환한다.

지원 전략:
    GroupKFold           - 브랜드 기준 분리 (데이터 누수 방지, 기본값)
    StratifiedGroupKFold - 브랜드 기준 분리 + 클래스 비율 유지
    StratifiedKFold      - 클래스 비율만 유지 (브랜드 분리 없음)
    KFold                - 순수 랜덤 (비교 베이스라인용)
"""

import logging
from pathlib import Path
from typing import Literal, Tuple

import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

logger = logging.getLogger(__name__)

# 지원하는 전략 타입
CVStrategy = Literal["GroupKFold", "StratifiedGroupKFold", "StratifiedKFold", "KFold"]


def load_data(
    input_path: str,
    col_map: dict[str, str],
    exclude_large: list[str] | None = None,
    exclude_tag: list[str] | None = None,
) -> pd.DataFrame:
    """
    CSV 또는 XLSX 파일을 읽어 표준 컬럼명으로 반환한다.

    Args:
        input_path:    CSV 또는 XLSX 경로
        col_map:       experiment.yaml data.input_columns 딕셔너리
                       {"product_name": "실제컬럼명", ...}
        exclude_large: 제외할 대분류 값 목록
        exclude_tag:   제외할 카테고리태그 값 목록

    Returns:
        DataFrame with columns:
            product_name, large_category, medium_category,
            category_tag, brand_name
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"입력 파일 없음: {input_path}")

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
        logger.info("XLSX 로드: %s (%d행)", path.name, len(df))
    elif suffix == ".csv":
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
        logger.info("CSV 로드: %s (%d행)", path.name, len(df))
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {suffix}. CSV 또는 XLSX만 허용.")

    # 컬럼 매핑: 내부 표준명 → 실제 파일 컬럼명
    reverse_map = {v: k for k, v in col_map.items()}
    missing = [v for v in col_map.values() if v not in df.columns]
    if missing:
        raise ValueError(f"파일에 없는 컬럼: {missing}\n실제 컬럼: {list(df.columns)}")

    df = df.rename(columns=reverse_map)

    # 표준 컬럼만 선택
    keep = ["product_name", "large_category", "medium_category", "category_tag", "brand_name"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # 결측치 제거
    before = len(df)
    df = df.dropna(subset=["product_name", "large_category", "medium_category"])
    df["brand_name"]   = df["brand_name"].fillna("unknown")
    df["category_tag"] = df["category_tag"].fillna("UNKNOWN")
    logger.info("결측치 제거: %d행 → %d행", before, len(df))

    # 제외 카테고리 필터
    if exclude_large:
        df = df[~df["large_category"].isin(exclude_large)]
    if exclude_tag:
        df = df[~df["category_tag"].isin(exclude_tag)]

    df = df.reset_index(drop=True)
    logger.info(
        "최종 학습 데이터: %d행, 대분류 %d종, 중분류 %d종, 태그 %d종",
        len(df),
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
    CV 전략에 따라 (train_idx, val_idx) 쌍 목록을 반환한다.

    Args:
        df:         load_data() 반환 DataFrame
        n_splits:   Fold 수
        group_col:  그룹 기준 컬럼 (GroupKFold 계열에서 사용)
        target_col: 층화 기준 컬럼 (Stratified 계열에서 사용)
        strategy:   CV 전략 이름
        seed:       재현성 시드 (KFold / Stratified 계열에서 사용)

    Returns:
        List of (train_indices, val_indices)

    전략별 특성:
        GroupKFold           - 브랜드 완전 분리, 클래스 비율 보장 없음
        StratifiedGroupKFold - 브랜드 분리 + 클래스 비율 유지 (권장)
        StratifiedKFold      - 클래스 비율 유지, 브랜드 누수 가능
        KFold                - 완전 랜덤, 베이스라인 비교용
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
            len(set(groups[va])),
            len(set(y[va])),
        )

    _log_fold_summary(strategy, folds, groups, y)
    return folds


def _make_splitter(strategy: CVStrategy, n_splits: int, seed: int):
    """전략 이름으로 splitter 객체를 생성한다."""
    match strategy:
        case "GroupKFold":
            # GroupKFold는 shuffle 없음 — 브랜드 수 기준으로 결정론적 분할
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


def _make_split_kwargs(
    strategy: CVStrategy,
    df: pd.DataFrame,
    groups,
    y,
) -> dict:
    """
    전략별로 splitter.split()에 넘길 kwargs를 구성한다.

    sklearn splitter마다 split() 시그니처가 달라서 분기 처리.
        GroupKFold:           split(X, groups=groups)
        StratifiedGroupKFold: split(X, y, groups=groups)
        StratifiedKFold:      split(X, y)
        KFold:                split(X)
    """
    X = df  # X는 인덱스 용도로만 사용 (실제 피처 불필요)
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


def _log_fold_summary(
    strategy: CVStrategy,
    folds: list[Tuple[list[int], list[int]]],
    groups,
    y,
) -> None:
    """Fold 구성 요약을 INFO로 출력한다 (전략별 경고 포함)."""
    import numpy as np

    val_sizes    = [len(va) for _, va in folds]
    val_brands   = [len(set(groups[va])) for _, va in folds]
    val_classes  = [len(set(y[va])) for _, va in folds]
    total_classes = len(set(y))

    logger.info(
        "[%s] %d-Fold 구성 완료\n"
        "  val 크기:     %s (mean=%.0f)\n"
        "  val 브랜드 수: %s\n"
        "  val 클래스 수: %s / 전체 %d",
        strategy, len(folds),
        val_sizes, np.mean(val_sizes),
        val_brands,
        val_classes, total_classes,
    )

    # 전략별 주의사항 경고
    if strategy == "KFold":
        logger.warning(
            "[KFold] 브랜드 기준 분리 없음 — 동일 브랜드가 train/val에 동시 등장 가능. "
            "성능이 실제보다 낙관적으로 측정될 수 있음. 베이스라인 비교용으로만 사용 권장."
        )
    if strategy == "StratifiedKFold":
        logger.warning(
            "[StratifiedKFold] 클래스 비율은 유지되지만 브랜드 누수 발생 가능. "
            "데이터가 적어 StratifiedGroupKFold가 실패하는 경우 fallback으로 사용."
        )
    if strategy == "GroupKFold":
        missing_cls = [
            f"Fold {i+1}: {total_classes - c}개 누락"
            for i, (_, va) in enumerate(folds)
            if (c := len(set(y[va]))) < total_classes
        ]
        if missing_cls:
            logger.warning(
                "[GroupKFold] 일부 Fold의 val에서 클래스 누락 — %s. "
                "StratifiedGroupKFold로 전환을 권장.",
                ", ".join(missing_cls),
            )


# ──────────────────────────────────────────────────────────────────
# 데이터 요약
# ──────────────────────────────────────────────────────────────────

def data_summary(df: pd.DataFrame) -> dict:
    """학습 데이터 통계 요약 딕셔너리 반환 (HTML 리포트용)."""
    return {
        "n_samples": len(df),
        "n_large":   df["large_category"].nunique(),
        "n_medium":  df["medium_category"].nunique(),
        "n_tag":     df["category_tag"].nunique(),
        "n_brands":  df["brand_name"].nunique(),
        "large_dist": df["large_category"].value_counts().to_dict(),
        "tag_dist":   df["category_tag"].value_counts().to_dict(),
    }