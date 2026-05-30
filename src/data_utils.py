"""
데이터 로드 및 GroupKFold 분할 유틸리티.

CSV / XLSX 파일을 읽어 표준 스키마로 정규화하고,
brand 컬럼 기준 5-Fold GroupKFold 인덱스를 반환한다.
"""

import logging
from pathlib import Path
from typing import Iterator, Tuple

import pandas as pd
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)


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
        df = pd.read_csv(path, dtype=str, encoding="cp949")
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
    df["brand_name"] = df["brand_name"].fillna("unknown")
    df["category_tag"] = df["category_tag"].fillna("UNKNOWN")
    logger.info("결측치 제거: %d행 → %d행", before, len(df))

    # 제외 카테고리 필터
    if exclude_large:
        df = df[~df["large_category"].isin(exclude_large)]
    if exclude_tag:
        df = df[~df["category_tag"].isin(exclude_tag)]

    df = df.reset_index(drop=True)
    logger.info("최종 학습 데이터: %d행, 대분류 %d종, 중분류 %d종, 태그 %d종",
                len(df),
                df["large_category"].nunique(),
                df["medium_category"].nunique(),
                df["category_tag"].nunique())
    return df


def make_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "brand_name",
    seed: int = 42,
) -> list[Tuple[list[int], list[int]]]:
    """
    GroupKFold로 (train_idx, val_idx) 쌍 목록을 반환한다.

    Args:
        df:        load_data() 반환 DataFrame
        n_splits:  Fold 수
        group_col: 그룹 기준 컬럼 (브랜드 기준 데이터 누수 방지)
        seed:      재현성 시드 (GroupKFold는 내부적으로 미사용; 문서 목적)

    Returns:
        List of (train_indices, val_indices)
    """
    groups = df[group_col].values
    gkf = GroupKFold(n_splits=n_splits)
    folds: list[Tuple[list[int], list[int]]] = []
    for fold_idx, (tr, va) in enumerate(gkf.split(df, groups=groups)):
        folds.append((tr.tolist(), va.tolist()))
        logger.debug(
            "Fold %d — train: %d, val: %d, val 그룹 수: %d",
            fold_idx + 1, len(tr), len(va),
            len(set(groups[va]))
        )
    return folds


def data_summary(df: pd.DataFrame) -> dict:
    """학습 데이터 통계 요약 딕셔너리 반환 (HTML 리포트용)."""
    return {
        "n_samples": len(df),
        "n_large": df["large_category"].nunique(),
        "n_medium": df["medium_category"].nunique(),
        "n_tag": df["category_tag"].nunique(),
        "n_brands": df["brand_name"].nunique(),
        "large_dist": df["large_category"].value_counts().to_dict(),
        "tag_dist": df["category_tag"].value_counts().to_dict(),
    }
