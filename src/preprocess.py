"""
공용 전처리 모듈.

파이프라인:
    1. REFProductNameParser로 refined_text 생성 (기존 parser 재사용)
    2. Okt 명사 추출 + 불용어 제거 → nouns_text
    3. 술 카테고리 특수처리: 브랜드명 복구
    4. 타깃 컬럼 LabelEncoder 인코딩 (대/중분류, 카테고리태그)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# KoNLPy 선택적 임포트 (Java 미설치 환경 대응)
try:
    from konlpy.tag import Okt
    _OKT_AVAILABLE = True
except Exception:
    _OKT_AVAILABLE = False
    logger.warning("KoNLPy Okt 초기화 실패. nouns_text는 refined_text와 동일하게 설정됩니다.")

# product_name_parser는 프로젝트 루트에 위치
try:
    from product_name_parser import REFProductNameParser, create_brand_matcher
    _PARSER_AVAILABLE = True
except ImportError:
    _PARSER_AVAILABLE = False
    logger.warning("product_name_parser 임포트 실패. 원본 product_name을 그대로 사용합니다.")

ALCOHOL_LARGE = "술"


class REFPreprocessor:
    """
    RE:FRIDGE 제품명 전처리기.

    fit_transform()으로 DataFrame을 변환하여:
        - refined_text : 파서 정제 제품명
        - nouns_text   : Okt 명사 추출 + 불용어 제거 (공백 구분)
        - label_large  : 대분류 인코딩 (int)
        - label_medium : 중분류 인코딩 (int)
        - label_tag    : 카테고리태그 인코딩 (int)
    컬럼을 추가한다.

    label_encoders 프로퍼티로 LabelEncoder 3개 접근 가능.
    """

    def __init__(
        self,
        brand_dict_path: str = "product_data_collection/not_grocery_and_brand_list/grocery_brand_name.json",
        stopwords: list[str] | None = None,
        alcohol_brand_preserve: bool = True,
        use_parser: bool = True,
        morpheme_analyzer: str = "Okt",
    ) -> None:
        self._stopwords: set[str] = set(stopwords or [])
        self._alcohol_brand_preserve = alcohol_brand_preserve
        self._use_parser = use_parser and _PARSER_AVAILABLE

        # LabelEncoder (fit 전까지 None)
        self._le_large:  LabelEncoder | None = None
        self._le_medium: LabelEncoder | None = None
        self._le_tag:    LabelEncoder | None = None

        # 파서 초기화
        self._parser: Optional[REFProductNameParser] = None
        if self._use_parser:
            self._parser = self._init_parser(brand_dict_path)

        # 형태소 분석기
        self._okt = None
        if _OKT_AVAILABLE and morpheme_analyzer == "Okt":
            self._okt = Okt()
            logger.info("Okt 형태소 분석기 초기화 완료")

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame에 refined_text, nouns_text, label_* 컬럼을 추가하여 반환.

        Args:
            df: load_data() 반환 DataFrame
                (product_name, large_category, medium_category,
                 category_tag, brand_name 컬럼 필요)

        Returns:
            원본 + 신규 컬럼 DataFrame (inplace 변경 없음)
        """
        result = df.copy()

        logger.info("제품명 정제 시작 (%d개)", len(result))
        result["refined_text"] = result.apply(self._refine_row, axis=1)

        logger.info("명사 추출 시작")
        result["nouns_text"] = result["refined_text"].apply(self._extract_nouns)

        logger.info("레이블 인코딩")
        self._le_large  = LabelEncoder().fit(result["large_category"])
        self._le_medium = LabelEncoder().fit(result["medium_category"])
        self._le_tag    = LabelEncoder().fit(result["category_tag"])

        result["label_large"]  = self._le_large.transform(result["large_category"])
        result["label_medium"] = self._le_medium.transform(result["medium_category"])
        result["label_tag"]    = self._le_tag.transform(result["category_tag"])

        logger.info(
            "전처리 완료 — refined_text 평균 길이: %.1f자, nouns_text 평균 길이: %.1f자",
            result["refined_text"].str.len().mean(),
            result["nouns_text"].str.len().mean(),
        )
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """fit 완료 후 새 데이터에 적용 (인코더 재학습 없음)."""
        if self._le_large is None:
            raise RuntimeError("fit_transform()을 먼저 호출하세요.")
        result = df.copy()
        result["refined_text"] = result.apply(self._refine_row, axis=1)
        result["nouns_text"]   = result["refined_text"].apply(self._extract_nouns)
        result["label_large"]  = self._le_large.transform(result["large_category"])
        result["label_medium"] = self._le_medium.transform(result["medium_category"])
        result["label_tag"]    = self._le_tag.transform(result["category_tag"])
        return result

    @property
    def label_encoders(self) -> dict[str, LabelEncoder]:
        """{"large": le, "medium": le, "tag": le} 형태로 반환."""
        if self._le_large is None:
            raise RuntimeError("fit_transform()을 먼저 호출하세요.")
        return {
            "large":  self._le_large,
            "medium": self._le_medium,
            "tag":    self._le_tag,
        }

    @property
    def n_classes(self) -> dict[str, int]:
        """각 타깃의 클래스 수."""
        les = self.label_encoders
        return {k: len(v.classes_) for k, v in les.items()}

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    def _refine_row(self, row: pd.Series) -> str:
        """행 단위 제품명 정제."""
        name = str(row["product_name"]) if pd.notna(row["product_name"]) else ""
        large = str(row.get("large_category", "")) if pd.notna(row.get("large_category", "")) else ""

        if not self._parser:
            return name

        parsed = self._parser.parse(name)
        refined = parsed.refined_text or name

        # 술 카테고리: 브랜드명이 사실상 제품명인 경우 복구
        if self._alcohol_brand_preserve and large == ALCOHOL_LARGE:
            if (not refined or len(refined.strip()) < 2) and parsed.brand_name:
                refined = parsed.brand_name
                logger.debug("술 카테고리 브랜드 복구: '%s' → '%s'", name, refined)

        return refined.strip() if refined else name

    def _extract_nouns(self, text: str) -> str:
        """Okt 명사 추출 후 불용어 제거, 공백으로 연결."""
        if not text:
            return ""
        if self._okt is None:
            return text
        try:
            nouns = self._okt.nouns(text)
            filtered = [n for n in nouns if n not in self._stopwords and len(n) > 1]
            return " ".join(filtered) if filtered else text
        except Exception as e:
            logger.debug("Okt 명사 추출 실패 '%s': %s", text, e)
            return text

    @staticmethod
    def _init_parser(brand_dict_path: str) -> Optional["REFProductNameParser"]:
        """브랜드 사전을 로드하여 파서를 초기화한다."""
        path = Path(brand_dict_path)
        brand_names: list[str] = []
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    brand_names = [str(b) for b in data if b]
                elif isinstance(data, dict):
                    brand_names = list(data.keys())
                logger.info("브랜드 사전 로드: %d개", len(brand_names))
            except Exception as e:
                logger.warning("브랜드 사전 로드 실패: %s", e)
        else:
            logger.warning("브랜드 사전 파일 없음: %s. 파서는 브랜드 없이 실행됩니다.", brand_dict_path)

        matcher = create_brand_matcher(brand_names)
        return REFProductNameParser(brand_matcher=matcher)
