"""
제품명 파싱 메인 구현체.
Java의 REFProductNameParserAdapter를 Python으로 1:1 포팅.

파싱 순서:
    1. 브랜드명 추출 (Aho-Corasick 사전 매칭)
    2. 용량 추출 (정규식)
    3. 수량 추출 (정규식)
    4. 불필요 요소 제거 (가격, 할인, 배송, 평점 등)
    5. 용량/수량 텍스트 제거
    6. 최종 정제
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from .brand_matcher import REFBrandMatcher
from .models import REFParsedProductName
from .patterns import (
    NOISE_PATTERNS,
    QUANTITY_PATTERN,
    VOLUME_PATTERN,
    _IDX_BRACKET,
    _IDX_PAREN_END,
    _IDX_PAREN_START,
    _IDX_PRICE_END,
)

logger = logging.getLogger(__name__)


@dataclass
class _VolumeInfo:
    """용량 추출 내부 결과 (Java의 private record VolumeInfo에 대응)."""
    volume_text: Optional[str]
    unit: Optional[str]


class REFProductNameParser:
    """
    제품명 파싱 구현체.

    Java의 REFProductNameParserAdapter 를 Python으로 1:1 포팅.
    Spring @Component / @RequiredArgsConstructor 대응으로
    생성자에서 REFBrandMatcher를 주입받는다.

    Usage:
        from product_name_parser import REFProductNameParser, create_brand_matcher

        brands = ["오뚜기", "농심", "CJ", "풀무원"]
        matcher = create_brand_matcher(brands)
        parser = REFProductNameParser(brand_matcher=matcher)

        result = parser.parse("오뚜기 진라면 순한맛 120g 5개입 [코스트코]")
        print(result.refined_text)   # → "진라면 순한맛"
        print(result.brand_name)     # → "오뚜기"
        print(result.volume)         # → "120g"
        print(result.quantity)       # → 5
    """

    def __init__(self, brand_matcher: REFBrandMatcher) -> None:
        self._brand_matcher = brand_matcher

    # ──────────────────────────────────────────────────────────────
    # Public API (Java의 @Override parse())
    # ──────────────────────────────────────────────────────────────

    def parse(self, raw_product_name: Optional[str]) -> REFParsedProductName:
        """
        원본 제품명을 파싱하여 구조화된 결과를 반환한다.

        Args:
            raw_product_name: OCR 또는 크롤링으로 수집된 원본 제품명

        Returns:
            REFParsedProductName: 브랜드, 용량, 수량, 정제명 포함
        """
        if not raw_product_name or not raw_product_name.strip():
            return REFParsedProductName(original_text=raw_product_name)

        working = raw_product_name.strip()

        # 1. 브랜드명 추출
        brand_name = self._extract_brand(working)
        if brand_name:
            working = working.replace(brand_name, "", 1).strip()

        # 2. 용량 추출
        volume_info = self._extract_volume(working)

        # 3. 수량 추출
        quantity = self._extract_quantity(working)

        # 4. 불필요 요소 제거
        refined = self._remove_noise_patterns(working)

        # 5. 용량/수량 텍스트 제거
        refined = self._remove_volume_and_quantity(refined)

        # 6. 최종 정제
        refined = self._final_cleanup(refined)

        logger.debug(
            "파싱 완료 - 원본: '%s' → 정제: '%s', 브랜드: '%s', 용량: '%s', 수량: %s",
            raw_product_name, refined, brand_name, volume_info.volume_text, quantity,
        )

        return REFParsedProductName(
            original_text=raw_product_name,
            refined_text=refined,
            brand_name=brand_name,
            quantity=quantity,
            volume=volume_info.volume_text,
            volume_unit=volume_info.unit,
        )

    # ──────────────────────────────────────────────────────────────
    # Private helpers (Java private 메서드에 1:1 대응)
    # ──────────────────────────────────────────────────────────────

    def _extract_brand(self, text: str) -> Optional[str]:
        """Java: private String extractBrand(String text)"""
        return self._brand_matcher.find_match(text)

    @staticmethod
    def _extract_volume(text: str) -> _VolumeInfo:
        """
        Java: private VolumeInfo extractVolume(String text)

        용량값과 단위를 추출한다. '리터' → 'l' 정규화 포함.
        """
        m = VOLUME_PATTERN.search(text)
        if m:
            value = m.group(1).replace(",", ".")
            unit = m.group(2).lower()
            if unit == "리터":
                unit = "l"
            return _VolumeInfo(volume_text=value + unit, unit=unit)
        return _VolumeInfo(volume_text=None, unit=None)

    @staticmethod
    def _extract_quantity(text: str) -> Optional[int]:
        """
        Java: private Integer extractQuantity(String text)

        수량 단위(개입, 팩, 봉 …) 앞 숫자를 추출한다.
        """
        m = QUANTITY_PATTERN.search(text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                logger.warning("수량 파싱 실패: %s", m.group(1))
        return None

    @staticmethod
    def _remove_noise_patterns(text: str) -> str:
        """
        Java: private String removeNoisePatterns(String text)

        NOISE_PATTERNS를 순서대로 적용하되 조기 스킵 최적화를 동일하게 적용.
        - 인덱스 0-3  : '원' 없으면 스킵 (가격 패턴)
        - 인덱스 4    : '[' 없으면 스킵 (대괄호 패턴)
        - 인덱스 5-35 : '(' 없으면 스킵 (괄호 패턴)
        """
        if not text:
            return text

        result = text

        has_bracket = "[" in result
        has_paren   = "(" in result
        has_price   = "원" in result

        for i, pattern in enumerate(NOISE_PATTERNS):
            result = pattern.sub("", result)


        return result

    @staticmethod
    def _remove_volume_and_quantity(text: str) -> str:
        """Java: private String removeVolumeAndQuantity(String text)"""
        result = VOLUME_PATTERN.sub("", text)
        result = QUANTITY_PATTERN.sub("", result)
        return result

    @staticmethod
    def _final_cleanup(text: str) -> str:
        """
        Java: private String finalCleanup(String text)

        연속 쉼표·공백 정규화, 빈 괄호 제거, 앞뒤 쉼표/공백 제거.
        """
        if not text:
            return text

        result = text
        result = re.sub(r",+", ",", result)
        result = re.sub(r"\s*,\s*", " ", result)
        result = re.sub(r"\s+", " ", result)
        result = re.sub(r"\(\s*\)", "", result)
        result = re.sub(r"\[\s*\]", "", result)
        result = result.strip()
        result = re.sub(r"^[,\s]+|[,\s]+$", "", result)
        result = result.strip()
        return result