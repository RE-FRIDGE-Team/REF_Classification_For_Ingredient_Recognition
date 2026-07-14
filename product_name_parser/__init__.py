"""
REF 제품명 파서 패키지 (2026-07 개편판).

Quick start:
    from product_name_parser import (
        REFProductNameParser, ParserOptions, create_brand_matcher,
    )

    brands  = ["오뚜기", "농심", "CJ", "풀무원", "동원"]
    matcher = create_brand_matcher(brands)

    # 기본(브랜드/용량/수량 제거) 파서
    parser = REFProductNameParser(brand_matcher=matcher)

    # 플레이스홀더 파서 — 제거 대신 BRANDTOK/VOLTOK/QTYTOK 치환
    parser_ph = REFProductNameParser(
        brand_matcher=matcher,
        options=ParserOptions(placeholder=True),
    )
"""

from .brand_matcher import (
    AhoCorasickBrandMatcher,
    REFBrandMatcher,
    SimpleBrandMatcher,
    create_brand_matcher,
)
from .models import REFParsedProductName
from .parser import (
    BRAND_TOKEN,
    QTY_TOKEN,
    VOL_TOKEN,
    ParserOptions,
    REFProductNameParser,
)

__all__ = [
    "REFProductNameParser",
    "REFParsedProductName",
    "ParserOptions",
    "REFBrandMatcher",
    "AhoCorasickBrandMatcher",
    "SimpleBrandMatcher",
    "create_brand_matcher",
    "BRAND_TOKEN",
    "VOL_TOKEN",
    "QTY_TOKEN",
]
