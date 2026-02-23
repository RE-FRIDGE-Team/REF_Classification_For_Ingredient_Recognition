"""
REF 제품명 파서 패키지.

Quick start:
    from product_name_parser import REFProductNameParser, create_brand_matcher

    brands = ["오뚜기", "농심", "CJ", "풀무원", "동원"]
    matcher = create_brand_matcher(brands)
    parser  = REFProductNameParser(brand_matcher=matcher)

    result = parser.parse("오뚜기 진라면 순한맛 120g 5개입 [코스트코]")
    print(result)
"""

from .brand_matcher import (
    AhoCorasickBrandMatcher,
    REFBrandMatcher,
    SimpleBrandMatcher,
    create_brand_matcher,
)
from .models import REFParsedProductName
from .parser import REFProductNameParser

__all__ = [
    "REFProductNameParser",
    "REFParsedProductName",
    "REFBrandMatcher",
    "AhoCorasickBrandMatcher",
    "SimpleBrandMatcher",
    "create_brand_matcher",
]
