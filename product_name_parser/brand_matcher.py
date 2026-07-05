"""
브랜드명 매칭 포트 및 구현체.
Java의 REFBrandMatcher 인터페이스 + Aho-Corasick 기반 구현체에 대응.

의존성:
    pip install pyahocorasick
"""

from abc import ABC, abstractmethod
from typing import Optional, Iterable

try:
    import ahocorasick
    _AHO_AVAILABLE = True
except ImportError:
    _AHO_AVAILABLE = False


class REFBrandMatcher(ABC):
    """브랜드 사전 매칭 포트 인터페이스."""

    @abstractmethod
    def find_match(self, text: str) -> Optional[str]:
        """
        텍스트에서 첫 번째 브랜드명을 찾아 반환한다.

        Args:
            text: 검색 대상 원본 텍스트

        Returns:
            매칭된 브랜드명 문자열, 없으면 None
        """
        ...


class AhoCorasickBrandMatcher(REFBrandMatcher):
    """
    Aho-Corasick 알고리즘을 사용한 브랜드명 매칭 구현체.
    Java의 Aho-Corasick 기반 REFBrandMatcher 구현에 대응.

    빈 브랜드 사전으로 초기화하면 find_match()가 항상 None을 반환한다.
    (술 카테고리용 parser_no_brand 생성 시 사용)

    Usage:
        brands = ["CJ", "풀무원", "오뚜기", "농심", "삼양"]
        matcher = AhoCorasickBrandMatcher(brands)
        brand = matcher.find_match("오뚜기 진라면 순한맛 120g 5개입")
        # → "오뚜기"
    """

    def __init__(self, brand_names: Iterable[str]) -> None:
        if not _AHO_AVAILABLE:
            raise ImportError(
                "pyahocorasick 라이브러리가 필요합니다.\n"
                "설치: pip install pyahocorasick"
            )
        self._automaton, self._is_empty = self._build_automaton(brand_names)

    @staticmethod
    def _build_automaton(brand_names: Iterable[str]):
        """
        Automaton을 빌드한다.

        브랜드 사전이 비어 있으면 make_automaton()을 호출하지 않고
        is_empty=True 를 반환한다. (빈 Automaton에 make_automaton()을
        호출하면 iter() 시 AttributeError 발생)
        """
        A = ahocorasick.Automaton()
        added = 0
        for idx, brand in enumerate(brand_names):
            if brand:
                A.add_word(brand, (idx, brand))
                added += 1

        if added == 0:
            # 빈 사전: Automaton 미완성 상태로 두고 is_empty 플래그로 관리
            return A, True

        A.make_automaton()
        return A, False

    def find_match(self, text: str) -> Optional[str]:
        """텍스트에서 첫 번째로 등장하는 브랜드명을 반환한다."""
        # 빈 사전이면 항상 None (술 카테고리용 parser_no_brand 동작)
        if self._is_empty or not text:
            return None

        best: Optional[str] = None
        best_pos = len(text)

        for end_idx, (_, brand) in self._automaton.iter(text):
            start_idx = end_idx - len(brand) + 1
            if start_idx < best_pos:
                best_pos = start_idx
                best = brand

        return best

    def find_all_matches(self, text: str) -> list[str]:
        """텍스트에서 매칭된 모든 브랜드명을 등장 순서대로 반환한다."""
        if self._is_empty or not text:
            return []
        matches = []
        for end_idx, (_, brand) in self._automaton.iter(text):
            matches.append((end_idx - len(brand) + 1, brand))
        matches.sort(key=lambda x: x[0])
        return [b for _, b in matches]


class SimpleBrandMatcher(REFBrandMatcher):
    """
    pyahocorasick 없이 동작하는 간단한 선형 탐색 브랜드 매칭 구현체.
    소규모 브랜드 사전이나 테스트 환경에서 사용.
    """

    def __init__(self, brand_names: Iterable[str]) -> None:
        self._brands: list[str] = [b for b in brand_names if b]

    def find_match(self, text: str) -> Optional[str]:
        if not text:
            return None

        best: Optional[str] = None
        best_pos = len(text)

        for brand in self._brands:
            pos = text.find(brand)
            if pos != -1 and pos < best_pos:
                best_pos = pos
                best = brand

        return best


def create_brand_matcher(brand_names: Iterable[str]) -> REFBrandMatcher:
    """
    환경에 따라 적합한 브랜드 매처를 자동 생성한다.
    pyahocorasick 설치 여부에 따라 구현체를 선택한다.

    빈 brand_names를 넘기면 find_match()가 항상 None을 반환하는
    매처가 생성된다. (술 카테고리용 parser_no_brand 생성 시 활용)
    """
    if _AHO_AVAILABLE:
        return AhoCorasickBrandMatcher(brand_names)
    return SimpleBrandMatcher(brand_names)