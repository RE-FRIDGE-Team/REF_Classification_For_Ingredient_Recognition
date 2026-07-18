"""
제품명 파싱 메인 구현체.
Java의 REFProductNameParserAdapter를 Python으로 포팅한 버전에
2026-07 개편(단어 파먹기 버그 수정 + 토글/플레이스홀더 옵션)을 반영.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[개편 요약]

1. 브랜드 제거 버그 수정 (★최우선)
   기존: working.replace(brand_name, "", 1)
         → 브랜드 사전의 '카스텔'이 '카스텔라' 안쪽을 substring 치환
           ("생크림 카스텔라" → "생크림 라")
   수정: _remove_brand_with_boundary()
         - 매칭 위치의 양옆이 '단어 문자(한글/영문/숫자)'가 아닐 때만 제거
         - PROTECTED_NOUNS(탕수육·카스텔라·건면 등)를 포함하는 위치는 제거 금지
         - 경계를 만족하는 위치가 없으면 브랜드를 제거하지 않음(안전 우선)

2. 파싱 정책 토글화 (ParserOptions)
   - remove_brand   : 브랜드 제거 ON/OFF (OFF면 브랜드 원문 유지)
   - remove_volume  : 용량 제거 ON/OFF
   - remove_quantity: 수량 제거 ON/OFF
   - placeholder    : 제거 대신 <BRAND>/<VOL>/<QTY> 토큰으로 치환
                      ("사이즈 있는 식품"이라는 신호 자체를 피처로 남기기 위함)
   ※ 가격/배송/적립/리뷰수 등 '순수 노이즈'는 토글 없이 무조건 제거한다.

3. 보호 명사 리스트 (PROTECTED_NOUNS)
   정상 제품명 토큰이 브랜드/단위 치환에 의해 깨지는 것을 막는 마지막 방어선.
   코드에 시드를 내장하고, resources/protected_nouns.txt(한 줄 한 단어)가
   존재하면 자동으로 병합 로드한다 — 사람 개입이 필요할 때 파일만 추가하면 됨.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .brand_matcher import REFBrandMatcher
from .models import REFParsedProductName
from .patterns import NOISE_PATTERNS, QUANTITY_PATTERN, VOLUME_PATTERN

logger = logging.getLogger(__name__)

# 플레이스홀더 토큰 상수 — TF-IDF word 토큰으로 그대로 살아남도록 영문으로 구성
BRAND_TOKEN = "BRANDTOK"   # <BRAND> 는 char n-gram 에서 <, > 가 노이즈가 되므로 영숫자 토큰 사용
VOL_TOKEN   = "VOLTOK"
QTY_TOKEN   = "QTYTOK"

# ── 보호 명사 시드 ──────────────────────────────────────────────
# 브랜드/단위 치환이 절대 건드리면 안 되는 정상 제품 토큰.
# (버그 리포트에서 실제로 깨졌거나 깨질 위험이 확인된 단어들)
_PROTECTED_NOUNS_SEED: frozenset[str] = frozenset({
    "카스텔라", "탕수육", "건면", "생면", "봉지라면", "쌀봉지",
    "구미", "봉골레", "미역", "포기김치", "정과", "본죽",
    "마리네이드", "세트지", "입욕", "매생이", "미숫가루",
})


def _load_protected_nouns(extra_path: str | Path | None = None) -> frozenset[str]:
    """
    보호 명사 집합을 로드한다.

    내장 시드(_PROTECTED_NOUNS_SEED)에 더해, extra_path 파일이 존재하면
    한 줄에 한 단어 형식으로 읽어 병합한다. 파일이 없어도 정상 동작한다.
    (5순위 원칙: 사람 개입 없이도 동작, 개입이 필요하면 파일 한 개만 만들면 됨)
    """
    words = set(_PROTECTED_NOUNS_SEED)
    if extra_path:
        p = Path(extra_path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                w = line.strip()
                if w and not w.startswith("#"):   # 주석 라인 허용
                    words.add(w)
            logger.info("보호 명사 확장 파일 로드: %s (%d개)", p, len(words))
    return frozenset(words)


@dataclass
class ParserOptions:
    """
    파싱 정책 토글 컨테이너 (2026-07 2차 개편 — keep 의미 재정의).

    [keep_* 의 정확한 의미]
      쪼개기(브랜드/용량/수량 메타데이터 추출)는 항상 수행.
      다음 프로세스로 보내는 refined_text 를 어떻게 조립할지만 결정.

        keep_brand_in_text=True, keep_volume_in_text=True 인 경우:
          "농심 양파링 오리지널, 80g, 3개 3,960원 …"
            → brand=농심 / volume=80g / quantity=3 으로 쪼갠 뒤,
              refined_text = "농심 양파링 오리지널 80g"  (수량·노이즈만 탈락)

      기존 remove_* 방식(브랜드를 '원문 위치에 그대로 방치')과 달리,
      코어를 완전히 정제한 뒤 [브랜드] + 코어 + [용량] 순으로 '재조립'하므로
      노이즈 패턴이 용량을 먼저 삼키거나 표기 순서가 뒤죽박죽인 원문에서도
      항상 일관된 형태의 출력이 보장되는 구조.

    Attributes:
        remove_brand:    브랜드명 제거 여부 (keep_*_in_text 미사용 시의 구버전 동작)
        remove_volume:   용량 제거 여부
        remove_quantity: 수량 제거 여부
        keep_brand_in_text:  코어 정제 후 브랜드를 맨 앞에 재부착
        keep_volume_in_text: 코어 정제 후 정규화된 용량(80g)을 맨 뒤에 재부착
        placeholder:     True면 '제거' 대신 BRANDTOK/VOLTOK/QTYTOK 로 '치환'
        protected_nouns_path: 보호 명사 확장 파일 경로 (없어도 됨)
    """
    remove_brand:    bool = True
    remove_volume:   bool = True
    remove_quantity: bool = True
    keep_brand_in_text:  bool = False
    keep_volume_in_text: bool = False
    placeholder:     bool = False
    protected_nouns_path: str | None = "resources/protected_nouns.txt"

    # 로드된 보호 명사 집합 (자동 초기화)
    protected_nouns: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # 인스턴스 생성 시점에 보호 명사 자동 로드 (사람 개입 불필요)
        self.protected_nouns = _load_protected_nouns(self.protected_nouns_path)


@dataclass
class _VolumeInfo:
    """용량 추출 내부 결과 (Java의 private record VolumeInfo 대응)."""
    volume_text: Optional[str]
    unit: Optional[str]


def _is_word_char(ch: str) -> bool:
    """한글/영문/숫자면 True — 단어 경계 판정에 사용."""
    return bool(re.match(r"[0-9a-zA-Z가-힣]", ch))


class REFProductNameParser:
    """
    제품명 파싱 구현체 (버그 수정 + 토글 버전).

    Usage:
        matcher = create_brand_matcher(["오뚜기", "농심"])
        parser  = REFProductNameParser(matcher, options=ParserOptions(placeholder=True))
        result  = parser.parse("오뚜기 진라면 순한맛 120g 5개입")
        result.refined_text  # → "BRANDTOK 진라면 순한맛 VOLTOK QTYTOK"
    """

    def __init__(
        self,
        brand_matcher: REFBrandMatcher,
        options: ParserOptions | None = None,
    ) -> None:
        self._brand_matcher = brand_matcher
        self._opt = options or ParserOptions()

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def parse(self, raw_product_name: Optional[str]) -> REFParsedProductName:
        """원본 제품명을 파싱하여 구조화된 결과를 반환한다."""
        if not raw_product_name or not raw_product_name.strip():
            return REFParsedProductName(original_text=raw_product_name)

        working = raw_product_name.strip()

        # 1. 메타데이터 추출 — keep/remove 여부와 무관하게 항상 쪼개기 수행
        brand_name  = self._extract_brand(working)
        volume_info = self._extract_volume(working)
        quantity    = self._extract_quantity(working)

        keep_mode = self._opt.keep_brand_in_text or self._opt.keep_volume_in_text

        # 2. 브랜드 제거/치환 — ★토큰 경계 + 보호명사 검사 적용
        #    keep 모드에서는 일단 코어에서 브랜드를 떼어낸 뒤 마지막에 재부착
        if brand_name and (self._opt.remove_brand or keep_mode):
            replacement = BRAND_TOKEN if (self._opt.placeholder and not keep_mode) else ""
            working = self._remove_brand_with_boundary(working, brand_name, replacement)

        # 3. 순수 노이즈 제거 (가격·배송·적립·리뷰 등 — 토글 대상 아님, 무조건 제거)
        refined = self._remove_noise_patterns(working)

        # 4. 용량/수량 텍스트 처리 — keep 모드에서는 코어에서 전부 제거 후 재부착
        refined = self._handle_volume_and_quantity(refined, force_remove=keep_mode)

        # 5. 코어 확정 (공백/쉼표/빈 괄호 정리)
        core = self._final_cleanup(refined)

        # 6. 재조립 — [브랜드] + 코어 + [정규화 용량] 순의 일관된 출력 조립
        refined_out = core
        if keep_mode:
            parts: list[str] = []
            if self._opt.keep_brand_in_text and brand_name and brand_name not in core:
                parts.append(brand_name)          # 코어에 이미 남아있으면 중복 부착 방지
            parts.append(core)
            if self._opt.keep_volume_in_text and volume_info.volume_text:
                parts.append(volume_info.volume_text)
            refined_out = " ".join(p for p in parts if p).strip()

        return REFParsedProductName(
            original_text=raw_product_name,
            refined_text=refined_out,
            core_text=core,
            brand_name=brand_name,
            quantity=quantity,
            volume=volume_info.volume_text,
            volume_unit=volume_info.unit,
        )

    # ──────────────────────────────────────────────────────────────
    # 브랜드 제거 (★버그 수정 핵심)
    # ──────────────────────────────────────────────────────────────

    def _remove_brand_with_boundary(
        self, text: str, brand: str, replacement: str
    ) -> str:
        """
        브랜드 문자열을 '토큰 경계'에서만 제거/치환한다.

        검사 순서 (모두 통과해야 제거):
          (a) 매칭 시작 직전 문자가 단어 문자가 아니어야 함  (앞 경계)
          (b) 매칭 끝 직후  문자가 단어 문자가 아니어야 함  (뒤 경계)
              → '카스텔' in '카스텔라' 는 (b)에서 탈락 ("라"가 한글)
          (c) 매칭 구간을 포함하는 보호 명사가 그 위치에 존재하지 않아야 함
              → 사전 오염(브랜드 사전에 일반명사 유입)에 대한 이중 방어

        경계를 만족하는 등장 위치가 하나도 없으면 원문을 그대로 반환한다.
        (제품명을 깨뜨리느니 브랜드를 남기는 편이 학습에 안전하다)
        """
        start = 0
        while True:
            idx = text.find(brand, start)     # 다음 등장 위치 탐색
            if idx == -1:
                return text                    # 경계 만족 위치 없음 → 원문 유지

            end = idx + len(brand)
            prev_ok = idx == 0 or not _is_word_char(text[idx - 1])      # (a)
            next_ok = end == len(text) or not _is_word_char(text[end])  # (b)

            if prev_ok and next_ok and not self._overlaps_protected(text, idx, end):
                # 경계 OK + 보호명사 비침범 → 이 위치만 제거/치환
                return (text[:idx] + replacement + text[end:]).strip()

            start = idx + 1                    # 다음 등장 위치로 이동

    def _overlaps_protected(self, text: str, m_start: int, m_end: int) -> bool:
        """
        [m_start, m_end) 매칭 구간이 보호 명사의 내부에 걸치는지 검사한다. (c)

        예) text="생크림 카스텔라", brand="카스텔" (m=[4,7))
            보호명사 "카스텔라"가 [4,8)에 존재 → 매칭 구간이 내부에 포함 → True(제거 금지)
        """
        for noun in self._opt.protected_nouns:
            if len(noun) <= (m_end - m_start):
                continue          # 보호명사가 매칭보다 짧으면 '내부 포함' 불가능 → 스킵
            pos = text.find(noun)
            while pos != -1:
                # 보호명사 구간 [pos, pos+len) 안에 매칭 구간이 완전히 포함되는가
                if pos <= m_start and m_end <= pos + len(noun):
                    return True
                pos = text.find(noun, pos + 1)
        return False

    # ──────────────────────────────────────────────────────────────
    # 용량/수량 처리 (토글 + 플레이스홀더)
    # ──────────────────────────────────────────────────────────────

    def _handle_volume_and_quantity(self, text: str, force_remove: bool = False) -> str:
        """
        용량/수량 텍스트를 옵션에 따라 제거 또는 플레이스홀더로 치환하는 단계.

        force_remove=True (keep 재조립 모드):
            원문 위치의 용량/수량 표기는 코어에서 전부 제거하고,
            용량은 parse() 6단계에서 정규화된 형태(80g)로 재부착.
        placeholder=True 인 경우:
            "300g" → "VOLTOK", "4개입" → "QTYTOK" — 규격 표기 신호의 word 피처 보존용.
        """
        result = text
        if force_remove:
            result = VOLUME_PATTERN.sub("", result)
            result = QUANTITY_PATTERN.sub("", result)
            return result
        if self._opt.remove_volume:
            repl = f" {VOL_TOKEN} " if self._opt.placeholder else ""
            result = VOLUME_PATTERN.sub(repl, result)
        if self._opt.remove_quantity:
            repl = f" {QTY_TOKEN} " if self._opt.placeholder else ""
            result = QUANTITY_PATTERN.sub(repl, result)
        return result

    # ──────────────────────────────────────────────────────────────
    # 이하 기존 로직 유지 (메타데이터 추출 / 노이즈 제거 / 최종 정제)
    # ──────────────────────────────────────────────────────────────

    def _extract_brand(self, text: str) -> Optional[str]:
        """브랜드 사전(Aho-Corasick) 매칭으로 브랜드명 추출."""
        return self._brand_matcher.find_match(text)

    @staticmethod
    def _extract_volume(text: str) -> _VolumeInfo:
        """용량값과 단위를 추출한다. '리터' → 'l' 정규화 포함."""
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
        """수량 단위(개입, 팩, 봉 …) 앞 숫자를 추출한다."""
        m = QUANTITY_PATTERN.search(text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                logger.warning("수량 파싱 실패: %s", m.group(1))
        return None

    @staticmethod
    def _remove_noise_patterns(text: str) -> str:
        """순수 노이즈(가격·배송·적립·리뷰 등)를 무조건 제거한다."""
        if not text:
            return text
        result = text
        for pattern in NOISE_PATTERNS:
            result = pattern.sub("", result)
        return result

    @staticmethod
    def _final_cleanup(text: str) -> str:
        """연속 쉼표·공백 정규화, 빈 괄호 제거, 앞뒤 쉼표/공백 제거."""
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
        return result.strip()
