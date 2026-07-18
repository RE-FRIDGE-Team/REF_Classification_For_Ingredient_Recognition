"""
GIN 핵어(head) 추출 모듈 — TF-IDF에 '위치 개념'을 부여하는 위치 인지 피처.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[설계 배경 — 플레이스홀더 재개발]
  기존 placeholder(BRANDTOK/VOLTOK/QTYTOK)는 '규격 표기 존재' 신호의 단순
  치환일 뿐, 본래 의도했던 "복합어 내 마지막 GIN = 핵심 GIN" 위치 정보와
  무관한 구현이었음. 본 모듈이 그 본래 목적의 재구현체.

[언어학적 근거]
  한국어 합성명사는 우핵(right-headed) 구조 — 전체 의미 범주를 결정하는
  핵어(head)가 가장 오른쪽에 위치. 예) 김치치즈돈가스 = '돈가스'의 하위어.
  따라서 복합어 내에서 가장 오른쪽 끝에서 매칭되는 GIN 이 분류를 결정하는
  핵심 GIN 이라는 가설이 언어학적으로 정당한 구조.

[현업 기법 대응 — 위치 정보의 BoW 주입 방식]
  검색/랭킹 현업 표준은 BM25F(fielded weighting): 문서를 title/body 필드로
  나누고 중요한 필드의 term frequency 에 boost 를 곱해 단일 점수로 결합하는
  방식 (Robertson et al., CIKM 2004). 토큰에 위치를 문자열로 태깅하는
  방식(돈가스@LAST)은 어휘가 위치 수만큼 폭증해 희소 데이터에서 불리하므로,
  본 구현은 필드 분리 방식 채택:
    - gin_head 필드: 마지막 토큰의 우핵 GIN (강가중 벡터화)
    - gin_mods 필드: 그 외 위치의 GIN 수식어들 (약가중 벡터화)
  같은 단어라도 핵어 위치에서 등장하면 강한 신호, 수식어 위치면 약한 신호가
  되는 구조 — 이것이 TF-IDF 가 원천적으로 잃는 어순 정보의 복원 방식.

[동작 예]
  vocab = {김치, 치즈, 돈가스, 피자, ...}
  "김치치즈돈가스" → head="돈가스", mods=["김치", "치즈"]
  → 냉동식품(돈가스) 신호 강화, 채소(김치)로의 오인 억제.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import ahocorasick
    _AHO_AVAILABLE = True
except ImportError:
    _AHO_AVAILABLE = False


class GinHeadExtractor:
    """
    GIN/PGIN 어휘 기반 복합어 분해 + 우핵 핵어 추출기.

    Usage:
        ext = GinHeadExtractor(["김치", "치즈", "돈가스", "만두"])
        ext.decompose("김치치즈돈가스")   # → ("돈가스", ["김치", "치즈"])
        head, mods = ext.extract_fields(nouns_series, refined_series)
    """

    _MIN_LEN = 2   # 1글자 GIN 은 오매칭 위험이 커서 분해 어휘에서 제외하는 기준

    def __init__(self, vocab: Iterable[str] | None = None) -> None:
        # 분해 어휘 정리: 한글 2자 이상, 공백 포함 PGIN 은 토큰 단위로 분리 등재
        words: set[str] = set()
        for w in vocab or []:
            if not w or not isinstance(w, str):
                continue
            for tok in [w.replace(" ", ""), *w.split()]:
                tok = tok.strip()
                if len(tok) >= self._MIN_LEN and re.fullmatch(r"[가-힣]+", tok):
                    words.add(tok)
        self._vocab = words
        self._automaton = self._build_automaton(words) if words else None
        logger.info("GinHeadExtractor 초기화 — 분해 어휘 %d개 (aho=%s)",
                    len(words), self._automaton is not None)

    @staticmethod
    def _build_automaton(words: set[str]):
        """Aho-Corasick 오토마톤 빌드 (미설치 시 None → find 폴백 경로 사용)."""
        if not _AHO_AVAILABLE:
            return None
        A = ahocorasick.Automaton()
        for w in words:
            A.add_word(w, w)
        A.make_automaton()
        return A

    # ──────────────────────────────────────────────────────────────
    # 복합어 분해
    # ──────────────────────────────────────────────────────────────

    def _find_matches(self, token: str) -> list[tuple[int, int, str]]:
        """token 내부의 모든 어휘 매칭 [(start, end, word)] 수집."""
        matches: list[tuple[int, int, str]] = []
        if self._automaton is not None:
            for end_idx, w in self._automaton.iter(token):
                end = end_idx + 1
                matches.append((end - len(w), end, w))
        else:
            for w in self._vocab:
                start = 0
                while True:
                    idx = token.find(w, start)
                    if idx == -1:
                        break
                    matches.append((idx, idx + len(w), w))
                    start = idx + 1
        return matches

    def decompose(self, token: str) -> tuple[Optional[str], list[str]]:
        """
        단일 토큰의 (핵어, 수식어 리스트) 분해.

        핵어 선정 기준 — 우핵성 반영:
          1) 매칭 종료 위치(end)가 가장 오른쪽인 매칭이 핵어 후보
          2) 동률이면 더 긴 매칭 우선 (소비뇽블랑 > 블랑 류의 포함 관계 처리)
        수식어: 핵어 구간과 겹치지 않는 매칭들을 위치순 정렬, 겹침 구간은
        긴 단어 우선 선점 방식.
        핵어 미검출(어휘 무매칭) 시 (None, []) 반환 — 호출측 폴백 위임.
        """
        if not token or not self._vocab:
            return None, []
        matches = self._find_matches(token)
        if not matches:
            return None, []

        # 핵어: 종료 위치 최우선, 동률 시 길이 우선 선정
        head_start, head_end, head = max(
            matches, key=lambda m: (m[1], m[1] - m[0])
        )

        # 수식어: 핵어 앞 구간에서 긴 단어 우선으로 비중첩 선점
        mods: list[tuple[int, str]] = []
        consumed = [False] * len(token)
        for k in range(head_start, head_end):
            consumed[k] = True
        for start, end, w in sorted(matches, key=lambda m: -(m[1] - m[0])):
            if (start, end, w) == (head_start, head_end, head):
                continue
            if not any(consumed[start:end]):
                mods.append((start, w))
                for k in range(start, end):
                    consumed[k] = True
        mods.sort()
        return head, [w for _, w in mods]

    # ──────────────────────────────────────────────────────────────
    # 필드 생성 (FeatureBuilder 연동 진입점)
    # ──────────────────────────────────────────────────────────────

    def extract_fields(
        self, nouns_text: pd.Series, refined_text: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """
        행 단위 (gin_head 필드, gin_mods 필드) 시리즈 생성.

        - 마지막 명사 토큰을 분해해 핵어를 gin_head 로 배치.
          어휘 무매칭 시 마지막 토큰 자체를 핵어로 폴백 (기존 head-noun 동작 계승).
        - 마지막 토큰의 내부 수식어 + 앞쪽 토큰들의 GIN 매칭을 gin_mods 로 집계.
        """
        heads: list[str] = []
        mods_out: list[str] = []
        for nouns, refined in zip(nouns_text.values, refined_text.values):
            toks = str(nouns).split() if pd.notna(nouns) and str(nouns).strip() else []
            if not toks:
                toks = str(refined).split() if pd.notna(refined) else []
            if not toks:
                heads.append("")
                mods_out.append("")
                continue

            last = toks[-1]
            head, last_mods = self.decompose(last)
            if head is None:
                head = last                               # 무매칭 폴백: 토큰 통짜 핵어 처리

            # 앞쪽 토큰들: 내부 GIN 매칭 전부 수식어 취급 (무매칭이면 토큰 통짜)
            front_mods: list[str] = []
            for t in toks[:-1]:
                h, ms = self.decompose(t)
                if h is None:
                    front_mods.append(t)
                else:
                    front_mods.extend(ms + [h])

            heads.append(head)
            mods_out.append(" ".join(front_mods + last_mods))

        return (
            pd.Series(heads, index=nouns_text.index),
            pd.Series(mods_out, index=nouns_text.index),
        )
