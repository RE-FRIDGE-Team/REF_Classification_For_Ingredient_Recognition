"""
도메인 어휘 사전(lexicon) 모듈 — 술 카테고리 정면 대응용.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[설계 배경]
  StratifiedGroupKFold 오류 분석 결과, macro_F1 최대 누수처는 '술' 대분류였다.
  "배비치 블랙라벨 소비뇽블랑", "닛카 프롬 더 배럴" 같은 제품명은
  브랜드 고유명사 위주라 char n-gram이 일반화 신호를 얻지 못한다.

  대응: 출처(브랜드)와 무관하게 일반화되는 **품종/주종 '스타일어' 사전**을
  만들어 카운트 피처로 주입한다. 브랜드 사전이 아니므로 새 브랜드가 나와도
  "말벡", "배럴" 같은 스타일어만 있으면 걸린다.

[파이프라인 '사전 시스템'과의 중복 여부 — 수레바퀴 재발명 아님]
  5단계 인식 파이프라인의 사전(PGIN Aho-Corasick, 브랜드 사전)은
  '제품명 → GIN 직접 매칭'용이다. 본 lexicon은 GIN이 아니라
  **카테고리 판별 신호(스타일어)** 로, 개념적으로 다른 사전이다.
  (예: "말벡"은 PGIN이 아니지만 '술>와인'을 강하게 지시한다)

[사람 개입 최소화 — 5순위 원칙]
  시드는 코드에 내장되어 별도 문서 작성 없이 즉시 동작한다.
  확장이 필요하면 resources/alcohol_style_words.txt 에
  `카테고리<TAB>단어` 한 줄씩만 추가하면 자동 병합된다.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 술 스타일어 시드 사전 ───────────────────────────────────────────
# key   : 스타일어 그룹(주종) — 그룹별 카운트가 각각 하나의 피처 차원이 된다.
# value : 해당 주종을 지시하는 '일반화 가능한' 어휘 (브랜드명 금지!)
ALCOHOL_STYLE_LEXICON: dict[str, tuple[str, ...]] = {
    # 와인 — 포도 품종 / 색·유형 / 산지 등급 표기
    "wine": (
        "소비뇽", "소비뇽블랑", "블랑", "샤도네이", "샤르도네", "메를로", "멜롯",
        "말벡", "리즐링", "피노누아", "피노", "시라", "쉬라즈", "카베르네",
        "까베르네", "카버네", "템프라니요", "산지오베제", "모스카토", "람브루스코",
        "브뤼", "브륏", "샴페인", "스파클링", "로제", "와인", "빈티지", "리제르바",
        "리저브와인", "그랑크뤼", "도멘", "샤또", "샤토", "뀌베", "쿠베",
    ),
    # 위스키 — 증류/숙성 용어
    "whisky": (
        "위스키", "버번", "배럴", "싱글배럴", "싱글몰트", "몰트", "블렌디드",
        "캐스크", "쉐리캐스크", "피트", "스카치", "리저브", "숙성", "년산",
        "라이위스키", "테네시", "하이랜드", "아일라",
    ),
    # 사케/일본주
    "sake": (
        "사케", "쥰마이", "준마이", "혼죠조", "혼조조", "긴죠", "다이긴죠",
        "니고리", "우메슈", "청주",
    ),
    # 증류주/리큐르 일반
    "spirits": (
        "보드카", "진", "데킬라", "테킬라", "럼", "리큐르", "리큐어",
        "브랜디", "꼬냑", "코냑", "아페리티프", "하이볼", "칵테일",
    ),
    # 맥주
    "beer": (
        "맥주", "라거", "에일", "IPA", "스타우트", "필스너", "바이젠", "페일에일",
    ),
    # 전통주
    "traditional": (
        "막걸리", "약주", "탁주", "소주", "증류식", "과실주", "복분자주",
    ),
}

_EXTRA_PATH_DEFAULT = "resources/alcohol_style_words.txt"


def load_alcohol_lexicon(
    extra_path: str | Path | None = _EXTRA_PATH_DEFAULT,
) -> dict[str, tuple[str, ...]]:
    """
    술 스타일어 사전을 로드한다 (내장 시드 + 선택적 확장 파일 병합).

    확장 파일 형식 (resources/alcohol_style_words.txt):
        wine\t그르나슈
        whisky\t버번배럴
        # 로 시작하는 줄은 주석

    Returns:
        {그룹명: (단어, ...)} — 그룹은 시드에 없던 새 그룹도 추가 가능
    """
    lex: dict[str, list[str]] = {k: list(v) for k, v in ALCOHOL_STYLE_LEXICON.items()}

    if extra_path:
        p = Path(extra_path)
        if p.exists():
            added = 0
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue                       # 빈 줄·주석 스킵
                parts = line.split("\t")
                if len(parts) != 2:
                    logger.warning("스타일어 확장 파일 형식 오류 무시: %r", line)
                    continue
                group, word = parts[0].strip(), parts[1].strip()
                lex.setdefault(group, []).append(word)
                added += 1
            logger.info("술 스타일어 확장 파일 병합: %s (+%d개)", p, added)

    # 중복 제거 + 긴 단어 우선 정렬(부분 문자열 매칭 시 긴 것부터 확인하기 위함)
    return {
        g: tuple(sorted(set(words), key=len, reverse=True))
        for g, words in lex.items()
    }
