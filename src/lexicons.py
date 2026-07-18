"""
도메인 어휘 사전(lexicon) 모듈 — 술 카테고리 정면 대응용 (2026-07 2차 개편).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[2차 개편 요약 — 집중도 리포트 반영]

1. 저집중도 스타일어 제거
   실측 집중도(술빈도/전체빈도) 리포트에서 보편 상품에 침투하는 것으로
   확인된 단어 삭제: 년산(0.00), 로제(0.00), 진(0.10), 숙성(0.22),
   스파클링(0.76 — 스파클링 워터 오염).

2. 스타일어 가드(blocker/boundary) 도입
   '칵테일 새우' 사태의 일반해. 단어 자체는 정당하나 특정 공기어와
   함께 나오면 술 신호가 아닌 케이스를 단어별 blocker 로 차단하는 구조.
     - 칵테일: 새우/홍합 동반 시 미발동
     - 에일:   진저에일(음료) 차단
     - 막걸리/와인: 식초 가공품 차단
     - 브뤼:   브뤼셀(스프라우트) 차단
     - 럼:     '크럼블' 등 단어 내부 부분매칭 차단 (boundary=True)
   blocker 는 해당 단어 하나만 침묵시키므로, 같은 문장의 다른 스타일어
   ("리큐르" 등)는 정상 발동 — 과차단 위험이 없는 국소 억제 방식.

3. 술 브랜드 가제티어 신설 — '봄베이 사파이어' 문제 대응
   스타일어가 전혀 없는 순수 브랜드형 제품명은 사전 신호로만 잡을 수 있음.
     (a) 큐레이션 시드: 국내외 주류 브랜드 내장 리스트
     (b) 데이터 마이닝: tools/mine_alcohol_brands.py 가 학습 데이터에서
         브랜드별 술 집중도를 계산, min_support·min_concentration 필터를
         통과한 브랜드만 resources/alcohol_brands.txt 로 export
   집중도 필터(기본 0.8)가 오리지널(0.10)·청정원(0.02)·웅진빅토리아(0.33)·
   김소형원방(0.20)·하이트진로(0.40, 음료 겸업) 류를 자동 배제하는 설계.

[사람 개입 최소화 — 5순위 원칙]
  시드는 코드 내장. 확장 파일(resources/alcohol_style_words.txt,
  resources/alcohol_brands.txt)이 존재하면 자동 병합.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# 1. 술 스타일어 — 가드 메타데이터 포함 구조
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StyleWord:
    """
    단일 스타일어 항목.

    Attributes:
        word:     스타일어 표층형
        group:    주종 그룹 (wine/whisky/sake/spirits/beer/traditional)
        blockers: 텍스트에 하나라도 공기하면 이 단어만 미발동시키는 차단어 목록
        boundary: True 면 앞글자가 한글일 때 매칭 금지 (단어 내부 부분매칭 차단용)
    """
    word: str
    group: str
    blockers: tuple[str, ...] = ()
    boundary: bool = False


# ── 스타일어 시드 ──────────────────────────────────────────────────
# 제거: 년산·로제·진·숙성·스파클링 (집중도 리포트 기준 저집중 확정)
# 가드: 집중도 1.0 미만이지만 규칙 자체는 정당한 단어들에 blocker/boundary 부여
_STYLE_SEED: tuple[StyleWord, ...] = (
    # ── 와인: 품종/색·유형/산지 등급 표기 ──
    StyleWord("소비뇽블랑", "wine"),
    StyleWord("소비뇽", "wine"),
    StyleWord("블랑", "wine"),
    StyleWord("샤도네이", "wine"),
    StyleWord("샤르도네", "wine"),
    StyleWord("메를로", "wine"),
    StyleWord("멜롯", "wine"),
    StyleWord("말벡", "wine"),
    StyleWord("리즐링", "wine"),
    StyleWord("피노누아", "wine"),
    StyleWord("피노", "wine"),
    StyleWord("시라", "wine", boundary=True),          # '카시라' 류 내부매칭 방지
    StyleWord("쉬라즈", "wine"),
    StyleWord("카베르네", "wine"),
    StyleWord("까베르네", "wine"),
    StyleWord("카버네", "wine"),
    StyleWord("템프라니요", "wine"),
    StyleWord("산지오베제", "wine"),
    StyleWord("모스카토", "wine"),
    StyleWord("람브루스코", "wine"),
    StyleWord("브뤼", "wine", blockers=("브뤼셀",)),    # 브뤼셀 스프라우트 차단
    StyleWord("브륏", "wine"),
    StyleWord("샴페인", "wine"),
    StyleWord("와인", "wine", blockers=("식초", "비네거")),  # 와인식초 차단
    StyleWord("빈티지", "wine"),
    StyleWord("리제르바", "wine"),
    StyleWord("리저브와인", "wine"),
    StyleWord("그랑크뤼", "wine"),
    StyleWord("도멘", "wine"),
    StyleWord("샤또", "wine"),
    StyleWord("샤토", "wine"),
    StyleWord("뀌베", "wine"),
    StyleWord("쿠베", "wine"),
    # ── 위스키: 증류/숙성 용어 (년산·숙성 제거) ──
    StyleWord("위스키", "whisky"),
    StyleWord("버번", "whisky"),
    StyleWord("싱글배럴", "whisky"),
    StyleWord("싱글몰트", "whisky"),
    StyleWord("배럴", "whisky"),
    StyleWord("몰트", "whisky", blockers=("몰티즈",)),
    StyleWord("블렌디드", "whisky", blockers=("커피", "라떼", "티백", "주스", "쥬스", "음료")),
    StyleWord("캐스크", "whisky"),
    StyleWord("쉐리캐스크", "whisky"),
    StyleWord("피트", "whisky", boundary=True),
    StyleWord("스카치", "whisky", blockers=("버터스카치", "캔디", "사탕")),
    StyleWord("리저브", "whisky"),
    StyleWord("라이위스키", "whisky"),
    StyleWord("테네시", "whisky"),
    StyleWord("하이랜드", "whisky"),
    StyleWord("아일라", "whisky"),
    # ── 사케/일본주 ──
    StyleWord("사케", "sake"),
    StyleWord("쥰마이", "sake"),
    StyleWord("준마이", "sake"),
    StyleWord("혼죠조", "sake"),
    StyleWord("혼조조", "sake"),
    StyleWord("긴죠", "sake"),
    StyleWord("다이긴죠", "sake"),
    StyleWord("니고리", "sake"),
    StyleWord("우메슈", "sake"),
    StyleWord("청주", "sake", blockers=("식초",)),
    # ── 증류주/리큐르 (진 제거) ──
    StyleWord("보드카", "spirits"),
    StyleWord("데킬라", "spirits"),
    StyleWord("테킬라", "spirits"),
    StyleWord("럼", "spirits", boundary=True),          # 크럼블 등 내부매칭 차단
    StyleWord("리큐르", "spirits"),
    StyleWord("리큐어", "spirits"),
    StyleWord("브랜디", "spirits"),
    StyleWord("꼬냑", "spirits"),
    StyleWord("코냑", "spirits"),
    StyleWord("아페리티프", "spirits"),
    StyleWord("하이볼", "spirits"),
    StyleWord("칵테일", "spirits", blockers=("새우", "홍합", "쉬림프")),  # 칵테일새우 차단
    # ── 맥주 ──
    StyleWord("맥주", "beer"),
    StyleWord("라거", "beer"),
    StyleWord("에일", "beer", blockers=("진저",)),       # 진저에일(음료) 차단
    StyleWord("페일에일", "beer"),
    StyleWord("IPA", "beer"),
    StyleWord("스타우트", "beer"),
    StyleWord("필스너", "beer"),
    StyleWord("바이젠", "beer"),
    # ── 전통주 ──
    StyleWord("막걸리", "traditional", blockers=("식초",)),  # 막걸리식초 차단
    StyleWord("약주", "traditional"),
    StyleWord("탁주", "traditional"),
    StyleWord("소주", "traditional", blockers=("식초",)),
    StyleWord("증류식", "traditional"),
    StyleWord("과실주", "traditional"),
    StyleWord("복분자주", "traditional"),
)

_STYLE_EXTRA_PATH = "resources/alcohol_style_words.txt"


def load_style_words(extra_path: str | Path | None = _STYLE_EXTRA_PATH) -> tuple[StyleWord, ...]:
    """
    스타일어 목록 로드 (내장 시드 + 선택적 확장 파일 병합).

    확장 파일 형식 (탭 구분, 3열 이후 선택):
        wine\t그르나슈
        spirits\t칵테일\t새우,홍합        ← 3열 = 쉼표 구분 blocker 목록
        spirits\t럼\t\tboundary          ← 4열 = boundary 플래그
    """
    words: list[StyleWord] = list(_STYLE_SEED)
    if extra_path:
        p = Path(extra_path)
        if p.exists():
            added = 0
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    logger.warning("스타일어 확장 파일 형식 오류 무시: %r", line)
                    continue
                group, word = parts[0].strip(), parts[1].strip()
                blockers = tuple(
                    b.strip() for b in parts[2].split(",") if b.strip()
                ) if len(parts) >= 3 and parts[2].strip() else ()
                boundary = len(parts) >= 4 and parts[3].strip().lower() == "boundary"
                words.append(StyleWord(word, group, blockers, boundary))
                added += 1
            logger.info("스타일어 확장 파일 병합: %s (+%d개)", p, added)

    # 중복 제거 (뒤에 로드된 항목 우선 = 확장 파일이 시드 오버라이드 가능)
    dedup: dict[str, StyleWord] = {}
    for sw in words:
        dedup[sw.word] = sw
    return tuple(dedup.values())


class StyleWordMatcher:
    """
    가드(blocker/boundary)를 반영한 스타일어 매처.

    count_groups(text) → {그룹: 발동 횟수}
      - blocker 가 텍스트에 공기하면 해당 단어만 미발동
      - boundary=True 단어는 앞글자가 한글이면 미발동 (단어 내부 부분매칭 차단)
      - 긴 단어 우선 매칭 후 소비 구간 마스킹 → '소비뇽블랑'과 '소비뇽' 이중
        카운트 방지 (기존 substring 방식의 이중 카운트 문제 해결)
    """

    def __init__(self, style_words: tuple[StyleWord, ...] | None = None) -> None:
        self._words = sorted(
            style_words or load_style_words(),
            key=lambda sw: len(sw.word), reverse=True,     # 긴 단어 우선 매칭 순서
        )
        self.groups: tuple[str, ...] = tuple(sorted({sw.group for sw in self._words}))
        self._group_idx = {g: i for i, g in enumerate(self.groups)}

    def count_groups(self, text: str) -> list[int]:
        """텍스트 1건에 대한 그룹별 발동 횟수 벡터 (groups 순서)."""
        counts = [0] * len(self.groups)
        if not text:
            return counts
        consumed = [False] * len(text)                     # 이중 카운트 방지 마스크

        for sw in self._words:
            if sw.blockers and any(b in text for b in sw.blockers):
                continue                                   # 공기 차단어 존재 → 이 단어만 침묵
            start = 0
            while True:
                idx = text.find(sw.word, start)
                if idx == -1:
                    break
                end = idx + len(sw.word)
                boundary_ok = not (
                    sw.boundary and idx > 0 and re.match(r"[가-힣]", text[idx - 1])
                )
                if boundary_ok and not any(consumed[idx:end]):
                    counts[self._group_idx[sw.group]] += 1
                    for k in range(idx, end):
                        consumed[k] = True
                start = end
        return counts


# ══════════════════════════════════════════════════════════════════
# 2. 술 브랜드 가제티어 — '봄베이 사파이어'형 무단서 제품명 대응
# ══════════════════════════════════════════════════════════════════

# 큐레이션 시드: 스타일어 부재 시에도 술임을 확정할 수 있는 국내외 주류 브랜드
ALCOHOL_BRAND_SEED: tuple[str, ...] = (
    # 진/보드카/럼/데킬라
    "봄베이사파이어", "봄베이 사파이어", "탱커레이", "헨드릭스", "고든스",
    "앱솔루트", "스미노프", "그레이구스", "바카디", "캡틴모건", "말리부",
    "호세쿠엘보", "패트론", "돈훌리오",
    # 위스키
    "조니워커", "발렌타인", "시바스리갈", "글렌피딕", "글렌리벳", "맥캘란",
    "발베니", "글렌모렌지", "몽키숄더", "라프로익", "아드벡", "제임슨",
    "잭다니엘", "짐빔", "와일드터키", "메이커스마크", "버팔로트레이스",
    "산토리", "히비키", "야마자키", "하쿠슈", "가쿠빈",
    # 리큐르/기타
    "예거마이스터", "깔루아", "베일리스", "디사론노", "디카이퍼",
    # 와인/샴페인
    "모엣샹동", "돔페리뇽", "뵈브클리코", "옐로우테일", "몬테스", "1865",
    "킴크로포드", "빌라엠", "까시에로델디아블로",
    # 맥주
    "하이네켄", "버드와이저", "칭따오", "아사히", "삿포로", "기린이치방",
    "산미구엘", "호가든", "스텔라아르투아", "코로나엑스트라", "구스아일랜드",
    "제주맥주", "곰표맥주",
    # 소주/국산
    "참이슬", "처음처럼", "진로이즈백", "새로", "한라산소주", "화요", "일품진로",
    "카스", "테라", "켈리", "클라우드", "필라이트", "필굿",
    # 전통주/사케
    "복순도가", "지평막걸리", "느린마을", "국순당", "백세주", "월계관", "초야",
)

# 술 브랜드 자격 박탈 대상 — 집중도와 무관하게 항상 배제하는 일반어/겸업 브랜드
ALCOHOL_BRAND_BLOCKLIST: frozenset[str] = frozenset({
    "오리지널", "오리지날", "클래식", "프리미엄", "스페셜", "골드", "리얼",
    "청정원", "웅진빅토리아", "웅진 빅토리아", "김소형원방", "전통주",  # '전통주'는 브랜드가 아닌 일반어
})

_BRAND_EXTRA_PATH = "resources/alcohol_brands.txt"


def load_alcohol_brands(extra_path: str | Path | None = _BRAND_EXTRA_PATH) -> tuple[str, ...]:
    """
    술 브랜드 가제티어 로드 (시드 + 마이닝 산출 파일 병합, blocklist 필터 적용).

    확장 파일: tools/mine_alcohol_brands.py 가 생성하는 한 줄 한 브랜드 텍스트.
    """
    brands: set[str] = set(ALCOHOL_BRAND_SEED)
    if extra_path:
        p = Path(extra_path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                w = line.strip()
                if w and not w.startswith("#"):
                    brands.add(w)
            logger.info("술 브랜드 마이닝 파일 병합: %s", p)
    brands -= ALCOHOL_BRAND_BLOCKLIST
    # 긴 브랜드 우선 정렬 매칭 순서
    return tuple(sorted(brands, key=len, reverse=True))


class AlcoholBrandMatcher:
    """
    술 브랜드 가제티어 매처 — 토큰 경계 검사 포함.

    '카스'가 '카스텔라' 내부에 매칭되는 사고 방지를 위해 매칭 구간 양옆이
    단어 문자(한글/영문/숫자)가 아닐 때만 발동 (parser 의 브랜드 제거와 동일 원칙).
    """

    _WORD_CHAR = re.compile(r"[0-9a-zA-Z가-힣]")

    def __init__(self, brands: tuple[str, ...] | None = None) -> None:
        self._brands = brands or load_alcohol_brands()

    def count(self, text: str) -> int:
        """텍스트 내 경계 검증을 통과한 술 브랜드 발동 횟수."""
        if not text:
            return 0
        n_hit = 0
        for brand in self._brands:
            start = 0
            while True:
                idx = text.find(brand, start)
                if idx == -1:
                    break
                end = idx + len(brand)
                prev_ok = idx == 0 or not self._WORD_CHAR.match(text[idx - 1])
                next_ok = end == len(text) or not self._WORD_CHAR.match(text[end])
                if prev_ok and next_ok:
                    n_hit += 1
                    start = end
                else:
                    start = idx + 1
        return n_hit


def mine_alcohol_brands(
    brand_series,
    large_series,
    alcohol_label: str = "술",
    min_support: int = 2,
    min_concentration: float = 0.8,
) -> list[tuple[str, int, int, float]]:
    """
    학습 데이터의 (파싱 브랜드, 대분류) 쌍에서 술 집중도 기준으로 브랜드 마이닝.

    Returns:
        [(브랜드, 술빈도, 전체빈도, 집중도)] — 필터 통과분만, 집중도 내림차순.
        기본 필터(support≥2, concentration≥0.8)가 오리지널(0.10)·청정원(0.02)·
        웅진빅토리아(0.33)·김소형원방(0.20)·하이트진로(0.40) 류 자동 배제.
    """
    from collections import Counter
    total: Counter = Counter()
    alcohol: Counter = Counter()
    for brand, large in zip(brand_series, large_series):
        if not brand or not isinstance(brand, str):
            continue
        b = brand.strip()
        if not b or b in ALCOHOL_BRAND_BLOCKLIST:
            continue
        total[b] += 1
        if str(large).strip() == alcohol_label:
            alcohol[b] += 1

    mined: list[tuple[str, int, int, float]] = []
    for b, n_alc in alcohol.items():
        n_all = total[b]
        conc = n_alc / n_all
        if n_alc >= min_support and conc >= min_concentration:
            mined.append((b, n_alc, n_all, conc))
    mined.sort(key=lambda x: (-x[3], -x[1]))
    return mined


# ══════════════════════════════════════════════════════════════════
# 3. 하위 호환 API — 기존 features.py 임포트 경로 유지용
# ══════════════════════════════════════════════════════════════════

# 구버전 dict 뷰 (가드 정보 없는 평면 사전) — Mecab 사용자사전 빌더 등에서 사용
ALCOHOL_STYLE_LEXICON: dict[str, tuple[str, ...]] = {}
for _sw in _STYLE_SEED:
    ALCOHOL_STYLE_LEXICON.setdefault(_sw.group, ())
    ALCOHOL_STYLE_LEXICON[_sw.group] = ALCOHOL_STYLE_LEXICON[_sw.group] + (_sw.word,)


def load_alcohol_lexicon(extra_path: str | Path | None = _STYLE_EXTRA_PATH) -> dict[str, tuple[str, ...]]:
    """구버전 호환 로더 — 가드 정보를 제외한 {그룹: (단어,...)} 평면 뷰 반환."""
    lex: dict[str, list[str]] = {}
    for sw in load_style_words(extra_path):
        lex.setdefault(sw.group, []).append(sw.word)
    return {g: tuple(sorted(set(w), key=len, reverse=True)) for g, w in lex.items()}
