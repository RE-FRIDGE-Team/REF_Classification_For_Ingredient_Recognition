"""
공용 전처리 모듈 (2026-07 개편판).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[개편 요약]

1. 파싱 정책 토글화 (PreprocessOptions → ParserOptions 전달)
   - remove_brand / remove_volume / remove_quantity : ON/OFF
   - placeholder : 제거 대신 BRANDTOK/VOLTOK/QTYTOK 치환
   ※ 가격·배송·적립·리뷰수 등 '순수 노이즈'는 토글 없이 무조건 제거.

2. 형태소 분석기 전환: KoNLPy Okt → Mecab-ko + 자동 사용자사전
   - python-mecab-ko (pip) 사용. 시스템 mecab 설치 불필요.
   - 사용자사전은 코드가 '자동 생성·자동 컴파일'한다 (사람 개입 0):
       (a) 보호 명사 시드 (탕수육·카스텔라·건면 …)
       (b) 술 스타일어 사전 (소비뇽블랑·싱글배럴 …)
       (c) 데이터셋의 PGIN/GIN 어휘 (호출 측에서 domain_words 로 주입)
     → resources/mecab_userdic.csv 생성 → `python -m mecab dict-index` 로
       컴파일 → MeCab(user_dictionary_path=...) 로 로드.
     단어 집합의 해시가 같으면 재컴파일을 생략한다(캐시).
   - Okt 는 하위 호환용으로 유지 (--morpheme okt).

3. Okt/Mecab 모두 '명사 추출' 시 플레이스홀더 토큰(영문)을 보존하도록
   Mecab 은 pos 태그 {NNG, NNP, SL} 를 채택한다.

4. post_refine() 2차 정제는 기존 그대로 유지.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ── 선택적 임포트 (환경별 대응) ─────────────────────────────────────
try:
    from konlpy.tag import Okt
    _OKT_AVAILABLE = True
except Exception:
    _OKT_AVAILABLE = False

try:
    from mecab import MeCab            # python-mecab-ko
    _MECAB_AVAILABLE = True
except Exception:
    _MECAB_AVAILABLE = False

try:
    from product_name_parser import (
        ParserOptions,
        REFProductNameParser,
        create_brand_matcher,
    )
    _PARSER_AVAILABLE = True
except ImportError:
    _PARSER_AVAILABLE = False
    logger.warning("product_name_parser 임포트 실패. 원본 product_name을 그대로 사용합니다.")

from src.lexicons import ALCOHOL_STYLE_LEXICON

ALCOHOL_LARGE = "술"

# Mecab 사용자사전 산출물 경로 (자동 생성 — 사람이 만들 필요 없음)
_USERDIC_CSV = Path("resources/mecab_userdic.csv")
_USERDIC_DIC = Path("resources/mecab_userdic.dic")
_USERDIC_HASH = Path("resources/mecab_userdic.hash")


# ──────────────────────────────────────────────────────────────────
# 2차 정제 패턴 — 기존 유지 (파서 이후 잔재 노이즈 제거)
# ──────────────────────────────────────────────────────────────────
_POST_REFINE_PATTERNS: List[re.Pattern] = [
    # 단가 괄호 잔재: (100g당 ), (1정당 ), (당 ) 등
    # ★단위 목록 보강: 가격 노이즈 패턴(인덱스 3)이 "(1정당 1,057원)"의 금액만
    #   지우고 남긴 "(1정당 )" 류를 마저 제거하기 위해 정/구/본/매/팩/세트 추가
    re.compile(r'\(\s*(?:\d+(?:\.\d+)?(?:g|kg|ml|l|개|원|정|구|본|매|팩|세트))?\s*당\s*\)'),
    # 홍보 문구 잔재
    re.compile(r'최대\s*\d+[원\w]*\s*적립'),
    re.compile(r'최대\s*적립'),
    re.compile(r'최저\s*\d*[원\w]*'),
    re.compile(r'최대~'),
    # 부속 설명 괄호: (소스 포함), (증정) 등
    re.compile(r'\([^)]*(?:포함|동봉|증정)\)'),
    # 단독 할인율 잔재: NN%
    re.compile(r'(?<!\w)\d+%(?!\s*\()'),
    # 빈 의미 괄호: ( + ), ( X 4) 등
    re.compile(r'\(\s*[+\-×xX*/÷|,\s\d]*\s*\)'),
    # 끝 단독 특수문자 / *숫자단위
    re.compile(r'\s+[*×xX]\s*$'),
    re.compile(r'\*\d+[가-힣a-zA-Z]*$'),
    # 분수/슬래시 잔재
    re.compile(r'(?<!\))\s+\d+/\s*$'),
    re.compile(r'(?<![가-힣a-zA-Z(])\d+/\s*$'),
]


def post_refine(text: str) -> str:
    """파서 적용 이후 남은 잔재 노이즈(단가 괄호·홍보 문구 등)를 제거한다."""
    if not text:
        return text
    result = text.replace('\xa0', ' ')          # non-breaking space 정규화
    for pat in _POST_REFINE_PATTERNS:
        result = pat.sub('', result)
    result = re.sub(r'\(\s*\)', '', result)     # 패턴 적용 후 생긴 빈 괄호
    result = re.sub(r'\[\s*\]', '', result)
    result = re.sub(r'\s+', ' ', result).strip()
    result = re.sub(r'[\s,.\-]+$', '', result).strip()
    return result


# ──────────────────────────────────────────────────────────────────
# Mecab 자동 사용자사전 빌더
# ──────────────────────────────────────────────────────────────────

def _has_jongseong(word: str) -> str:
    """
    mecab-ko-dic CSV의 has_jongseong(받침 유무) 필드값 T/F 를 계산한다.
    (한글 음절 = 0xAC00 + 초성*588 + 중성*28 + 종성 → 종성 = (code-0xAC00) % 28)
    """
    ch = word.strip()[-1]
    if '가' <= ch <= '힣':
        return 'T' if (ord(ch) - ord('가')) % 28 != 0 else 'F'
    return 'F'


def build_mecab_userdic(domain_words: Iterable[str]) -> Optional[Path]:
    """
    도메인 명사 집합으로 Mecab 사용자사전(.dic)을 자동 생성한다.

    입력 단어 = (보호 명사) ∪ (술 스타일어) ∪ (호출측 domain_words: PGIN 어휘 등).
    - 한글 2자 이상 단어만 등재 (1자·영문은 시스템 사전이 이미 잘 처리)
    - 단어 집합 해시가 이전과 같으면 재컴파일 생략 (캐시)
    - 컴파일: `python -m mecab dict-index --userdic <out.dic> <in.csv>`
      (python-mecab-ko 공식 문서의 사용자사전 빌드 절차)

    Returns:
        컴파일된 .dic 경로. mecab 미설치/컴파일 실패 시 None (기본 사전으로 동작).
    """
    if not _MECAB_AVAILABLE:
        return None

    # 1. 등재 단어 수집 — 보호명사 + 술 스타일어 + 호출측 도메인 어휘
    from product_name_parser.parser import _PROTECTED_NOUNS_SEED
    words: set[str] = set(_PROTECTED_NOUNS_SEED)
    for group_words in ALCOHOL_STYLE_LEXICON.values():
        words.update(group_words)
    for w in domain_words:
        if w and isinstance(w, str):
            # PGIN이 "냉동 치킨"처럼 공백 포함이면 각 토큰과 붙인 형태 모두 등재
            for tok in [w.replace(" ", ""), *w.split()]:
                words.add(tok.strip())

    # 한글 2자 이상만 등재 (mecab CSV는 한글 표층형 기준으로 작성)
    words = {w for w in words if len(w) >= 2 and re.fullmatch(r"[가-힣]+", w)}
    if not words:
        return None

    # 2. 캐시 검사 — 단어 집합 해시가 같으면 기존 .dic 재사용
    digest = hashlib.sha256("\n".join(sorted(words)).encode()).hexdigest()
    if _USERDIC_DIC.exists() and _USERDIC_HASH.exists():
        if _USERDIC_HASH.read_text().strip() == digest:
            logger.info("Mecab 사용자사전 캐시 재사용: %s (%d단어)", _USERDIC_DIC, len(words))
            return _USERDIC_DIC

    # 3. CSV 생성 — mecab-ko-dic 형식:
    #    표층형,,,비용,품사,의미,받침유무,읽기,타입,첫품사,끝품사,표현
    #    비용 0 = 최우선 (기존 분절보다 사용자 단어가 항상 이김)
    _USERDIC_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(_USERDIC_CSV, "w", encoding="utf-8") as f:
        for w in sorted(words):
            f.write(f"{w},,,0,NNG,*,{_has_jongseong(w)},{w},*,*,*,*\n")

    # 4. 컴파일 — python -m mecab dict-index
    try:
        subprocess.run(
            [sys.executable, "-m", "mecab", "dict-index",
             "--userdic", str(_USERDIC_DIC), str(_USERDIC_CSV)],
            check=True, capture_output=True, text=True,
        )
        _USERDIC_HASH.write_text(digest)
        logger.info("Mecab 사용자사전 컴파일 완료: %s (%d단어)", _USERDIC_DIC, len(words))
        return _USERDIC_DIC
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("사용자사전 컴파일 실패 — 기본 사전으로 진행: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────
# 전처리 옵션
# ──────────────────────────────────────────────────────────────────

@dataclass
class PreprocessOptions:
    """
    전처리 A/B 실험 토글 모음 (CLI 인수 → 이 객체 → 파서/형태소기로 전달).

    Attributes:
        remove_brand:      브랜드 제거 ON/OFF  (OFF = 브랜드 원문 유지 실험)
        remove_volume:     용량 제거 ON/OFF
        remove_quantity:   수량 제거 ON/OFF
        placeholder:       제거 대신 BRANDTOK/VOLTOK/QTYTOK 치환
        morpheme_analyzer: "mecab"(신규 기본) | "okt"(구버전) | "none"
        alcohol_brand_preserve: 술 카테고리는 브랜드 제거를 항상 건너뜀
    """
    remove_brand:    bool = True
    remove_volume:   bool = True
    remove_quantity: bool = True
    placeholder:     bool = False
    morpheme_analyzer: str = "mecab"
    alcohol_brand_preserve: bool = True


class REFPreprocessor:
    """
    RE:FRIDGE 제품명 전처리기 (토글판).

    fit_transform(df) 결과 컬럼:
        refined_text : 파서 정제 + post_refine 2차 정제 (토글 반영)
        nouns_text   : 형태소 명사열 (Mecab 사용자사전 or Okt)
        label_large / label_medium / label_tag : LabelEncoder 정수 인코딩
    """

    def __init__(
        self,
        brand_dict_path: str = "product_data_collection/not_grocery_and_brand_list/grocery_brand_name.json",
        stopwords: list[str] | None = None,
        options: PreprocessOptions | None = None,
        use_parser: bool = True,
        domain_words: Iterable[str] | None = None,
    ) -> None:
        self._opt = options or PreprocessOptions()
        self._stopwords: set[str] = set(stopwords or [])
        self._use_parser = use_parser and _PARSER_AVAILABLE

        # LabelEncoder (fit 전까지 None)
        self._le_large: LabelEncoder | None = None
        self._le_medium: LabelEncoder | None = None
        self._le_tag: LabelEncoder | None = None

        # ── 파서 2종 초기화 (일반용 / 술용=브랜드 미제거) ──
        self._parser = None
        self._parser_no_brand = None
        if self._use_parser:
            self._parser, self._parser_no_brand = self._init_parsers(brand_dict_path)

        # ── 형태소 분석기 초기화 ──
        self._okt = None
        self._mecab = None
        analyzer = self._opt.morpheme_analyzer.lower()

        if analyzer == "mecab":
            if _MECAB_AVAILABLE:
                # 사용자사전 자동 빌드 (도메인 어휘 = PGIN/GIN 등, 호출측 주입)
                userdic = build_mecab_userdic(domain_words or [])
                try:
                    self._mecab = (
                        MeCab(user_dictionary_path=[str(userdic)]) if userdic else MeCab()
                    )
                    logger.info("Mecab 초기화 완료 (userdic=%s)", bool(userdic))
                except Exception as e:
                    logger.warning("Mecab 초기화 실패 → 형태소 분석 생략: %s", e)
            else:
                logger.warning("python-mecab-ko 미설치 → 형태소 분석 생략. pip install python-mecab-ko")
        elif analyzer == "okt":
            if _OKT_AVAILABLE:
                self._okt = Okt()
                logger.info("Okt 형태소 분석기 초기화 완료")
            else:
                logger.warning("KoNLPy Okt 초기화 실패 → 형태소 분석 생략")
        # analyzer == "none" 이면 nouns_text = refined_text (char n-gram 단독 실험용)

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """refined_text / nouns_text / label_* 컬럼을 추가한 새 DataFrame 반환."""
        result = df.copy()

        logger.info("제품명 정제 시작 (%d개) — 옵션: %s", len(result), self._opt)
        result["refined_text"] = result.apply(self._refine_row, axis=1)

        logger.info("명사 추출 시작 (analyzer=%s)", self._opt.morpheme_analyzer)
        result["nouns_text"] = result["refined_text"].apply(self._extract_nouns)

        logger.info("레이블 인코딩")
        self._le_large  = LabelEncoder().fit(result["large_category"])
        self._le_medium = LabelEncoder().fit(result["medium_category"])
        self._le_tag    = LabelEncoder().fit(result["category_tag"])
        result["label_large"]  = self._le_large.transform(result["large_category"])
        result["label_medium"] = self._le_medium.transform(result["medium_category"])
        result["label_tag"]    = self._le_tag.transform(result["category_tag"])

        logger.info(
            "전처리 완료 — refined 평균 %.1f자, nouns 평균 %.1f자",
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
        if self._le_large is None:
            raise RuntimeError("fit_transform()을 먼저 호출하세요.")
        return {"large": self._le_large, "medium": self._le_medium, "tag": self._le_tag}

    @property
    def n_classes(self) -> dict[str, int]:
        return {k: len(v.classes_) for k, v in self.label_encoders.items()}

    # ──────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────

    def _refine_row(self, row: pd.Series) -> str:
        """
        행 단위 정제.
        1) 술 카테고리 & alcohol_brand_preserve → 브랜드 미제거 파서 사용
        2) 일반 카테고리 → 토글 반영 파서
        3) post_refine 2차 정제
        """
        name  = str(row["product_name"]) if pd.notna(row["product_name"]) else ""
        large = str(row.get("large_category", "")) if pd.notna(row.get("large_category", "")) else ""

        if self._opt.alcohol_brand_preserve and large == ALCOHOL_LARGE:
            parser = self._parser_no_brand or self._parser
        else:
            parser = self._parser

        if not parser:
            return post_refine(name)

        parsed = parser.parse(name)
        refined = parsed.refined_text or name
        return post_refine(refined.strip() if refined else name)

    def _extract_nouns(self, text: str) -> str:
        """
        형태소 명사 추출 (Mecab 우선).

        Mecab: pos 태그 {NNG(일반명사), NNP(고유명사), SL(외국어)} 채택.
               SL 을 포함해야 BRANDTOK/VOLTOK/QTYTOK 플레이스홀더가 살아남는다.
        Okt:   기존 nouns() 방식 유지.
        불용어와 1글자 한글 명사는 제거 (플레이스홀더·영문은 길이 무관 유지).
        """
        if not text:
            return ""

        tokens: list[str] = []
        try:
            if self._mecab is not None:
                tokens = [
                    surface for surface, tag in self._mecab.pos(text)
                    if tag in ("NNG", "NNP", "SL")
                ]
            elif self._okt is not None:
                tokens = self._okt.nouns(text)
            else:
                return text                          # 분석기 없음 → 원문 그대로
        except Exception as e:
            logger.debug("명사 추출 실패 '%s': %s", text, e)
            return text

        filtered = [
            t for t in tokens
            if t not in self._stopwords
            and (len(t) > 1 or not re.fullmatch(r"[가-힣]", t))  # 1글자 한글만 배제
        ]
        return " ".join(filtered) if filtered else text

    def _init_parsers(
        self, brand_dict_path: str
    ) -> Tuple[Optional["REFProductNameParser"], Optional["REFProductNameParser"]]:
        """
        브랜드 사전을 로드하여 파서 2개를 초기화한다.
        - parser          : 토글 옵션 반영 (일반 카테고리)
        - parser_no_brand : 빈 브랜드 사전 (술 카테고리 — 브랜드 보존)
        """
        path = Path(brand_dict_path)
        brand_names: list[str] = []
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                brand_names = (
                    [str(b) for b in data if b] if isinstance(data, list)
                    else list(data.keys())
                )
                logger.info("브랜드 사전 로드: %d개", len(brand_names))
            except Exception as e:
                logger.warning("브랜드 사전 로드 실패: %s", e)
        else:
            logger.warning("브랜드 사전 파일 없음: %s", brand_dict_path)

        # ParserOptions 로 전처리 토글을 파서에 그대로 전달
        popts = ParserOptions(
            remove_brand=self._opt.remove_brand,
            remove_volume=self._opt.remove_volume,
            remove_quantity=self._opt.remove_quantity,
            placeholder=self._opt.placeholder,
        )
        matcher          = create_brand_matcher(brand_names)
        matcher_no_brand = create_brand_matcher([])     # 빈 사전 → 항상 미매칭

        return (
            REFProductNameParser(brand_matcher=matcher, options=popts),
            REFProductNameParser(brand_matcher=matcher_no_brand, options=popts),
        )
