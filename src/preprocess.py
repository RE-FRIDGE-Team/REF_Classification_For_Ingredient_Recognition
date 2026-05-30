"""
공용 전처리 모듈.

파이프라인:
    1. REFProductNameParser로 refined_text 생성 (기존 parser 재사용)
       - 일반 카테고리: 브랜드 사전 적용 (브랜드명 제거)
       - 술 카테고리:   브랜드 사전 미적용 (브랜드명 보존, 노이즈만 제거)
    2. [추가] post_refine() — 파서 이후 잔재 노이즈 2차 정제
       - 단가 괄호 잔재: (100g당 ), (당 ), (10g당 ) 등
       - 홍보 문구 잔재: 최대 적립, 최저, 최대~
       - 부속 설명 괄호: (소스 포함), (보자기 동봉), (증정) 등
       - 빈 의미 괄호:   ( + ), ( X 4), (/), ( X) 등
       - 특수문자 잔재:  끝의 * X, *24캔, 1/ 등
       - 할인율 잔재:    단독으로 남은 NN%
    3. Okt 명사 추출 + 불용어 제거 → nouns_text
    4. 타깃 컬럼 LabelEncoder 인코딩 (대/중분류, 카테고리태그)

변경 이력:
변경 이력:
    - alcohol_brand_preserve 로직 변경:
      기존: 파서 적용 후 refined_text가 짧으면 brand_name으로 사후 복구
      변경: 술 카테고리는 처음부터 빈 브랜드 사전을 사용하는 별도 파서를 적용
            → 브랜드 제거 자체를 하지 않아 "오프리16" → "오프리16" 유지
    - post_refine() 추가:
      파서의 정규식 패턴이 부분 매칭 후 남기는 잔재와
      크롤링 원본에서 유입된 홍보/부속 문구를 2차로 제거
"""


import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

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


# ──────────────────────────────────────────────────────────────────
# 2차 정제 패턴 (POST_REFINE_PATTERNS)
#
# 파서(REFProductNameParser) 적용 이후에도 남아있는 잔재 노이즈를
# 순서대로 제거한다. 패턴은 보수적으로 설계하여 의미있는 괄호
# (특), (글루텐 프리) 등은 보존한다.
# ──────────────────────────────────────────────────────────────────
_POST_REFINE_PATTERNS: List[re.Pattern] = [
    # ── 단가 괄호 잔재 ───────────────────────────────────────────
    # (100g당 ), (10g당 ), (당 ), (1개당 ), (ml당 ) 등
    # 숫자+단위+당 또는 단독 '당'이 괄호 안에 있고 뒤에 값이 없는 경우
    re.compile(r'\(\s*(?:\d+(?:\.\d+)?(?:g|kg|ml|l|개|원))?\s*당\s*\)'),

    # ── 홍보 문구 잔재 ───────────────────────────────────────────
    # 최대 N원 적립 (숫자 포함)
    re.compile(r'최대\s*\d+[원\w]*\s*적립'),
    # 최대 적립 (숫자 없는 잔재)
    re.compile(r'최대\s*적립'),
    # 최저 N원 잔재
    re.compile(r'최저\s*\d*[원\w]*'),
    # 최대~ 형태 (예: 최대~ 최대~)
    re.compile(r'최대~'),

    # ── 부속 설명 괄호 ───────────────────────────────────────────
    # (소스 포함), (보자기 동봉), (티백 포함), (갈치속젓 증정) 등
    # '포함', '동봉', '증정'으로 끝나는 괄호 전체 제거
    re.compile(r'\([^)]*(?:포함|동봉|증정)\)'),

    # ── 할인율 잔재 ─────────────────────────────────────────────
    # 단독으로 남은 NN% (뒤에 괄호가 오지 않는 경우)
    # 예: "넛츠팜 구운땅콩 25%" → "넛츠팜 구운땅콩"
    re.compile(r'(?<!\w)\d+%(?!\s*\()'),

    # ── 빈 의미 괄호 ────────────────────────────────────────────
    # ( + ), ( X 4), ( X), (/), ( * ) 등
    # 괄호 안에 한글/영문 의미 단어 없이 기호+숫자+공백만 있는 경우
    re.compile(r'\(\s*[+\-×xX*/÷|,\s\d]*\s*\)'),

    # ── 끝 단독 특수문자 잔재 ───────────────────────────────────
    # 끝에 공백 후 단독 * X 등 (예: "칠레산 블루베리 *", "동결건조 매생이 X")
    re.compile(r'\s+[*×xX]\s*$'),
    # *숫자단위 형태 (예: *24캔, *4개) — 공백 없이 붙은 경우도 처리
    re.compile(r'\*\d+[가-힣a-zA-Z]*$'),

    # ── 분수/슬래시 잔재 ─────────────────────────────────────────
    # 끝에 남은 숫자/ 형태 (예: "친환경 적양배추 1/")
    # 보존: "(중과/)" 같은 괄호 안 슬래시는 영향 없음
    re.compile(r'(?<!\))\s+\d+/\s*$'),
    re.compile(r'(?<![가-힣a-zA-Z(])\d+/\s*$'),
]


def post_refine(text: str) -> str:
    """
    파서(REFProductNameParser) 적용 이후 남아있는 잔재 노이즈를 제거한다.

    제거 대상:
        - 단가 괄호 잔재: (100g당 ), (당 ), (10g당 ) 등
        - 홍보 문구:      최대 적립, 최저, 최대~, NN%
        - 부속 설명 괄호: (소스 포함), (보자기 동봉), (증정) 등
        - 빈 의미 괄호:   ( + ), ( X 4), (/), ( X) 등
        - 특수문자 잔재:  끝의 * X, *24캔, 숫자/ 등
        - 비인쇄 공백:    \\xa0 등

    보존 대상:
        - (특), (글루텐 프리) 등 의미 있는 한글 포함 괄호
        - 1.3kg, 3kg, CS 등 제품 규격/시리즈 정보

    Args:
        text: REFProductNameParser.parse() 이후의 refined_text

    Returns:
        2차 정제된 문자열
    """
    if not text:
        return text

    result = text.replace('\xa0', ' ')  # non-breaking space 정규화

    for pat in _POST_REFINE_PATTERNS:
        result = pat.sub('', result)

    # 빈 괄호 잔재 정리 (패턴 적용 후 생성될 수 있음)
    result = re.sub(r'\(\s*\)', '', result)
    result = re.sub(r'\[\s*\]', '', result)

    # 연속 공백 → 단일 공백, 앞뒤 정리
    result = re.sub(r'\s+', ' ', result).strip()

    # 끝에 남은 단독 구두점/공백 제거
    result = re.sub(r'[\s,.\-]+$', '', result).strip()

    return result


class REFPreprocessor:
    """
    RE:FRIDGE 제품명 전처리기.

    fit_transform()으로 DataFrame을 변환하여:
        - refined_text : 파서 정제 + post_refine() 2차 정제 제품명
        - nouns_text   : Okt 명사 추출 + 불용어 제거 (공백 구분)
        - label_large  : 대분류 인코딩 (int)
        - label_medium : 중분류 인코딩 (int)
        - label_tag    : 카테고리태그 인코딩 (int)
    컬럼을 추가한다.

    label_encoders 프로퍼티로 LabelEncoder 3개 접근 가능.

    파서 전략:
        - 일반 카테고리: _parser (브랜드 사전 적용 → 브랜드명 제거)
        - 술 카테고리  : _parser_no_brand (빈 브랜드 사전 → 브랜드명 보존, 노이즈만 제거)
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
        # _parser         : 브랜드 사전 적용 (일반 카테고리용)
        # _parser_no_brand: 빈 브랜드 사전   (술 카테고리용 — 브랜드명 보존)
        self._parser: Optional[REFProductNameParser] = None
        self._parser_no_brand: Optional[REFProductNameParser] = None

        if self._use_parser:
            self._parser, self._parser_no_brand = self._init_parser(brand_dict_path)

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
        """
        행 단위 제품명 정제.

        1단계 — 파서 적용:
            술 카테고리: _parser_no_brand → 브랜드명 보존, 노이즈(대괄호 등)만 제거
            일반 카테고리: _parser → 브랜드명 제거 + 용량/수량/노이즈 제거

        2단계 — post_refine():
            파서 이후 남아있는 잔재 노이즈 제거
            (단가 괄호, 최대 적립, 부속 설명 괄호, 특수문자 등)
        """
        name  = str(row["product_name"]) if pd.notna(row["product_name"]) else ""
        large = str(row.get("large_category", "")) if pd.notna(row.get("large_category", "")) else ""

        # 술 카테고리: 브랜드 사전 미적용 파서 사용 (브랜드명 보존)
        if self._alcohol_brand_preserve and large == ALCOHOL_LARGE:
            parser = self._parser_no_brand or self._parser
        else:
            parser = self._parser

        if not parser:
            return post_refine(name)

        parsed = parser.parse(name)
        refined = parsed.refined_text or name

        # 2차 정제: 파서 이후 잔재 노이즈 제거
        return post_refine(refined.strip() if refined else name)

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
    def _init_parser(brand_dict_path: str) -> Tuple[
        Optional["REFProductNameParser"],
        Optional["REFProductNameParser"],
    ]:
        """
        브랜드 사전을 로드하여 파서 두 개를 초기화한다.

        Returns:
            (parser, parser_no_brand)
            - parser          : 브랜드 사전 적용 (일반 카테고리용)
            - parser_no_brand : 빈 브랜드 사전   (술 카테고리용)
        """
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
            logger.warning(
                "브랜드 사전 파일 없음: %s. 파서는 브랜드 없이 실행됩니다.",
                brand_dict_path,
            )

        # 일반용: 브랜드 사전 적용
        matcher = create_brand_matcher(brand_names)

        # 술용: 빈 사전 → find_match()가 항상 None 반환 → 브랜드 제거 없음
        matcher_no_brand = create_brand_matcher([])

        return (
            REFProductNameParser(brand_matcher=matcher),
            REFProductNameParser(brand_matcher=matcher_no_brand),
        )