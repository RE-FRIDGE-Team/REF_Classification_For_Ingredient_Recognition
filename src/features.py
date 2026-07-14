"""
피처 빌더 — char/word n-gram FeatureUnion + head-noun 가중 + 술 스타일어 피처.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[구성 피처 블록] (각 블록은 토글 가능, hstack 으로 수평 결합)

  1. char n-gram TF-IDF (refined_text)     — 기본. 오분절에 강한 백본.
  2. char n-gram TF-IDF (nouns_text)       — 형태소 명사열 보조 뷰.
  3. word n-gram TF-IDF (nouns_text)       — ★신규. "탕수육"·"짜장"·"건면" 같은
                                              통토큰이 head-noun 신호로 들어온다.
                                              char 와 '택일'이 아니라 '합집합'.
  4. head-noun 필드 TF-IDF                 — ★신규. 마지막 명사(LAST)와
                                              마지막 2개 명사(LAST_2)만 담은 별도
                                              필드를 벡터화하고 가중치 α(기본 2.0)를
                                              스칼라 곱. FIRST=/LAST= 문자열 접두
                                              방식(어휘 폭증)을 피하는 FeatureUnion 방식.
  5. 술 스타일어 카운트                     — ★신규. lexicons.py 의 주종별 스타일어
                                              그룹 등장 횟수(그룹당 1차원). 브랜드
                                              고유명사 제품명에도 "말벡"·"배럴"이
                                              걸리므로 술 카테고리 일반화 신호.

[BM25 토글]
  vectorizer="bm25" 선택 시 char/word 블록을 Okapi BM25 가중으로 계산.
  짧은 제품명 특성상 sublinear_tf 근사와 큰 차이가 없을 것으로 기대치는
  낮게 잡되(README TODO 참조), 완결성/실험 차원에서 제공한다.

[사람 개입 최소화]
  head-noun 추출은 형태소 분석 결과(nouns_text)의 마지막 토큰을 자동 사용.
  스타일어 사전은 내장 시드 + 선택적 확장 파일 자동 병합(lexicons.py).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .lexicons import load_alcohol_lexicon

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# BM25 벡터라이저 (Okapi BM25)
# ══════════════════════════════════════════════════════════════════

class BM25Vectorizer:
    """
    CountVectorizer 기반 Okapi BM25 벡터라이저.

    score(t, d) = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d|/avgdl))
    idf(t)      = ln( (N - df + 0.5) / (df + 0.5) + 1 )   (음수 방지형)

    sklearn 벡터라이저와 동일한 fit_transform / transform 인터페이스만 제공한다.
    (파이프라인 결합은 FeatureBuilder 가 담당하므로 최소 구현)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, **count_kwargs) -> None:
        self._k1 = k1                                   # tf 포화 파라미터
        self._b = b                                     # 문서 길이 정규화 강도
        self._cv = CountVectorizer(**count_kwargs)      # 토큰화/어휘는 CountVectorizer 재사용
        self._idf: np.ndarray | None = None             # fit 시 계산되는 idf 벡터
        self._avgdl: float = 0.0                        # 평균 문서 길이

    def fit_transform(self, raw_documents) -> sparse.csr_matrix:
        X = self._cv.fit_transform(raw_documents)       # 원시 term count 행렬
        n_docs = X.shape[0]
        df = np.bincount(X.indices, minlength=X.shape[1])   # 각 term 의 문서빈도
        # 음수 방지형 BM25 idf
        self._idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        self._avgdl = float(X.sum(axis=1).mean()) or 1.0
        return self._weight(X)

    def transform(self, raw_documents) -> sparse.csr_matrix:
        if self._idf is None:
            raise RuntimeError("fit_transform()을 먼저 호출하세요.")
        return self._weight(self._cv.transform(raw_documents))

    def _weight(self, X: sparse.csr_matrix) -> sparse.csr_matrix:
        """count 행렬에 BM25 가중을 적용한다 (CSR data 배열 직접 갱신)."""
        X = X.tocsr().astype(np.float64)
        doc_len = np.asarray(X.sum(axis=1)).ravel()     # 각 문서 길이 |d|
        # 행별 길이 정규화 계수: k1 * (1 - b + b*|d|/avgdl)
        norm = self._k1 * (1.0 - self._b + self._b * doc_len / self._avgdl)
        for i in range(X.shape[0]):                     # CSR 행 단위 순회
            start, end = X.indptr[i], X.indptr[i + 1]
            tf = X.data[start:end]
            X.data[start:end] = (
                self._idf[X.indices[start:end]]         # idf(t)
                * tf * (self._k1 + 1.0)                 # 분자
                / (tf + norm[i])                        # 분모
            )
        return X


# ══════════════════════════════════════════════════════════════════
# 피처 빌더 본체
# ══════════════════════════════════════════════════════════════════

@dataclass
class FeatureConfig:
    """
    피처 블록 토글 + 벡터라이저 하이퍼파라미터.
    (전처리 토글과 달리 이 값들은 모델 내부이므로 Optuna 탐색 대상이 될 수 있다)
    """
    # ── 블록 토글 ──
    use_char:            bool = True     # char n-gram (백본, 끄는 것은 진단용)
    use_word:            bool = True     # ★word n-gram 추가 (char 와 합집합)
    use_head_noun:       bool = True     # ★head-noun 별도 필드 + 가중
    use_alcohol_lexicon: bool = True     # ★술 스타일어 카운트 피처
    vectorizer:          str  = "tfidf"  # "tfidf" | "bm25"

    # ── char n-gram 파라미터 ──
    char_ngram_range: tuple = (2, 5)
    max_features:     int   = 10000
    min_df:           int   = 2
    sublinear_tf:     bool  = True

    # ── word n-gram 파라미터 ──
    word_ngram_range:  tuple = (1, 2)
    word_max_features: int   = 8000
    word_min_df:       int   = 2

    # ── head-noun ──
    head_weight: float = 2.0             # head 필드 가중 (스칼라 곱 = FeatureUnion transformer_weights 동치)

    # ── BM25 ──
    bm25_k1: float = 1.5
    bm25_b:  float = 0.75

    # 내부 상태 (fit 후 채워짐) — dataclass 필드로 두지 않음
    _fitted: bool = field(default=False, repr=False)


def extract_head_field(nouns_text: pd.Series, refined_text: pd.Series) -> pd.Series:
    """
    head-noun 필드 생성: "마지막 명사 + 마지막 2개 명사" 를 담은 짧은 텍스트.

    예) nouns_text="냉동 김치 피자 탕수육"
        → "탕수육 피자 탕수육"   (LAST + LAST_2 를 공백 연결)

    형태소 명사열이 비어 있으면 refined_text 의 공백 분리 마지막 토큰으로 폴백.
    FIRST=/LAST= 접두 방식과 달리 어휘를 부풀리지 않고, 별도 벡터라이저 +
    가중치 곱으로 '위치 신호'를 표현한다.
    """
    def _head(nouns: str, refined: str) -> str:
        toks = str(nouns).split() if pd.notna(nouns) and str(nouns).strip() else []
        if not toks:                                   # 명사 추출 실패 시 폴백
            toks = str(refined).split() if pd.notna(refined) else []
        if not toks:
            return ""
        last = toks[-1]                                # LAST: 대분류를 결정하는 헤드 후보
        last2 = " ".join(toks[-2:])                    # LAST_2: 복합 헤드(냉동+만두 등) 보조
        return f"{last} {last2}"

    return pd.Series(
        [_head(n, r) for n, r in zip(nouns_text.values, refined_text.values)],
        index=nouns_text.index,
    )


class FeatureBuilder:
    """
    refined_text / nouns_text 두 시리즈로부터 최종 희소 피처 행렬을 생성한다.

    fit_transform(refined, nouns) → csr_matrix
    transform(refined, nouns)     → csr_matrix   (fit 어휘 재사용)
    """

    def __init__(self, cfg: FeatureConfig | None = None) -> None:
        self._cfg = cfg or FeatureConfig()
        # 각 블록의 벡터라이저 (fit 시 생성)
        self._vec_char_refined = None
        self._vec_char_nouns = None
        self._vec_word = None
        self._vec_head = None
        # 술 스타일어 사전: {그룹: (단어,...)} — 그룹당 1개 카운트 차원
        self._alcohol_lex = load_alcohol_lexicon() if self._cfg.use_alcohol_lexicon else {}
        self._lex_groups = sorted(self._alcohol_lex.keys())

    # ──────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────

    def fit_transform(self, refined: pd.Series, nouns: pd.Series) -> sparse.csr_matrix:
        blocks = []
        c = self._cfg

        # 블록 1·2: char n-gram (refined + nouns 이중 뷰) — 기존 백본 유지
        if c.use_char:
            self._vec_char_refined = self._make_vectorizer(
                analyzer="char_wb", ngram_range=c.char_ngram_range,
                max_features=c.max_features, min_df=c.min_df,
            )
            self._vec_char_nouns = self._make_vectorizer(
                analyzer="char_wb", ngram_range=c.char_ngram_range,
                max_features=c.max_features, min_df=c.min_df,
            )
            blocks.append(self._vec_char_refined.fit_transform(refined.fillna("")))
            blocks.append(self._vec_char_nouns.fit_transform(nouns.fillna("")))

        # 블록 3: word n-gram — 통토큰 신호 (char 와 합집합)
        if c.use_word:
            self._vec_word = self._make_vectorizer(
                analyzer="word", ngram_range=c.word_ngram_range,
                max_features=c.word_max_features, min_df=c.word_min_df,
            )
            blocks.append(self._vec_word.fit_transform(nouns.fillna("")))

        # 블록 4: head-noun 필드 (가중 α 스칼라 곱)
        if c.use_head_noun:
            head = extract_head_field(nouns.fillna(""), refined.fillna(""))
            self._vec_head = self._make_vectorizer(
                analyzer="word", ngram_range=(1, 2),
                max_features=4000, min_df=1,       # head 어휘는 작으므로 min_df=1
            )
            blocks.append(self._vec_head.fit_transform(head) * c.head_weight)

        # 블록 5: 술 스타일어 카운트 (그룹당 1차원 밀집 → 희소 변환)
        if c.use_alcohol_lexicon and self._lex_groups:
            blocks.append(self._lexicon_counts(refined.fillna("")))

        X = sparse.hstack(blocks).tocsr()
        logger.debug("FeatureBuilder — 최종 피처 차원: %s", X.shape)
        return X

    def transform(self, refined: pd.Series, nouns: pd.Series) -> sparse.csr_matrix:
        """fit 완료된 어휘로 새 데이터를 변환한다 (블록 순서는 fit 과 동일)."""
        blocks = []
        c = self._cfg
        if c.use_char:
            blocks.append(self._vec_char_refined.transform(refined.fillna("")))
            blocks.append(self._vec_char_nouns.transform(nouns.fillna("")))
        if c.use_word:
            blocks.append(self._vec_word.transform(nouns.fillna("")))
        if c.use_head_noun:
            head = extract_head_field(nouns.fillna(""), refined.fillna(""))
            blocks.append(self._vec_head.transform(head) * c.head_weight)
        if c.use_alcohol_lexicon and self._lex_groups:
            blocks.append(self._lexicon_counts(refined.fillna("")))
        return sparse.hstack(blocks).tocsr()

    # ──────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────

    def _make_vectorizer(self, analyzer: str, ngram_range, max_features, min_df):
        """vectorizer 토글에 따라 TF-IDF 또는 BM25 벡터라이저를 생성한다."""
        if self._cfg.vectorizer == "bm25":
            return BM25Vectorizer(
                k1=self._cfg.bm25_k1, b=self._cfg.bm25_b,
                analyzer=analyzer, ngram_range=ngram_range,
                max_features=max_features, min_df=min_df,
            )
        return TfidfVectorizer(
            analyzer=analyzer, ngram_range=ngram_range,
            max_features=max_features, min_df=min_df,
            sublinear_tf=self._cfg.sublinear_tf,
        )

    def _lexicon_counts(self, texts: pd.Series) -> sparse.csr_matrix:
        """
        술 스타일어 그룹별 등장 횟수 행렬 (n_samples × n_groups).

        단순 substring 카운트 — "소비뇽블랑"과 "소비뇽"이 이중 카운트될 수 있으나
        분류 신호로는 문제없고(모두 wine 그룹), 구현 단순성이 우선.
        값은 log1p 로 눌러 극단값 영향을 완화한다.
        """
        mat = np.zeros((len(texts), len(self._lex_groups)), dtype=np.float64)
        for col, group in enumerate(self._lex_groups):
            words = self._alcohol_lex[group]
            for row, text in enumerate(texts.values):
                t = str(text)
                mat[row, col] = sum(t.count(w) for w in words)
        return sparse.csr_matrix(np.log1p(mat))
