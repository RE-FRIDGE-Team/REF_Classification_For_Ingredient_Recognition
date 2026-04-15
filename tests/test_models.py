"""
3개 모델 smoke test (소량 데이터, 2 trials).

실행:
    pytest tests/test_models.py -v
    pytest tests/test_models.py -v -k "tfidf"   # 특정 모델만
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 프로젝트 루트를 경로에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_df() -> pd.DataFrame:
    """10샘플 더미 DataFrame (모델 smoke test용)."""
    products = [
        "오뚜기 진라면 순한맛 120g 5개입",
        "풀무원 국산콩 순두부 300g",
        "CJ 비비고 왕교자 1.05kg",
        "농심 새우깡 90g",
        "롯데 꼬깔콘 굴비맛 67g",
        "하이트진로 참이슬 후레쉬 360ml",
        "서울우유 흰우유 1L",
        "동원참치 살코기 135g",
        "청정원 굴소스 500g",
        "CJ 햇반 210g 3개",
    ]
    large_cats = ["간편식", "채소/곡물류", "간편식", "간식류", "간식류", "술", "유제품", "수산물", "조미료", "상온식품"]
    medium_cats = ["라면", "콩/두부", "만두", "스낵", "스낵", "소주", "흰우유", "참치캔", "소스", "즉석밥"]
    tags = ["INSTANT", "SIMPLE_PROCESSED", "RTC_MEAL", "SNACK", "SNACK", "ALCOHOL", "DAIRY", "CANNED", "SAUCE_SEASONING", "INSTANT"]
    brands = ["오뚜기", "풀무원", "CJ", "농심", "롯데", "하이트진로", "서울우유", "동원", "청정원", "CJ"]

    return pd.DataFrame({
        "product_name":    products,
        "category_large":  large_cats,
        "category_medium": medium_cats,
        "category_tag":    tags,
        "brand_name":      brands,
    })


@pytest.fixture(scope="module")
def preprocessed(tiny_df: pd.DataFrame):
    """전처리 완료 DataFrame + LabelEncoders."""
    from src.preprocess import REFPreprocessor
    prep = REFPreprocessor(use_parser=False)   # parser 없이 빠른 smoke test
    df = prep.fit_transform(tiny_df)
    return df, prep


@pytest.fixture(scope="module")
def folds(preprocessed):
    """2-Fold (데이터가 작아 n=2 사용)."""
    from src.data_utils import make_folds
    df, _ = preprocessed
    return make_folds(df, n_splits=2, group_col="brand_name", seed=42)


# ──────────────────────────────────────────────────────────────────
# 전처리 단위 테스트
# ──────────────────────────────────────────────────────────────────

class TestPreprocessor:
    def test_columns_added(self, preprocessed):
        df, prep = preprocessed
        assert "refined_text" in df.columns
        assert "nouns_text" in df.columns
        assert "label_large" in df.columns
        assert "label_medium" in df.columns
        assert "label_tag" in df.columns

    def test_label_range(self, preprocessed):
        df, prep = preprocessed
        n = prep.n_classes
        assert df["label_large"].between(0, n["large"] - 1).all()
        assert df["label_medium"].between(0, n["medium"] - 1).all()
        assert df["label_tag"].between(0, n["tag"] - 1).all()

    def test_no_nulls(self, preprocessed):
        df, _ = preprocessed
        assert df["refined_text"].notna().all()
        assert df["nouns_text"].notna().all()

    def test_label_encoders_accessible(self, preprocessed):
        _, prep = preprocessed
        les = prep.label_encoders
        assert set(les.keys()) == {"large", "medium", "tag"}


# ──────────────────────────────────────────────────────────────────
# GroupKFold 단위 테스트
# ──────────────────────────────────────────────────────────────────

class TestDataUtils:
    def test_folds_count(self, folds):
        assert len(folds) == 2

    def test_no_brand_overlap(self, preprocessed, folds):
        df, _ = preprocessed
        for tr_idx, va_idx in folds:
            tr_brands = set(df.iloc[tr_idx]["brand_name"])
            va_brands = set(df.iloc[va_idx]["brand_name"])
            assert tr_brands.isdisjoint(va_brands), "브랜드 누수 발생"

    def test_index_coverage(self, preprocessed, folds):
        df, _ = preprocessed
        all_indices: set = set()
        for tr_idx, va_idx in folds:
            all_indices.update(va_idx)
        assert all_indices == set(range(len(df)))


# ──────────────────────────────────────────────────────────────────
# TF-IDF + LightGBM smoke test
# ──────────────────────────────────────────────────────────────────

class TestTfidfLgbm:
    @pytest.fixture(autouse=True)
    def _check_lgb(self):
        pytest.importorskip("lightgbm")

    def test_fit_predict_shape(self, preprocessed):
        from src.models.tfidf_lgbm import TfidfLgbmClassifier
        df, prep = preprocessed
        n = prep.n_classes
        model = TfidfLgbmClassifier(n_estimators=10, num_leaves=5)
        model.fit(
            df["refined_text"], df["nouns_text"],
            df["label_large"].values,
            df["label_medium"].values,
            df["label_tag"].values,
        )
        pl, pm, pt = model.predict(df["refined_text"], df["nouns_text"])
        assert pl.shape == (len(df),)
        assert pm.shape == (len(df),)
        assert pt.shape == (len(df),)

    def test_predict_proba_shape(self, preprocessed):
        from src.models.tfidf_lgbm import TfidfLgbmClassifier
        df, prep = preprocessed
        n = prep.n_classes
        model = TfidfLgbmClassifier(n_estimators=10, num_leaves=5)
        model.fit(
            df["refined_text"], df["nouns_text"],
            df["label_large"].values, df["label_medium"].values, df["label_tag"].values,
        )
        prob_l, _, _ = model.predict_proba(df["refined_text"], df["nouns_text"])
        assert prob_l.shape == (len(df), n["large"])

    def test_get_params_returns_dict(self, preprocessed):
        from src.models.tfidf_lgbm import TfidfLgbmClassifier
        df, _ = preprocessed
        model = TfidfLgbmClassifier()
        assert isinstance(model.get_params(), dict)


# ──────────────────────────────────────────────────────────────────
# FastText + KoNLPy smoke test
# ──────────────────────────────────────────────────────────────────

class TestFastText:
    @pytest.fixture(autouse=True)
    def _check_gensim(self):
        pytest.importorskip("gensim")

    def test_fit_predict_shape(self, preprocessed):
        from src.models.fasttext_konlpy import FastTextKonlpyClassifier
        df, prep = preprocessed
        model = FastTextKonlpyClassifier(vector_size=16, epochs=2, min_count=1)
        model.fit(
            df["refined_text"], df["nouns_text"],
            df["label_large"].values, df["label_medium"].values, df["label_tag"].values,
        )
        pl, pm, pt = model.predict(df["refined_text"], df["nouns_text"])
        assert pl.shape == (len(df),)

    def test_classifier_type_logreg(self, preprocessed):
        from src.models.fasttext_konlpy import FastTextKonlpyClassifier
        from sklearn.linear_model import LogisticRegression
        df, _ = preprocessed
        model = FastTextKonlpyClassifier(vector_size=16, epochs=2, classifier_type="logreg")
        model.fit(
            df["refined_text"], df["nouns_text"],
            df["label_large"].values, df["label_medium"].values, df["label_tag"].values,
        )
        assert isinstance(model._clf_large, LogisticRegression)


# ──────────────────────────────────────────────────────────────────
# Optuna HPO smoke test (2 trials)
# ──────────────────────────────────────────────────────────────────

class TestOptunaRunner:
    @pytest.fixture(autouse=True)
    def _check_deps(self):
        pytest.importorskip("lightgbm")
        pytest.importorskip("optuna")

    def test_tfidf_hpo_2trials(self, preprocessed, folds):
        from src.models.tfidf_lgbm import TfidfLgbmClassifier
        from src.tuning.optuna_runner import OptunaRunner

        df, _ = preprocessed

        search_space = {
            "tfidf": {"max_features": [500, 1000], "sublinear_tf": [True, False]},
            "lgbm":  {"n_estimators": [10, 30], "num_leaves": [5, 10]},
        }
        runner = OptunaRunner(
            model_name="tfidf_lgbm_test",
            model_cls=TfidfLgbmClassifier,
            search_space=search_space,
            folds=folds,
            df=df,
            n_trials=2,
            storage=None,
            timeout=None,
        )
        result = runner.run()
        assert 0.0 <= result.best_score <= 1.0
        assert isinstance(result.best_params, dict)


# ──────────────────────────────────────────────────────────────────
# CV evaluate smoke test
# ──────────────────────────────────────────────────────────────────

class TestCVEvaluate:
    @pytest.fixture(autouse=True)
    def _check_lgb(self):
        pytest.importorskip("lightgbm")

    def test_cv_result_fields(self, preprocessed, folds):
        from src.models.tfidf_lgbm import TfidfLgbmClassifier
        from src.evaluate import cv_evaluate

        df, _ = preprocessed
        best_params = {"n_estimators": 10, "num_leaves": 5}
        cv_result, details = cv_evaluate(
            model_cls=TfidfLgbmClassifier,
            best_params=best_params,
            extra_kwargs={},
            df=df,
            folds=folds,
        )
        assert 0.0 <= cv_result.large_f1 <= 1.0
        assert len(details) == len(folds)
        assert cv_result.train_time > 0


# ──────────────────────────────────────────────────────────────────
# data_utils smoke test
# ──────────────────────────────────────────────────────────────────

class TestLoadData:
    def test_load_csv(self, tmp_path):
        from src.data_utils import load_data

        csv = tmp_path / "test.csv"
        csv.write_text(
            "product_name,category_large,category_medium,item_type,brand_name\n"
            "테스트상품,음료,주스,BEVERAGE,브랜드A\n",
            encoding="utf-8",
        )
        col_map = {
            "product_name":    "product_name",
            "large_category":  "category_large",
            "medium_category": "category_medium",
            "category_tag":    "item_type",
            "brand":           "brand_name",
        }
        df = load_data(str(csv), col_map)
        assert len(df) == 1
        assert "large_category" in df.columns

    def test_missing_column_raises(self, tmp_path):
        from src.data_utils import load_data

        csv = tmp_path / "bad.csv"
        csv.write_text("wrong_col\nval\n", encoding="utf-8")
        col_map = {"product_name": "product_name", "large_category": "category_large",
                   "medium_category": "category_medium", "category_tag": "item_type",
                   "brand": "brand_name"}
        with pytest.raises(ValueError, match="없는 컬럼"):
            load_data(str(csv), col_map)
