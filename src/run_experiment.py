"""
RE:FRIDGE Phase 1 — 실험 CLI 진입점 (2026-07 개편).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[개편 요약]
  - MODEL_REGISTRY: fasttext 삭제, tfidf → text(TextPipelineClassifier) 교체.
    koelectra 는 술 OOV 구제 후순위 옵션으로 유지.
  - 모든 전처리/피처 정책이 CLI 토글로 노출됨 (2순위 원칙).
    CLI 미지정 시 experiment.yaml 값 → 그것도 없으면 코드 기본값.
  - Mecab 사용자사전: 입력 CSV의 PGIN 컬럼 어휘를 자동 수집해 주입 (사람 개입 0).
  - HeadRuleEngine(모호 헤드어 룰)을 LabelEncoder 로 생성해 모델에 주입.
  - dedup: 전처리 후 · fold 생성 전 근접중복 원본 제거.

[사용 예 — A/B 실험 축]
  # ① 버그픽스+dedup 재베이스라인 (판정은 항상 StratifiedGroupKFold)
  python src/run_experiment.py --input data.csv --models text --n-trials 50

  # ② 형태소 분석기 비교 (Mecab vs Okt)
  python src/run_experiment.py --input data.csv --models text --morpheme okt --study-version v2_okt

  # ③ 피처 절제(ablation) — word n-gram 끄기
  python src/run_experiment.py --input data.csv --models text --no-word-ngram --study-version v2_noword

  # ④ 전처리 변형 — 브랜드 유지 / 플레이스홀더
  python src/run_experiment.py --input data.csv --models text --keep-brand --study-version v2_kb
  python src/run_experiment.py --input data.csv --models text --placeholder --study-version v2_ph

  # ⑤ BM25 벡터라이저
  python src/run_experiment.py --input data.csv --models text --vectorizer bm25 --study-version v2_bm25

  # (후순위) 술 OOV 구제용 KoELECTRA 추가
  python src/run_experiment.py --input data.csv --models text,koelectra --parallel
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import optuna
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_utils import (
    CVStrategy,
    data_summary,
    dedup_near_duplicates,
    load_data,
    make_folds_orig_only,
)
from src.preprocess import PreprocessOptions, REFPreprocessor
from src.models import KoElectraMultiTaskClassifier, TextPipelineClassifier
from src.rules import HeadRuleEngine
from src.tuning.optuna_runner import OptunaRunner
from src.evaluate import (
    cv_evaluate,
    error_examples,
    per_class_f1_report,
    save_confusion_matrices,
)
from src.compare import (
    print_comparison_table,
    save_comparison_chart,
    save_comparison_csv,
    save_html_report,
)

# ──────────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    # key       (study/디렉토리명,  클래스)
    "text":      ("text_pipeline", TextPipelineClassifier),
    "koelectra": ("koelectra",     KoElectraMultiTaskClassifier),
}
# 하위 호환: 구 명령어의 --models tfidf 를 text 로 매핑
_LEGACY_ALIAS = {"tfidf": "text"}

CV_STRATEGIES: list[CVStrategy] = [
    "GroupKFold",
    "StratifiedGroupKFold",
    "StratifiedKFold",
    "KFold",
]


# ──────────────────────────────────────────────────────────────────
# Storage 관리 헬퍼 (기존 유지)
# ──────────────────────────────────────────────────────────────────

def _delete_study_if_exists(study_name: str, storage: str, logger: logging.Logger) -> None:
    try:
        optuna.delete_study(study_name=study_name, storage=storage)
        logger.info("study 삭제 완료: %s", study_name)
    except KeyError:
        logger.info("삭제할 study 없음 (신규): %s", study_name)


def _build_study_name(model_name: str, study_version: str, cv_strategy: str) -> str:
    """CV 전략이 다르면 완전히 다른 study 로 격리한다."""
    return f"ref_{model_name}_{cv_strategy}_{study_version}"


# ──────────────────────────────────────────────────────────────────
# 단일 모델 실험 함수 (ProcessPoolExecutor worker) — 기존 구조 유지
# ──────────────────────────────────────────────────────────────────

def run_single_model(
    model_key: str,
    model_name: str,
    model_cls,
    search_space: dict,
    n_trials: int,
    df_path: str,
    folds_path: str,
    label_encoders_path: str,
    extra_kwargs: dict,
    optuna_cfg: dict,
    output_dir: str,
    study_version: str,
    reset_storage: bool,
    cv_strategy: str,
) -> dict:
    import pickle
    from src.tuning.optuna_runner import OptunaRunner
    from src.evaluate import cv_evaluate, save_confusion_matrices

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    logger = logging.getLogger(model_key)

    with open(df_path, "rb") as f:
        df = pickle.load(f)
    with open(folds_path, "rb") as f:
        folds = pickle.load(f)
    with open(label_encoders_path, "rb") as f:
        label_encoders = pickle.load(f)

    storage    = optuna_cfg.get("storage")
    study_name = _build_study_name(model_name, study_version, cv_strategy)

    if reset_storage and storage:
        _delete_study_if_exists(study_name, storage, logger)

    logger.info("[%s] HPO 시작: %d trials (study=%s)", model_key, n_trials, study_name)
    runner = OptunaRunner(
        model_name=model_name,
        model_cls=model_cls,
        search_space=search_space,
        folds=folds,
        df=df,
        n_trials=n_trials,
        sampler=optuna_cfg.get("sampler", "TPE"),
        pruner=optuna_cfg.get("pruner", "MedianPruner"),
        n_startup_trials=optuna_cfg.get("n_startup_trials", 10),
        n_warmup_steps=optuna_cfg.get("n_warmup_steps", 2),
        timeout=optuna_cfg.get("timeout_per_model", 3600),
        storage=storage,
        extra_kwargs=extra_kwargs,
        study_name=study_name,
    )
    study_result = runner.run()

    logger.info("[%s] 최적 파라미터로 CV 평가 시작", model_key)
    cv_result, fold_details = cv_evaluate(
        model_cls=model_cls,
        best_params=study_result.best_params,
        extra_kwargs=extra_kwargs,
        df=df,
        folds=folds,
    )

    # CV 전략별 결과 디렉토리 분리 (덮어쓰기 방지)
    out = Path(output_dir) / cv_strategy / model_name
    out.mkdir(parents=True, exist_ok=True)

    save_confusion_matrices(
        details=fold_details,
        label_encoders=label_encoders,
        out_dir=out,
        model_name=model_name,
        normalize=True,
    )

    with open(out / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(study_result.best_params, f, indent=2, ensure_ascii=False)

    cv_scores = {
        "cv_strategy":   cv_strategy,
        "large_f1":      cv_result.large_f1,
        "large_f1_std":  cv_result.large_f1_std,
        "medium_f1":     cv_result.medium_f1,
        "medium_f1_std": cv_result.medium_f1_std,
        "tag_f1":        cv_result.tag_f1,
        "tag_f1_std":    cv_result.tag_f1_std,
        "large_acc":     cv_result.large_acc,
        "train_time":    cv_result.train_time,
        "infer_time_ms": cv_result.infer_time_ms,
    }
    with open(out / "cv_scores.json", "w", encoding="utf-8") as f:
        json.dump(cv_scores, f, indent=2)

    if study_result.all_trials_df is not None:
        study_result.all_trials_df.to_csv(out / "optuna_trials.csv", index=False)

    return {
        "model_key":    model_key,
        "model_name":   model_name,
        "cv_result":    cv_result,
        "study_result": study_result,
        "fold_details": fold_details,
    }


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RE:FRIDGE Phase 1 — 텍스트 분류 실험 (전처리/피처 토글판)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",   required=True, help="CSV 또는 XLSX 입력 파일 경로")
    parser.add_argument("--config",  default="configs/experiment.yaml", help="실험 설정 파일")
    parser.add_argument("--output",  default="results/", help="결과 저장 디렉토리")
    parser.add_argument(
        "--models", default="text",
        help="실행할 모델 (all | text | koelectra | 쉼표 분리). 구버전 'tfidf'도 허용.",
    )
    parser.add_argument("--n-trials", type=int, default=None,
                        help="모델당 Optuna trial 수 (설정 파일 값 override)")
    parser.add_argument("--parallel", action="store_true",
                        help="ProcessPoolExecutor 로 모델 동시 실행")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # ── CV 전략 ──
    parser.add_argument(
        "--cv-strategy", type=str, default=None, choices=CV_STRATEGIES,
        help=(
            "원본 fold 분할 전략 (증강은 항상 train 에만 합류). 미지정 시 yaml 값.\n"
            "★판정(성능 주장)은 반드시 StratifiedGroupKFold(group=brand) 로만 한다.\n"
            "  StratifiedKFold 는 브랜드 암기 누수로 낙관적 — 진단(갭 측정)용."
        ),
    )

    # ── ★전처리 토글 (미지정=None → yaml 값 사용) ──
    g_pre = parser.add_argument_group("전처리 토글 (A/B 실험 축)")
    g_pre.add_argument("--keep-brand", action="store_true", default=None,
                       help="브랜드를 제거하지 않고 원문 유지 (기본: 제거)")
    g_pre.add_argument("--keep-volume", action="store_true", default=None,
                       help="용량(300g 등)을 제거하지 않음 (기본: 제거)")
    g_pre.add_argument("--keep-quantity", action="store_true", default=None,
                       help="수량(4개입 등)을 제거하지 않음 (기본: 제거)")
    g_pre.add_argument("--placeholder", action="store_true", default=None,
                       help="브랜드/용량/수량을 제거 대신 BRANDTOK/VOLTOK/QTYTOK 로 치환")
    g_pre.add_argument("--morpheme", choices=["mecab", "okt", "none"], default=None,
                       help="형태소 분석기 (기본: mecab + 자동 사용자사전)")
    g_pre.add_argument("--dedup", action=argparse.BooleanOptionalAction, default=None,
                       help="refined_text 근접중복 원본 행 제거 (--no-dedup 로 비활성)")

    # ── ★피처 토글 (A/B 실험 축) ──
    g_feat = parser.add_argument_group("피처 토글")
    g_feat.add_argument("--word-ngram", dest="use_word",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="word n-gram 블록 (--no-word-ngram 으로 절제 실험)")
    g_feat.add_argument("--head-noun", dest="use_head_noun",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="head-noun 가중 필드 블록")
    g_feat.add_argument("--alcohol-lexicon", dest="use_alcohol_lexicon",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="술 스타일어 카운트 피처 블록")
    g_feat.add_argument("--rules", dest="use_rules",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="모호 헤드어 룰 오버라이드 (건면+짜장→라면 등)")
    g_feat.add_argument("--vectorizer", choices=["tfidf", "bm25"], default=None,
                        help="char/word 블록 가중 방식 (기본: tfidf)")

    # ── Storage 관리 ──
    parser.add_argument("--study-version", type=str, default="v1",
                        help="Optuna study name suffix. 실험 축이 바뀌면 반드시 올려서 격리.")
    parser.add_argument("--reset-storage", action="store_true",
                        help="실행 전 해당 study 의 기존 Optuna 결과를 삭제.")

    return parser.parse_args()


def _resolve(cli_val, yaml_val, default):
    """우선순위: CLI 인수 > experiment.yaml > 코드 기본값."""
    if cli_val is not None:
        return cli_val
    if yaml_val is not None:
        return yaml_val
    return default


def _collect_domain_words(input_path: str, pgin_col: str | None) -> list[str]:
    """
    Mecab 사용자사전용 도메인 어휘를 입력 파일의 PGIN 컬럼에서 자동 수집한다.
    (load_data 는 표준 컬럼만 남기므로 원본 파일을 직접 읽는다)
    컬럼이 없으면 빈 리스트 — 보호명사+술 스타일어만으로 사전을 만든다.
    """
    logger = logging.getLogger("run_experiment")
    if not pgin_col:
        return []
    try:
        suffix = Path(input_path).suffix.lower()
        raw = (
            pd.read_excel(input_path, dtype=str)
            if suffix in (".xlsx", ".xls")
            else pd.read_csv(input_path, dtype=str, encoding="utf-8-sig")
        )
        if pgin_col not in raw.columns:
            logger.info("PGIN 컬럼(%s) 없음 — 도메인 어휘 수집 생략", pgin_col)
            return []
        words = raw[pgin_col].dropna().unique().tolist()
        logger.info("Mecab 사용자사전 도메인 어휘 수집: %d개 (컬럼=%s)", len(words), pgin_col)
        return words
    except Exception as e:
        logger.warning("도메인 어휘 수집 실패(무시): %s", e)
        return []


def main() -> None:
    args = parse_args()

    # ── 로깅 설정 ──
    Path(args.output).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                Path(args.output) / "experiment.log", mode="a", encoding="utf-8"
            ),
        ],
    )
    logger = logging.getLogger("run_experiment")

    # ── 설정 로드 ──
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pre_cfg  = cfg.get("preprocessing", {})
    feat_cfg = cfg.get("features", {})

    # 판정 전략: CLI > yaml > StratifiedGroupKFold(★새 기본값)
    cv_strategy: CVStrategy = (
        args.cv_strategy or cfg.get("cv", {}).get("strategy", "StratifiedGroupKFold")
    )

    # ── 전처리 옵션 해석 (CLI > yaml > 기본값) ──
    pre_opt = PreprocessOptions(
        # --keep-* 플래그는 '제거 안 함'을 의미 → remove_* 반전
        remove_brand=not _resolve(args.keep_brand, not pre_cfg.get("remove_brand", True), False),
        remove_volume=not _resolve(args.keep_volume, not pre_cfg.get("remove_volume", True), False),
        remove_quantity=not _resolve(args.keep_quantity, not pre_cfg.get("remove_quantity", True), False),
        placeholder=_resolve(args.placeholder, pre_cfg.get("placeholder"), False),
        morpheme_analyzer=_resolve(args.morpheme, pre_cfg.get("morpheme_analyzer"), "mecab"),
        alcohol_brand_preserve=pre_cfg.get("alcohol_brand_preserve", True),
    )
    do_dedup = _resolve(args.dedup, pre_cfg.get("dedup"), True)

    # ── 피처 토글 해석 ──
    use_word    = _resolve(args.use_word, feat_cfg.get("use_word"), True)
    use_head    = _resolve(args.use_head_noun, feat_cfg.get("use_head_noun"), True)
    use_alcohol = _resolve(args.use_alcohol_lexicon, feat_cfg.get("use_alcohol_lexicon"), True)
    use_rules   = _resolve(args.use_rules, feat_cfg.get("use_rules"), True)
    vectorizer  = _resolve(args.vectorizer, feat_cfg.get("vectorizer"), "tfidf")

    logger.info(
        "실험 설정 — cv=%s | 전처리=%s dedup=%s | 피처: word=%s head=%s alcohol=%s rules=%s vec=%s",
        cv_strategy, pre_opt, do_dedup, use_word, use_head, use_alcohol, use_rules, vectorizer,
    )

    # ── 데이터 로드 & 전처리 ──
    logger.info("데이터 로드: %s", args.input)
    df_raw = load_data(
        input_path=args.input,
        col_map=cfg["data"]["input_columns"],
        exclude_large=cfg["data"].get("exclude_large"),
        exclude_tag=cfg["data"].get("exclude_tag"),
    )

    # Mecab 사용자사전용 도메인 어휘(PGIN) 자동 수집 — 사람 개입 0
    domain_words = (
        _collect_domain_words(args.input, cfg["data"].get("pgin_column"))
        if pre_opt.morpheme_analyzer == "mecab" else []
    )

    prep = REFPreprocessor(
        brand_dict_path=cfg["data"].get(
            "brand_dict_path",
            "product_data_collection/not_grocery_and_brand_list/grocery_brand_name.json",
        ),
        stopwords=pre_cfg.get("stopwords", []),
        options=pre_opt,
        use_parser=pre_cfg.get("use_parser", True),
        domain_words=domain_words,
    )
    df = prep.fit_transform(df_raw)

    # ── ★근접중복 제거 — 전처리 후 · fold 생성 전 ──
    if do_dedup:
        df = dedup_near_duplicates(df, text_col="refined_text")

    # 원본(is_augmented==0)만으로 fold 생성. 증강은 cv_evaluate 에서 train 에만 합류.
    folds = make_folds_orig_only(
        df,
        n_splits=cfg["cv"]["n_splits"],
        target_col=cfg["cv"].get("target_col", "large_category"),
        strategy=cv_strategy,
        seed=cfg["cv"]["seed"],
    )
    n_classes = prep.n_classes
    summary   = data_summary(df)
    summary["cv_strategy"] = cv_strategy

    # ── ★룰 엔진 생성 (LabelEncoder 필요 → 전처리 후) ──
    rule_engine = HeadRuleEngine(
        label_encoders=prep.label_encoders,
        enabled=use_rules,
    ) if use_rules else None

    # ── IPC용 임시 pickle ──
    import pickle, tempfile, shutil
    tmp_dir    = Path(tempfile.mkdtemp())
    df_path    = str(tmp_dir / "df.pkl")
    folds_path = str(tmp_dir / "folds.pkl")
    le_path    = str(tmp_dir / "label_encoders.pkl")
    with open(df_path, "wb") as f:
        pickle.dump(df, f)
    with open(folds_path, "wb") as f:
        pickle.dump(folds, f)
    with open(le_path, "wb") as f:
        pickle.dump(prep.label_encoders, f)

    # ── 실행 모델 선택 (구버전 alias 흡수) ──
    if args.models == "all":
        selected_keys = list(MODEL_REGISTRY.keys())
    else:
        selected_keys = []
        for k in args.models.split(","):
            k = _LEGACY_ALIAS.get(k.strip(), k.strip())
            if k in MODEL_REGISTRY and k not in selected_keys:
                selected_keys.append(k)
    if not selected_keys:
        logger.error("실행할 모델이 없습니다: %s (선택지: %s)", args.models, list(MODEL_REGISTRY))
        sys.exit(1)

    optuna_cfg   = cfg.get("optuna", {})
    n_trials_map = optuna_cfg.get("n_trials", {})

    # sqlite storage 디렉토리 자동 생성 — "unable to open database file" 방지
    storage_uri = optuna_cfg.get("storage") or ""
    if storage_uri.startswith("sqlite:///"):
        Path(storage_uri.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)

    # ── 모델별 생성자 인수 — 피처 토글과 룰 엔진을 여기서 주입 ──
    extra_kwargs_map = {
        "text": {
            "use_word": use_word,
            "use_head_noun": use_head,
            "use_alcohol_lexicon": use_alcohol,
            "vectorizer": vectorizer,
            "rule_engine": rule_engine,
        },
        "koelectra": {
            "n_large":  n_classes["large"],
            "n_medium": n_classes["medium"],
            "n_tag":    n_classes["tag"],
        },
    }

    model_jobs = []
    for key in selected_keys:
        model_name, model_cls = MODEL_REGISTRY[key]
        n_trials = args.n_trials or n_trials_map.get(model_name, 50)
        model_jobs.append((
            key, model_name, model_cls,
            cfg.get(model_name, {}),
            n_trials,
            extra_kwargs_map[key],
        ))

    # ── 실험 실행 (기존 구조 유지) ──
    all_results:       dict = {}
    all_study_results: dict = {}
    all_fold_details:  dict = {}

    t_start = time.perf_counter()

    if args.parallel and len(model_jobs) > 1:
        logger.info("병렬 실행 모드 (%d workers)", len(model_jobs))
        futures = {}
        with ProcessPoolExecutor(max_workers=len(model_jobs)) as executor:
            for key, mname, mcls, sspace, nt, ekwargs in model_jobs:
                future = executor.submit(
                    run_single_model,
                    key, mname, mcls, sspace, nt,
                    df_path, folds_path, le_path, ekwargs, optuna_cfg, args.output,
                    args.study_version, args.reset_storage, cv_strategy,
                )
                futures[future] = key

            for future in as_completed(futures):
                key = futures[future]
                try:
                    r = future.result()
                    all_results[r["model_name"]]       = r["cv_result"]
                    all_study_results[r["model_name"]] = r["study_result"]
                    all_fold_details[r["model_name"]]  = r["fold_details"]
                    logger.info("[%s] 완료: large_F1=%.4f", key, r["cv_result"].large_f1)
                except Exception as e:
                    logger.error("[%s] 실패: %s", key, e, exc_info=True)
    else:
        for key, mname, mcls, sspace, nt, ekwargs in model_jobs:
            logger.info("=== [%s] 실행 ===", key)
            r = run_single_model(
                key, mname, mcls, sspace, nt,
                df_path, folds_path, le_path, ekwargs, optuna_cfg, args.output,
                args.study_version, args.reset_storage, cv_strategy,
            )
            all_results[r["model_name"]]       = r["cv_result"]
            all_study_results[r["model_name"]] = r["study_result"]
            all_fold_details[r["model_name"]]  = r["fold_details"]

    total_time = time.perf_counter() - t_start
    logger.info("전체 실험 완료: %.1f초", total_time)

    # ── 비교 출력 (전략별 하위 디렉토리) ──
    gate    = cfg.get("gate", {}).get("large_f1_min", 0.90)
    out_dir = Path(args.output) / cv_strategy

    print_comparison_table(all_results, gate_large_f1=gate)

    chart_path = str(out_dir / "comparison_chart.png")
    save_comparison_chart(
        all_results, chart_path, gate_large_f1=gate,
        dpi=cfg.get("output", {}).get("chart_dpi", 150),
    )
    save_comparison_csv(all_results, str(out_dir / "comparison_table.csv"))

    le = prep.label_encoders
    per_class: dict = {}
    err_ex:    dict = {}
    for mname, details in all_fold_details.items():
        per_class[mname] = per_class_f1_report(details, le["large"], task="large")
        err_ex[mname]    = error_examples(details, df, le["large"], n=5000)

    if cfg.get("output", {}).get("html_report", True):
        html_path = str(out_dir / "comparison_report.html")
        save_html_report(
            results=all_results,
            study_results=all_study_results,
            data_summary=summary,
            error_examples=err_ex,
            per_class_dfs=per_class,
            output_path=html_path,
            chart_path=chart_path,
            gate_large_f1=gate,
        )
        logger.info("리포트: %s", html_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
