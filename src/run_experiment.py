"""
RE:FRIDGE Phase 1 — 3-Model 병렬 실험 CLI 진입점.

사용 예:
    # 기본 실행 (GroupKFold)
    python src/run_experiment.py \
        --input product_data_collection/refined_grocery_csv_for_classification/ML_grocery_data_sampled.csv \
        --config configs/experiment.yaml \
        --output results/

    # CV 전략 비교 실험
    python src/run_experiment.py --input ... --cv-strategy StratifiedGroupKFold
    python src/run_experiment.py --input ... --cv-strategy StratifiedKFold
    python src/run_experiment.py --input ... --cv-strategy KFold

    # 특정 모델만
    python src/run_experiment.py --input ... --models tfidf --n-trials 10

    # 3모델 병렬 실행
    python src/run_experiment.py --input ... --models all --parallel

    # storage 초기화
    python src/run_experiment.py --input ... --reset-storage
    python src/run_experiment.py --input ... --study-version v2
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
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_utils import CVStrategy, data_summary, load_data, make_folds
from src.preprocess import REFPreprocessor
from src.models import (
    FastTextKonlpyClassifier,
    KoElectraMultiTaskClassifier,
    TfidfLgbmClassifier,
)
from src.tuning.optuna_runner import OptunaRunner
from src.evaluate import cv_evaluate, error_examples, per_class_f1_report
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
    "tfidf":     ("tfidf_lgbm", TfidfLgbmClassifier),
    "fasttext":  ("fasttext",   FastTextKonlpyClassifier),
    "koelectra": ("koelectra",  KoElectraMultiTaskClassifier),
}

CV_STRATEGIES: list[CVStrategy] = [
    "GroupKFold",
    "StratifiedGroupKFold",
    "StratifiedKFold",
    "KFold",
]


# ──────────────────────────────────────────────────────────────────
# Storage 관리 헬퍼
# ──────────────────────────────────────────────────────────────────

def _delete_study_if_exists(study_name: str, storage: str, logger: logging.Logger) -> None:
    try:
        optuna.delete_study(study_name=study_name, storage=storage)
        logger.info("study 삭제 완료: %s", study_name)
    except KeyError:
        logger.info("삭제할 study 없음 (신규): %s", study_name)


def _build_study_name(model_name: str, study_version: str, cv_strategy: str) -> str:
    """
    study 식별자를 생성한다.

    CV 전략이 다르면 완전히 다른 study로 격리한다.
    예: "ref_tfidf_lgbm_GroupKFold_v1"
    """
    return f"ref_{model_name}_{cv_strategy}_{study_version}"


# ──────────────────────────────────────────────────────────────────
# 단일 모델 실험 함수 (ProcessPoolExecutor worker)
# ──────────────────────────────────────────────────────────────────

def run_single_model(
    model_key: str,
    model_name: str,
    model_cls,
    search_space: dict,
    n_trials: int,
    df_path: str,
    folds_path: str,
    extra_kwargs: dict,
    optuna_cfg: dict,
    output_dir: str,
    study_version: str,
    reset_storage: bool,
    cv_strategy: str,      # ← 추가: 결과 디렉토리 / study name 격리용
) -> dict:
    import pickle
    from src.tuning.optuna_runner import OptunaRunner
    from src.evaluate import cv_evaluate, per_class_f1_report, error_examples

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    logger = logging.getLogger(model_key)

    with open(df_path, "rb") as f:
        df = pickle.load(f)
    with open(folds_path, "rb") as f:
        folds = pickle.load(f)

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

    # CV 전략별로 결과 디렉토리를 분리해서 저장 (덮어쓰기 방지)
    out = Path(output_dir) / cv_strategy / model_name
    out.mkdir(parents=True, exist_ok=True)

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
        description="RE:FRIDGE Phase 1 — 3-Model 실험 비교"
    )
    parser.add_argument("--input",   required=True, help="CSV 또는 XLSX 입력 파일 경로")
    parser.add_argument("--config",  default="configs/experiment.yaml", help="실험 설정 파일")
    parser.add_argument("--output",  default="results/", help="결과 저장 디렉토리")
    parser.add_argument(
        "--models", default="all",
        help="실행할 모델 (all | tfidf | fasttext | koelectra | 쉼표 분리)",
    )
    parser.add_argument("--n-trials", type=int, default=None,
                        help="모델당 Optuna trial 수 (설정 파일 값 override)")
    parser.add_argument("--parallel", action="store_true",
                        help="ProcessPoolExecutor로 3모델 동시 실행")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # ── CV 전략 ──
    parser.add_argument(
        "--cv-strategy",
        type=str,
        default=None,   # None이면 experiment.yaml 값 사용
        choices=CV_STRATEGIES,
        help=(
            "CV 분할 전략. 미지정 시 experiment.yaml cv.strategy 값 사용.\n"
            "  GroupKFold           : 브랜드 기준 분리 (기본, 데이터 누수 방지)\n"
            "  StratifiedGroupKFold : 브랜드 분리 + 클래스 비율 유지 (권장)\n"
            "  StratifiedKFold      : 클래스 비율 유지, 브랜드 누수 가능\n"
            "  KFold                : 완전 랜덤, 베이스라인 비교용"
        ),
    )

    # ── Storage 관리 ──
    parser.add_argument(
        "--study-version",
        type=str,
        default="v1",
        help="Optuna study name suffix. 지표/전략 변경 시 v2, v3 등으로 올려서 격리.",
    )
    parser.add_argument(
        "--reset-storage",
        action="store_true",
        help="실행 전 해당 study의 기존 Optuna 결과를 삭제하고 trial 0부터 재시작.",
    )

    return parser.parse_args()


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

    # CV 전략: CLI 인수 > experiment.yaml > 기본값 순으로 결정
    cv_strategy: CVStrategy = (
        args.cv_strategy
        or cfg.get("cv", {}).get("strategy", "GroupKFold")
    )
    logger.info(
        "실험 설정 — cv_strategy=%s, study_version=%s, reset_storage=%s",
        cv_strategy, args.study_version, args.reset_storage,
    )

    # ── 데이터 로드 & 전처리 ──
    logger.info("데이터 로드: %s", args.input)
    df_raw = load_data(
        input_path=args.input,
        col_map=cfg["data"]["input_columns"],
        exclude_large=cfg["data"].get("exclude_large"),
        exclude_tag=cfg["data"].get("exclude_tag"),
    )

    prep = REFPreprocessor(
        brand_dict_path=cfg["data"].get(
            "brand_dict_path",
            "product_data_collection/not_grocery_and_brand_list/grocery_brand_name.json",
        ),
        stopwords=cfg["preprocessing"].get("stopwords", []),
        alcohol_brand_preserve=cfg["preprocessing"].get("alcohol_brand_preserve", True),
        use_parser=cfg["preprocessing"].get("use_parser", True),
        morpheme_analyzer=cfg["preprocessing"].get("morpheme_analyzer", "Okt"),
    )
    df = prep.fit_transform(df_raw)

    # make_folds()에 cv_strategy 전달
    folds = make_folds(
        df,
        n_splits=cfg["cv"]["n_splits"],
        group_col=cfg["cv"]["group_col"],
        target_col=cfg["cv"].get("target_col", "large_category"),
        strategy=cv_strategy,
        seed=cfg["cv"]["seed"],
    )
    n_classes = prep.n_classes
    summary   = data_summary(df)
    summary["cv_strategy"] = cv_strategy  # 리포트에 전략명 포함

    # ── IPC용 임시 pickle ──
    import pickle, tempfile, shutil
    tmp_dir    = Path(tempfile.mkdtemp())
    df_path    = str(tmp_dir / "df.pkl")
    folds_path = str(tmp_dir / "folds.pkl")
    with open(df_path, "wb") as f:
        pickle.dump(df, f)
    with open(folds_path, "wb") as f:
        pickle.dump(folds, f)

    # ── 실행 모델 선택 ──
    if args.models == "all":
        selected_keys = list(MODEL_REGISTRY.keys())
    else:
        selected_keys = [
            k.strip() for k in args.models.split(",")
            if k.strip() in MODEL_REGISTRY
        ]

    optuna_cfg   = cfg.get("optuna", {})
    n_trials_map = optuna_cfg.get("n_trials", {})

    extra_kwargs_map = {
        "tfidf":     {},
        "fasttext":  {},
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

    # ── 실험 실행 ──
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
                    df_path, folds_path, ekwargs, optuna_cfg, args.output,
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
                df_path, folds_path, ekwargs, optuna_cfg, args.output,
                args.study_version, args.reset_storage, cv_strategy,
            )
            all_results[r["model_name"]]       = r["cv_result"]
            all_study_results[r["model_name"]] = r["study_result"]
            all_fold_details[r["model_name"]]  = r["fold_details"]

    total_time = time.perf_counter() - t_start
    logger.info("전체 실험 완료: %.1f초", total_time)

    # ── 비교 출력 (전략별 output 하위 디렉토리에 저장) ──
    gate       = cfg.get("gate", {}).get("large_f1_min", 0.85)
    out_dir    = Path(args.output) / cv_strategy

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
        err_ex[mname]    = error_examples(details, df, le["large"], n=20)

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