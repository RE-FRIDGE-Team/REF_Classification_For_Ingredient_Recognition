"""
RE:FRIDGE Phase 1 — 3-Model 병렬 실험 CLI 진입점.

사용 예:
    # 전체 실행 (순차)
    python src/run_experiment.py \
        --input product_data_collection/refined_grocery_csv_for_classification/ML_grocery_data_sampled.csv \
        --config configs/experiment.yaml \
        --output results/

    # 특정 모델만
    python src/run_experiment.py --input ... --models tfidf --n-trials 10

    # 3모델 병렬 실행
    python src/run_experiment.py --input ... --models all --parallel

    # 평가 지표 변경 후 storage 초기화 (trial 0부터 재시작)
    python src/run_experiment.py --input ... --reset-storage

    # 기존 결과 보존하면서 새 study로 격리 (지표 변경 시 권장)
    python src/run_experiment.py --input ... --study-version v2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import optuna
import yaml

# 프로젝트 루트를 PYTHONPATH에 추가 (src/ 내부에서 실행 시 필요)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_utils import data_summary, load_data, make_folds
from src.preprocess import REFPreprocessor
from src.models import (
    TfidfLgbmClassifier,
    FastTextKonlpyClassifier,
    KoElectraMultiTaskClassifier,
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
# 모델 레지스트리
# ──────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "tfidf":      ("tfidf_lgbm",  TfidfLgbmClassifier),
    "fasttext":   ("fasttext",    FastTextKonlpyClassifier),
    "koelectra":  ("koelectra",   KoElectraMultiTaskClassifier),
}


# ──────────────────────────────────────────────────────────────────
# Storage 관리 헬퍼
# ──────────────────────────────────────────────────────────────────

def _delete_study_if_exists(study_name: str, storage: str, logger: logging.Logger) -> None:
    """
    지정된 study_name을 storage에서 삭제한다.
    존재하지 않으면 조용히 넘어간다.

    Args:
        study_name: 삭제할 study 식별자
        storage:    Optuna DB URI (예: "sqlite:///results/optuna.db")
        logger:     호출자 로거
    """
    try:
        optuna.delete_study(study_name=study_name, storage=storage)
        logger.info("study 삭제 완료: %s", study_name)
    except KeyError:
        logger.info("삭제할 study 없음 (신규): %s", study_name)


def _build_study_name(model_name: str, study_version: str) -> str:
    """
    study 식별자를 생성한다.

    Args:
        model_name:    모델 내부 이름 (예: "tfidf_lgbm")
        study_version: CLI --study-version 값 (예: "v1", "v2")

    Returns:
        "ref_tfidf_lgbm_v1" 형태의 study 이름
    """
    return f"ref_{model_name}_{study_version}"


# ──────────────────────────────────────────────────────────────────
# 단일 모델 실험 함수 (ProcessPoolExecutor worker)
# ──────────────────────────────────────────────────────────────────

def run_single_model(
    model_key: str,
    model_name: str,
    model_cls,
    search_space: dict,
    n_trials: int,
    df_path: str,          # pickle 경로 (IPC용)
    folds_path: str,
    extra_kwargs: dict,
    optuna_cfg: dict,
    output_dir: str,
    study_version: str,    # ← 추가: study name suffix
    reset_storage: bool,   # ← 추가: 기존 study 삭제 여부
) -> dict:
    """
    단일 모델의 HPO + CV 평가를 실행하고 결과 딕셔너리를 반환한다.
    ProcessPoolExecutor worker로 실행 가능하도록 최상위 함수로 정의.
    """
    import pickle
    from src.tuning.optuna_runner import OptunaRunner
    from src.evaluate import cv_evaluate, per_class_f1_report, error_examples

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    logger = logging.getLogger(model_key)

    with open(df_path, "rb") as f:
        df = pickle.load(f)
    with open(folds_path, "rb") as f:
        folds = pickle.load(f)

    storage   = optuna_cfg.get("storage")
    study_name = _build_study_name(model_name, study_version)

    # ── storage 초기화 (--reset-storage 플래그) ──
    # storage가 None(메모리 모드)이면 초기화 불필요
    if reset_storage and storage:
        _delete_study_if_exists(study_name, storage, logger)

    # ── HPO ──
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
        study_name=study_name,      # ← 주입
    )
    study_result = runner.run()

    # ── 최적 파라미터로 CV 평가 ──
    logger.info("[%s] 최적 파라미터로 CV 평가 시작", model_key)
    cv_result, fold_details = cv_evaluate(
        model_cls=model_cls,
        best_params=study_result.best_params,
        extra_kwargs=extra_kwargs,
        df=df,
        folds=folds,
    )

    # ── 결과 저장 ──
    out = Path(output_dir) / model_name
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(study_result.best_params, f, indent=2, ensure_ascii=False)

    cv_scores = {
        "large_f1":  cv_result.large_f1,  "large_f1_std":  cv_result.large_f1_std,
        "medium_f1": cv_result.medium_f1, "medium_f1_std": cv_result.medium_f1_std,
        "tag_f1":    cv_result.tag_f1,    "tag_f1_std":    cv_result.tag_f1_std,
        "large_acc": cv_result.large_acc,
        "train_time": cv_result.train_time,
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
    parser.add_argument("--input",    required=True,   help="CSV 또는 XLSX 입력 파일 경로")
    parser.add_argument("--config",   default="configs/experiment.yaml", help="실험 설정 파일")
    parser.add_argument("--output",   default="results/", help="결과 저장 디렉토리")
    parser.add_argument(
        "--models", default="all",
        help="실행할 모델 (all | tfidf | fasttext | koelectra | 쉼표 분리)"
    )
    parser.add_argument("--n-trials", type=int, default=None,
                        help="모델당 Optuna trial 수 (설정 파일 값 override)")
    parser.add_argument("--parallel", action="store_true",
                        help="ProcessPoolExecutor로 3모델 동시 실행")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # ── Storage 관리 인수 ──
    parser.add_argument(
        "--study-version",
        type=str,
        default="v1",
        help=(
            "Optuna study name suffix (기본: v1). "
            "평가 지표 변경 등으로 기존 결과와 격리할 때 v2, v3 등으로 올린다. "
            "storage가 None(메모리)이면 무시된다."
        ),
    )
    parser.add_argument(
        "--reset-storage",
        action="store_true",
        help=(
            "실행 전 해당 study_version의 기존 Optuna study를 삭제하고 trial 0부터 재시작한다. "
            "평가 지표를 변경한 경우 반드시 사용해야 TPE가 오염된 과거 trial을 참조하지 않는다. "
            "storage가 None(메모리)이면 무시된다."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 로깅 설정 ──
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
    Path(args.output).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_experiment")

    # ── study-version / reset-storage 요약 로깅 ──
    logger.info(
        "study_version=%s | reset_storage=%s",
        args.study_version, args.reset_storage,
    )

    # ── 설정 로드 ──
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ── 데이터 로드 & 전처리 ──
    logger.info("데이터 로드: %s", args.input)
    df_raw = load_data(
        input_path=args.input,
        col_map=cfg["data"]["input_columns"],
        exclude_large=cfg["data"].get("exclude_large"),
        exclude_tag=cfg["data"].get("exclude_tag"),
    )

    prep = REFPreprocessor(
        brand_dict_path=cfg["data"].get("brand_dict_path", "product_data_collection/not_grocery_and_brand_list/grocery_brand_name.json"),
        stopwords=cfg["preprocessing"].get("stopwords", []),
        alcohol_brand_preserve=cfg["preprocessing"].get("alcohol_brand_preserve", True),
        use_parser=cfg["preprocessing"].get("use_parser", True),
        morpheme_analyzer=cfg["preprocessing"].get("morpheme_analyzer", "Okt"),
    )
    df = prep.fit_transform(df_raw)

    folds = make_folds(
        df,
        n_splits=cfg["cv"]["n_splits"],
        group_col=cfg["cv"]["group_col"],
        seed=cfg["cv"]["seed"],
    )
    n_classes = prep.n_classes
    summary   = data_summary(df)

    # ── IPC용 임시 pickle (병렬 실행 시 worker에게 데이터 전달) ──
    import pickle, tempfile
    tmp_dir = Path(tempfile.mkdtemp())
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
        selected_keys = [k.strip() for k in args.models.split(",") if k.strip() in MODEL_REGISTRY]

    optuna_cfg = cfg.get("optuna", {})
    n_trials_map: dict = optuna_cfg.get("n_trials", {})

    # KoELECTRA는 n_classes 주입 필요
    extra_kwargs_map = {
        "tfidf":     {},
        "fasttext":  {},
        "koelectra": {
            "n_large":  n_classes["large"],
            "n_medium": n_classes["medium"],
            "n_tag":    n_classes["tag"],
        },
    }

    # ── 실험 실행 ──
    all_results:       dict = {}
    all_study_results: dict = {}
    all_fold_details:  dict = {}

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
                    args.study_version,   # ← 추가
                    args.reset_storage,   # ← 추가
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
                    logger.error("[%s] 실패: %s", key, e)
    else:
        for key, mname, mcls, sspace, nt, ekwargs in model_jobs:
            logger.info("=== [%s] 실행 ===", key)
            r = run_single_model(
                key, mname, mcls, sspace, nt,
                df_path, folds_path, ekwargs, optuna_cfg, args.output,
                args.study_version,   # ← 추가
                args.reset_storage,   # ← 추가
            )
            all_results[r["model_name"]]       = r["cv_result"]
            all_study_results[r["model_name"]] = r["study_result"]
            all_fold_details[r["model_name"]]  = r["fold_details"]

    total_time = time.perf_counter() - t_start
    logger.info("전체 실험 완료: %.1f초", total_time)

    # ── 비교 출력 ──
    gate = cfg.get("gate", {}).get("large_f1_min", 0.85)

    print_comparison_table(all_results, gate_large_f1=gate)

    chart_path = str(Path(args.output) / "comparison_chart.png")
    save_comparison_chart(all_results, chart_path, gate_large_f1=gate,
                          dpi=cfg.get("output", {}).get("chart_dpi", 150))

    save_comparison_csv(all_results, str(Path(args.output) / "comparison_table.csv"))

    # per-class F1 & error examples (대분류 기준)
    le = prep.label_encoders
    per_class: dict = {}
    err_ex:    dict = {}
    for mname, details in all_fold_details.items():
        per_class[mname] = per_class_f1_report(details, le["large"], task="large")
        err_ex[mname]    = error_examples(details, df, le["large"], n=20)

    if cfg.get("output", {}).get("html_report", True):
        html_path = str(Path(args.output) / "comparison_report.html")
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

    # 임시 파일 정리
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()