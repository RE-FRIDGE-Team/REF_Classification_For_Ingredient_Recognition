"""
RE:FRIDGE Phase 1 — 실험 CLI 진입점 (2026-07 2차 개편).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[2차 개편 요약]
  - --keep-brand 의미 재정의: 브랜드 '원위치 방치'가 아닌 '재조립' 방식.
    쪼개기(브랜드/용량/수량 추출)는 항상 수행하되, refined_text 를
    [브랜드] + 정제명 + [정규화 용량] 으로 재부착 (수량·노이즈만 탈락).
      예) "농심 양파링 오리지널, 80g, 3개 3,960원 …" → "농심 양파링 오리지널 80g"
  - 위치 피처(gin-head) 신설: PGIN 어휘 기반 복합어 우핵 분해로
    "마지막 GIN = 핵심 GIN" 위치 개념을 필드 가중(BM25F 방식)으로 구현.
      예) 김치치즈돈가스 → head=돈가스(강가중) / mods=김치·치즈(약가중)
    PGIN 어휘는 Mecab 사용자사전용 수집분을 공용으로 재사용 (사람 개입 0).
  - 술 사전 정비: 저집중 스타일어(년산·로제·진·숙성·스파클링) 삭제,
    단어별 blocker/boundary 가드 도입(칵테일새우·진저에일·크럼블 차단),
    술 브랜드 가제티어 블록 신설(봄베이사파이어형 무단서 브랜드 대응).
    가제티어 확장은 tools/mine_alcohol_brands.py 로 집중도 마이닝.
  - 결과 리포팅 개편: 출력 폴더/파일명에 모드태그_타임스탬프 부착으로
    덮어쓰기 원천 차단 (예: results/keep_brand_2026-07-18-14-27/
    comparison_report_keep_brand_2026-07-18-14-27.html).
    HTML 리포트에 전 fold OOF 오류분석(중분류 포함)·Other Statistics
    (오분류 순위·계층 오류 3분면·태그 정합성·F1 외 보조지표) 섹션 신설.

[1차 개편 유지 사항]
  - MODEL_REGISTRY: fasttext 삭제, tfidf → text(TextPipelineClassifier) 교체.
    koelectra 는 술 OOV 구제 후순위 옵션으로 유지.
  - 모든 전처리/피처 정책이 CLI 토글로 노출됨 (2순위 원칙).
    CLI 미지정 시 experiment.yaml 값 → 그것도 없으면 코드 기본값.
  - HeadRuleEngine(모호 헤드어 룰)을 LabelEncoder 로 생성해 모델에 주입.
  - dedup: 전처리 후 · fold 생성 전 근접중복 원본 제거.

[사용 예 — A/B 실험 축]
  # ① 재베이스라인 (판정은 항상 StratifiedGroupKFold)
  python src/run_experiment.py --input data.csv --models text --n-trials 50 --study-version v3

  # ② ★keep-brand 재조립 — 브랜드+용량 보존 축
  python src/run_experiment.py --input data.csv --models text --keep-brand --study-version v3_kb

  # ③ ★gin-head 절제 — 위치 피처 기여도 측정 (끄면 구버전 head-noun 폴백)
  python src/run_experiment.py --input data.csv --models text --no-gin-head --study-version v3_nogin

  # ④ ★술 피처 절제 — 스타일어/브랜드 가제티어 개별 기여도
  python src/run_experiment.py --input data.csv --models text --no-alcohol-lexicon --study-version v3_nolex
  python src/run_experiment.py --input data.csv --models text --no-alcohol-brands --study-version v3_nobrand

  # ⑤ 술 브랜드 가제티어 갱신 (학습 전 1회, resources/alcohol_brands.txt 생성)
  python tools/mine_alcohol_brands.py --input data.csv --out resources/alcohol_brands.txt

  # ⑥ 형태소/벡터라이저 비교 (1차 개편 축 유지)
  python src/run_experiment.py --input data.csv --models text --morpheme okt --study-version v3_okt
  python src/run_experiment.py --input data.csv --models text --vectorizer bm25 --study-version v3_bm25

  # (후순위) 술 OOV 구제용 KoELECTRA 추가
  python src/run_experiment.py --input data.csv --models text,koelectra --parallel

  [타임 스케줄러 사용 예]
 # ① 기본 계획서대로 밤새 실행 — 이거 하나 치고 자면 됨
 #    (재베이스라인 2h → keep-brand 2h → gin-head 절제 2h → 스타일어 절제 2h → 브랜드 가제티어 절제 2h)
 python src/run_batch.py --input data.csv

 # ② 자기 전 점검용 dry-run — 실행 없이 명령과 실험별 예상 종료 시각만 출력
 python src/run_batch.py --input data.csv --dry-run

 # ③ 예약/유예 시간 조정, 다른 계획서 사용
 python src/run_batch.py --input data.csv --plan configs/my_plan.yaml --reserve 15m --grace 20m

 # ④ 스케줄러 없이 단독 실험에 시간 제한만 걸 때
 python src/run_experiment.py --input data.csv --models text --n-trials 10000 --timeout 2h

 python src/run_experiment.py --input product_data_collection/refined_grocery_csv_for_classification/recognition_dataset_augmented.csv --models text --hpo-mode two_stage --n-trials 10000 --timeout 3h --study-version v4_2s
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
    build_oof_table,
    cv_evaluate,
    error_examples,
    hierarchical_error_stats,
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
    run_dir: str,
    hpo_mode: str = "joint",
    stage1_frac: float = 0.6,
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
    timeout_total = optuna_cfg.get("timeout_per_model", 3600)

    common = dict(
        model_name=model_name, model_cls=model_cls, search_space=search_space,
        folds=folds, df=df, n_trials=n_trials,
        sampler=optuna_cfg.get("sampler", "TPE"),
        pruner=optuna_cfg.get("pruner", "MedianPruner"),
        n_startup_trials=optuna_cfg.get("n_startup_trials", 10),
        n_warmup_steps=optuna_cfg.get("n_warmup_steps", 2),
        storage=storage, extra_kwargs=extra_kwargs,
    )

    if hpo_mode == "two_stage":
        # ── ★2단계 HPO: 1단계(피처+대분류, 대분류 F1) → 동결 → 2단계(중분류+디코딩, 중분류 F1) ──
        import pandas as _pd
        from src.tuning.optuna_runner import StudyResult

        t1 = timeout_total * stage1_frac
        t2 = max(timeout_total - t1, 300)
        if reset_storage and storage:
            _delete_study_if_exists(f"{study_name}_L", storage, logger)
            _delete_study_if_exists(f"{study_name}_M", storage, logger)

        logger.info("[%s] 2단계 HPO — stage1(large) %ds / stage2(medium) %ds", model_key, t1, t2)
        s1 = OptunaRunner(**common, timeout=t1, level="large",
                          study_name=f"{study_name}_L").run()
        logger.info("[%s] stage1 확정 large_F1=%.4f → 동결 후 stage2 진입", model_key, s1.best_score)
        s2 = OptunaRunner(**common, timeout=t2, level="medium",
                          frozen_params=s1.best_params,
                          study_name=f"{study_name}_M").run()
        logger.info("[%s] stage2 확정 medium_F1=%.4f (decode=%s)", model_key,
                    s2.best_score, s2.best_params.get("decode"))

        trials_all = _pd.concat(
            [d.assign(stage=s) for d, s in
             [(s1.all_trials_df, "1_large"), (s2.all_trials_df, "2_medium")] if d is not None],
            ignore_index=True,
        )
        study_result = StudyResult(
            model_name=model_name,
            best_score=s1.best_score,                      # 게이트 판정 기준 = 대분류 F1 유지
            best_params={**s1.best_params, **s2.best_params},
            all_trials_df=trials_all,
        )
    else:
        if reset_storage and storage:
            _delete_study_if_exists(study_name, storage, logger)
        logger.info("[%s] HPO 시작: %d trials (study=%s)", model_key, n_trials, study_name)
        study_result = OptunaRunner(**common, timeout=timeout_total,
                                    study_name=study_name).run()

    logger.info("[%s] 최적 파라미터로 CV 평가 시작", model_key)
    cv_result, fold_details = cv_evaluate(
        model_cls=model_cls,
        best_params=study_result.best_params,
        extra_kwargs=extra_kwargs,
        df=df,
        folds=folds,
    )

    # ★실행 단위 결과 디렉토리 — 모드_타임스탬프 폴더로 이전 결과와 절대 미충돌
    out = Path(output_dir) / run_dir / model_name
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
    parser.add_argument("--hpo-mode", choices=["joint", "two_stage"], default=None,
                        help="joint(기존: 대분류 F1 단일 목적) | two_stage(★1단계 대분류 → "
                             "동결 → 2단계 중분류+빔디코딩, 레벨별 목적함수 분리)")
    parser.add_argument("--stage1-frac", type=float, default=None,
                        help="two_stage 시 1단계(대분류)에 배정할 시간 비율 (기본 0.6)")
    parser.add_argument("--timeout", type=str, default=None,
                        help="모델당 HPO 시간 예산 (예: 2h, 90m, 1h30m). "
                             "지정 시 n-trials 와 먼저 도달하는 쪽에서 탐색 종료. "
                             "시간으로만 제한하려면 --n-trials 를 크게(예: 10000) 설정.")
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
                       help="★재정의: 쪼개기는 그대로 수행하되 refined_text 를 "
                            "'브랜드+정제명+용량'으로 재조립 (예: 농심 양파링 오리지널 80g)")
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
    g_feat.add_argument("--gin-head", dest="use_gin_head",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="GIN 어휘 우핵 분해 위치 필드 (--no-gin-head 로 구버전 head 폴백)")
    g_feat.add_argument("--alcohol-lexicon", dest="use_alcohol_lexicon",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="술 스타일어 카운트 피처 블록 (가드 반영 매처)")
    g_feat.add_argument("--alcohol-brands", dest="use_alcohol_brands",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="술 브랜드 가제티어 카운트 블록 (봄베이사파이어형 대응)")
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


# 결과물 타임스탬프 시간대 — 실행 환경(Docker/WSL)이 UTC 여도 한국 시간 고정
_TZ_NAME = "Asia/Seoul"

def now_kst():
    """KST 현재 시각 (tzdata 부재 등 실패 시 로컬 시각 폴백)."""
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(_TZ_NAME))
    except Exception:
        return datetime.now()


def parse_duration(text: str) -> int:
    """
    시간 예산 문자열 → 초 변환. 허용 형식: "2h", "90m", "30s", "1h30m", "7200"(초).
    """
    import re as _re
    s = str(text).strip().lower()
    if s.isdigit():
        return int(s)
    total, matched = 0, False
    for val, unit in _re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", s):
        total += float(val) * {"h": 3600, "m": 60, "s": 1}[unit]
        matched = True
    if not matched:
        raise ValueError(f"시간 형식 해석 불가: {text!r} (예: 2h, 90m, 1h30m, 7200)")
    return int(total)


def _build_run_tag(pre_opt, use_word, use_head, use_gin_head,
                   use_alcohol, use_alc_brand, vectorizer) -> str:
    """
    실행 모드 식별 태그 생성 — 결과 파일/폴더명에 부착되는 A/B 축 요약.

    기본값과 다른 토글만 나열하는 방식 (전부 기본이면 "default").
      예) --keep-brand 단독 실행 → "keep_brand"
          --keep-brand --no-gin-head → "keep_brand_no_gin"
    """
    parts: list[str] = []
    if pre_opt.keep_brand_volume:
        parts.append("keep_brand")
    if pre_opt.placeholder:
        parts.append("placeholder")
    if pre_opt.morpheme_analyzer != "mecab":
        parts.append(pre_opt.morpheme_analyzer)
    if not use_word:
        parts.append("no_word")
    if not use_head:
        parts.append("no_head")
    elif not use_gin_head:
        parts.append("no_gin")
    if not use_alcohol:
        parts.append("no_lex")
    if not use_alc_brand:
        parts.append("no_brandgaz")
    if vectorizer != "tfidf":
        parts.append(vectorizer)
    return "_".join(parts) or "default"


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
        # ★--keep-brand = 재조립 모드 ('브랜드+정제명+용량' 재부착, 수량·노이즈만 탈락)
        keep_brand_volume=_resolve(args.keep_brand, pre_cfg.get("keep_brand_volume"), False),
        # --keep-volume/--keep-quantity 는 구버전 '원위치 유지' 미세 토글로 존치
        remove_volume=not _resolve(args.keep_volume, not pre_cfg.get("remove_volume", True), False),
        remove_quantity=not _resolve(args.keep_quantity, not pre_cfg.get("remove_quantity", True), False),
        placeholder=_resolve(args.placeholder, pre_cfg.get("placeholder"), False),
        morpheme_analyzer=_resolve(args.morpheme, pre_cfg.get("morpheme_analyzer"), "mecab"),
        alcohol_brand_preserve=pre_cfg.get("alcohol_brand_preserve", True),
    )
    do_dedup = _resolve(args.dedup, pre_cfg.get("dedup"), True)

    # ── 피처 토글 해석 ──
    use_word     = _resolve(args.use_word, feat_cfg.get("use_word"), True)
    use_head     = _resolve(args.use_head_noun, feat_cfg.get("use_head_noun"), True)
    use_gin_head = _resolve(args.use_gin_head, feat_cfg.get("use_gin_head"), True)
    use_alcohol  = _resolve(args.use_alcohol_lexicon, feat_cfg.get("use_alcohol_lexicon"), True)
    use_alc_brand = _resolve(args.use_alcohol_brands, feat_cfg.get("use_alcohol_brands"), True)
    use_rules    = _resolve(args.use_rules, feat_cfg.get("use_rules"), True)
    vectorizer   = _resolve(args.vectorizer, feat_cfg.get("vectorizer"), "tfidf")

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
    # PGIN 어휘: Mecab 사용자사전 + GIN 핵어 분해 양쪽의 공용 도메인 어휘원
    domain_words = (
        _collect_domain_words(args.input, cfg["data"].get("pgin_column"))
        if (pre_opt.morpheme_analyzer == "mecab" or use_gin_head) else []
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

    # ── ★실행 식별자 — 모드 태그 + 결과 산출 시각 (덮어쓰기 원천 차단) ──
    hpo_mode    = _resolve(args.hpo_mode, cfg.get("optuna", {}).get("hpo_mode"), "joint")
    stage1_frac = _resolve(args.stage1_frac, cfg.get("optuna", {}).get("stage1_frac"), 0.6)

    run_ts   = now_kst().strftime("%Y-%m-%d-%H-%M")   # 폴더/파일명 타임스탬프 KST 고정
    mode_tag = _build_run_tag(pre_opt, use_word, use_head, use_gin_head,
                              use_alcohol, use_alc_brand, vectorizer)
    if hpo_mode == "two_stage":
        mode_tag = f"{mode_tag}_2stage"               # 결과 폴더에서 HPO 방식 즉시 식별
    run_suffix = f"{mode_tag}_{run_ts}"
    run_dir    = run_suffix                     # 폴더명 = 파일 접미와 동일 체계
    logger.info("실행 식별자: %s (cv=%s)", run_suffix, cv_strategy)

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

    # ── ★--timeout: HPO 시간 예산 override (배치 스케줄러의 시간 제어 지점) ──
    if args.timeout:
        budget = parse_duration(args.timeout)
        optuna_cfg = {**optuna_cfg, "timeout_per_model": budget}
        logger.info("HPO 시간 예산: %d초 (%.1f시간)", budget, budget / 3600)

    # sqlite storage 디렉토리 자동 생성 — "unable to open database file" 방지
    storage_uri = optuna_cfg.get("storage") or ""
    if storage_uri.startswith("sqlite:///"):
        Path(storage_uri.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)

    # ── 모델별 생성자 인수 — 피처 토글과 룰 엔진을 여기서 주입 ──
    extra_kwargs_map = {
        "text": {
            "use_word": use_word,
            "use_head_noun": use_head,
            "use_gin_head": use_gin_head,
            "use_alcohol_lexicon": use_alcohol,
            "use_alcohol_brands": use_alc_brand,
            "vectorizer": vectorizer,
            "rule_engine": rule_engine,
            "gin_vocab": domain_words if use_gin_head else None,
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
                    args.study_version, args.reset_storage, cv_strategy, run_dir,
                    hpo_mode, stage1_frac,
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
                args.study_version, args.reset_storage, cv_strategy, run_dir,
                hpo_mode, stage1_frac,
            )
            all_results[r["model_name"]]       = r["cv_result"]
            all_study_results[r["model_name"]] = r["study_result"]
            all_fold_details[r["model_name"]]  = r["fold_details"]

    total_time = time.perf_counter() - t_start
    logger.info("전체 실험 완료: %.1f초", total_time)

    # ── 비교 출력 — ★모드_타임스탬프 폴더 + 파일명 접미로 덮어쓰기 원천 차단 ──
    gate    = cfg.get("gate", {}).get("large_f1_min", 0.90)
    out_dir = Path(args.output) / run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print_comparison_table(all_results, gate_large_f1=gate)

    chart_path = str(out_dir / f"comparison_chart_{run_suffix}.png")
    save_comparison_chart(
        all_results, chart_path, gate_large_f1=gate,
        dpi=cfg.get("output", {}).get("chart_dpi", 150),
    )
    save_comparison_csv(all_results, str(out_dir / f"comparison_table_{run_suffix}.csv"))

    # ── ★전 fold OOF 기반 오류 분석·계층 통계 (중분류 포함) ──
    le = prep.label_encoders
    per_class:   dict = {}
    err_ex:      dict = {}
    other_stats: dict = {}
    for mname, details in all_fold_details.items():
        per_class[mname]   = per_class_f1_report(details, le["large"], task="large")
        err_ex[mname]      = error_examples(details, df, le, n=5000)
        oof                = build_oof_table(details, df, le)
        other_stats[mname] = hierarchical_error_stats(oof)
        # 오류 전량 CSV — HTML 표기 제한(300행) 초과분 열람용
        err_ex[mname].to_csv(
            out_dir / f"error_analysis_{mname}_{run_suffix}.csv",
            index=False, encoding="utf-8-sig",
        )

    if cfg.get("output", {}).get("html_report", True):
        html_path = str(out_dir / f"comparison_report_{run_suffix}.html")
        save_html_report(
            results=all_results,
            study_results=all_study_results,
            data_summary=summary,
            error_examples=err_ex,
            per_class_dfs=per_class,
            output_path=html_path,
            chart_path=chart_path,
            gate_large_f1=gate,
            other_stats=other_stats,
            run_label=f"{run_suffix} (cv={cv_strategy})",
        )
        logger.info("리포트: %s", html_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
