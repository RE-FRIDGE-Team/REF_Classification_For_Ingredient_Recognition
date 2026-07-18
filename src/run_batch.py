"""
RE:FRIDGE 배치 스케줄러 — 실험 설정 여러 개를 시간 예산대로 밤새 순차 실행.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[목적]
  독립 시행 실험을 하나 끝날 때마다 수동으로 재실행하는 대기 시간 낭비 제거.
  계획서(YAML)에 "재베이스라인 2h, keep-brand 2h, gin-head 절제 2h …" 식으로
  적어두면 명령 한 번으로 전부 순차 실행되는 구조 — 자고 일어나면 완료.

[시간 예산 동작 방식]
  각 run 의 time = 해당 실험의 총 벽시계 예산.
    HPO 탐색 시간   = time - reserve (기본 20분: 최종 CV 재학습 + 리포트 몫)
    강제 종료 한계  = time + grace   (기본 30분: 진행 중 trial 완주 허용분)
  Optuna timeout 은 '새 trial 시작 중단' 방식이라 마지막 trial 완주만큼
  초과될 수 있고, grace 는 그 오버런의 안전판. 한계 초과 시 프로세스를 죽이고
  다음 실험으로 넘어간다 (한 실험의 폭주가 밤 전체를 잡아먹는 사고 방지).

[안전 설계]
  - 각 실험은 독립 서브프로세스 — 실패/타임아웃이 다음 실험에 전파되지 않음.
  - 결과는 실험별 모드태그_타임스탬프 폴더로 자동 격리 (덮어쓰기 원천 차단).
  - Ctrl+C: 현재 실험만 중단하고 요약 출력 후 종료.
  - 종료 시 batch_summary_*.csv 에 실험별 상태·소요시간 기록.

[사용 예]
  # 기본 계획서(configs/batch_plan.yaml)대로 밤새 실행
  python src/run_batch.py --input data.csv

  # 실행 전 명령·예상 종료 시각만 확인 (실제 실행 없음)
  python src/run_batch.py --input data.csv --dry-run

  # 다른 계획서 지정
  python src/run_batch.py --input data.csv --plan configs/my_plan.yaml
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.run_experiment import now_kst, parse_duration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [batch] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
# 로그 시각도 KST 고정 — 실행 환경(Docker/WSL)이 UTC 여도 한국 시간 표기
logging.Formatter.converter = lambda *args: now_kst().timetuple()
logger = logging.getLogger("run_batch")


# ══════════════════════════════════════════════════════════════════
# 계획서 로드
# ══════════════════════════════════════════════════════════════════

@dataclass
class BatchRun:
    """계획서의 실험 1건."""
    name:          str
    budget_sec:    int
    args:          list[str] = field(default_factory=list)
    study_version: str = ""
    n_trials:      int = 10000       # 시간이 바인딩되도록 큰 기본값
    models:        str = "text"


def load_plan(plan_path: str | Path) -> tuple[list[BatchRun], dict]:
    """
    배치 계획서 YAML 로드.

    형식:
        defaults:            # 전 run 공통 기본값 (run 에서 개별 override 가능)
          n_trials: 10000
          models: text
        runs:
          - name: rebaseline
            time: 2h         # 총 벽시계 예산 (2h / 90m / 1h30m / 7200 허용)
            args: []
          - name: keep_brand
            time: 2h
            args: ["--keep-brand"]
    """
    with open(plan_path, encoding="utf-8") as f:
        plan = yaml.safe_load(f)

    defaults = plan.get("defaults", {}) or {}
    runs: list[BatchRun] = []
    for i, r in enumerate(plan.get("runs", [])):
        name = str(r.get("name") or f"run{i+1}")
        if "time" not in r:
            raise ValueError(f"run '{name}' 에 time 항목 누락 (예: time: 2h)")
        runs.append(BatchRun(
            name=name,
            budget_sec=parse_duration(r["time"]),
            args=[str(a) for a in (r.get("args") or [])],
            study_version=str(r.get("study_version") or f"batch_{name}"),
            n_trials=int(r.get("n_trials", defaults.get("n_trials", 10000))),
            models=str(r.get("models", defaults.get("models", "text"))),
        ))
    if not runs:
        raise ValueError("계획서에 runs 항목이 비어 있음")
    return runs, defaults


# ══════════════════════════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════════════════════════

def build_command(run: BatchRun, args: argparse.Namespace, hpo_sec: int) -> list[str]:
    """단일 실험의 run_experiment.py 서브프로세스 명령 조립."""
    cmd = [
        sys.executable, str(_ROOT / "src" / "run_experiment.py"),
        "--input", args.input,
        "--config", args.config,
        "--output", args.output,
        "--models", run.models,
        "--n-trials", str(run.n_trials),
        "--timeout", f"{hpo_sec}s",
        "--study-version", run.study_version,
    ]
    cmd += run.args
    return cmd


def format_td(seconds: float) -> str:
    """초 → 'H:MM:SS' 표기."""
    return str(timedelta(seconds=int(seconds)))


def main() -> None:
    parser = argparse.ArgumentParser(description="RE:FRIDGE 실험 배치 스케줄러")
    parser.add_argument("--input", required=True, help="CSV/XLSX 학습 데이터 경로")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output", default="results")
    parser.add_argument("--plan", default="configs/batch_plan.yaml",
                        help="배치 계획서 YAML 경로")
    parser.add_argument("--reserve", default="20m",
                        help="예산 중 최종 CV·리포트 몫으로 남길 시간 (HPO 는 time-reserve)")
    parser.add_argument("--grace", default="30m",
                        help="예산 초과 강제 종료까지의 유예 (진행 중 trial 완주 허용분)")
    parser.add_argument("--dry-run", action="store_true",
                        help="실행 없이 명령·예상 종료 시각만 출력")
    args = parser.parse_args()

    reserve = parse_duration(args.reserve)
    grace   = parse_duration(args.grace)
    runs, _ = load_plan(args.plan)

    # ── 실행 계획 요약 + 예상 종료 시각 ──
    total_sec = sum(r.budget_sec for r in runs)
    eta = now_kst()
    print(f"\n{'='*74}\n배치 계획: {len(runs)}개 실험, 총 예산 {format_td(total_sec)}"
          f" (실험별 최대 +{format_td(grace)} 유예)"
          f" — 현재 {eta.strftime('%m-%d %H:%M')} KST\n{'='*74}")
    print(f"{'#':<3}{'이름':<16}{'예산':>9}{'HPO':>9}{'예상종료':>18}  인수")
    for i, r in enumerate(runs, 1):
        hpo_sec = max(r.budget_sec - reserve, 300)   # HPO 최소 5분 보장
        eta += timedelta(seconds=r.budget_sec)
        print(f"{i:<3}{r.name:<16}{format_td(r.budget_sec):>9}"
              f"{format_td(hpo_sec):>9}{(eta.strftime('%m-%d %H:%M') + ' KST'):>18}"
              f"  {' '.join(r.args) or '(기본)'}")
    print("=" * 74)

    if args.dry_run:
        print("\n[dry-run] 실행 예정 명령:")
        for r in runs:
            hpo_sec = max(r.budget_sec - reserve, 300)
            print("  " + " ".join(build_command(r, args, hpo_sec)))
        return

    # ── 순차 실행 ──
    batch_ts = now_kst().strftime("%Y-%m-%d-%H-%M")
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / f"batch_{batch_ts}.log"
    records: list[dict] = []

    for i, r in enumerate(runs, 1):
        hpo_sec = max(r.budget_sec - reserve, 300)
        cmd = build_command(r, args, hpo_sec)
        hard_limit = r.budget_sec + grace

        logger.info("[%d/%d] '%s' 시작 — 예산 %s (HPO %s, 강제종료 한계 %s)",
                    i, len(runs), r.name, format_td(r.budget_sec),
                    format_td(hpo_sec), format_td(hard_limit))
        t0 = time.perf_counter()
        status = "OK"
        try:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"\n{'='*74}\n[{now_kst():%Y-%m-%d %H:%M:%S} KST] RUN {i}/{len(runs)}: "
                         f"{r.name}\nCMD: {' '.join(cmd)}\n{'='*74}\n")
                lf.flush()
                # stdout/stderr 를 배치 로그로 합류 — 새벽 크래시 원인 추적용
                proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                      timeout=hard_limit, cwd=str(_ROOT))
            if proc.returncode != 0:
                status = f"FAIL(exit={proc.returncode})"
        except subprocess.TimeoutExpired:
            status = "KILLED(시간초과)"
            logger.warning("'%s' 강제 종료 — 한계 %s 초과", r.name, format_td(hard_limit))
        except KeyboardInterrupt:
            status = "INTERRUPTED"
            logger.warning("사용자 중단 — 현재 실험을 끊고 요약으로 이동")
            records.append({"run": r.name, "status": status,
                            "elapsed": format_td(time.perf_counter() - t0),
                            "budget": format_td(r.budget_sec)})
            break

        elapsed = time.perf_counter() - t0
        records.append({"run": r.name, "status": status,
                        "elapsed": format_td(elapsed),
                        "budget": format_td(r.budget_sec)})
        logger.info("[%d/%d] '%s' 종료 — %s (%s 소요)",
                    i, len(runs), r.name, status, format_td(elapsed))

    # ── 요약 ──
    summary = pd.DataFrame(records)
    summary_path = out_root / f"batch_summary_{batch_ts}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*74}\n배치 완료 요약\n{'='*74}")
    print(summary.to_string(index=False))
    print(f"\n실험별 결과 폴더: {out_root}/<모드태그_타임스탬프>/")
    print(f"통합 로그: {log_path}")
    print(f"요약 CSV : {summary_path}")


if __name__ == "__main__":
    main()
