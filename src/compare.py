"""
결과 집계, Rich 터미널 표, matplotlib 차트, HTML 리포트 생성.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models.base import CVResult
from .tuning.optuna_runner import StudyResult

logger = logging.getLogger(__name__)

# Rich 선택적 임포트
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False
    logger.warning("rich 미설치. 터미널 표는 일반 텍스트로 출력됩니다.")


# ──────────────────────────────────────────────────────────────────
# 터미널 표
# ──────────────────────────────────────────────────────────────────

def print_comparison_table(
    results: dict[str, CVResult],
    gate_large_f1: float = 0.85,
) -> None:
    """
    Rich 표 또는 일반 텍스트로 모델 비교 결과를 출력한다.

    Args:
        results:       {"model_name": CVResult, ...}
        gate_large_f1: 합격 기준 (기본 0.85)
    """
    if _RICH_AVAILABLE:
        _print_rich_table(results, gate_large_f1)
    else:
        _print_plain_table(results, gate_large_f1)


def _print_rich_table(results: dict[str, CVResult], gate: float) -> None:
    console = Console()
    table = Table(
        title="[bold]RE:FRIDGE Phase 1 — Model Comparison[/bold]",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Model",      style="bold cyan",  no_wrap=True)
    table.add_column("large_F1",   justify="center")
    table.add_column("medium_F1",  justify="center")
    table.add_column("tag_F1",     justify="center")
    table.add_column("large_Acc",  justify="center")
    table.add_column("train(s)",   justify="right")
    table.add_column("infer(ms)",  justify="right")

    model_order = ["tfidf_lgbm", "fasttext", "koelectra"]
    display_names = {
        "tfidf_lgbm": "TF-IDF + LightGBM",
        "fasttext":   "FastText + KoNLPy",
        "koelectra":  "KoELECTRA Multi-task",
    }

    passed: list[str] = []
    for name in model_order:
        if name not in results:
            continue
        r = results[name]
        f1_str = (
            f"[bold green]{r.large_f1:.4f}±{r.large_f1_std:.4f}[/bold green]"
            if r.large_f1 >= gate
            else f"[red]{r.large_f1:.4f}±{r.large_f1_std:.4f}[/red]"
        )
        table.add_row(
            display_names.get(name, name),
            f1_str,
            f"{r.medium_f1:.4f}±{r.medium_f1_std:.4f}",
            f"{r.tag_f1:.4f}±{r.tag_f1_std:.4f}",
            f"{r.large_acc:.4f}",
            f"{r.train_time:.1f}",
            f"{r.infer_time_ms:.2f}",
        )
        if r.large_f1 >= gate:
            passed.append(display_names.get(name, name))

    console.print(table)
    if passed:
        console.print(f"\n  [bold green]✓ PASS:[/bold green] {', '.join(passed)} (large_F1 ≥ {gate})")
    else:
        console.print(f"\n  [bold red]✗ FAIL: 합격 모델 없음 (large_F1 기준 {gate})[/bold red]")


def _print_plain_table(results: dict[str, CVResult], gate: float) -> None:
    header = f"{'Model':<25} {'large_F1':>14} {'medium_F1':>14} {'tag_F1':>14} {'train(s)':>10} {'infer(ms)':>10}"
    print("\n" + "=" * len(header))
    print("RE:FRIDGE Phase 1 — Model Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        mark = "✓" if r.large_f1 >= gate else " "
        print(
            f"{mark} {name:<23} "
            f"{r.large_f1:.4f}±{r.large_f1_std:.4f}  "
            f"{r.medium_f1:.4f}±{r.medium_f1_std:.4f}  "
            f"{r.tag_f1:.4f}±{r.tag_f1_std:.4f}  "
            f"{r.train_time:>8.1f}  "
            f"{r.infer_time_ms:>8.2f}"
        )
    print("=" * len(header) + "\n")


# ──────────────────────────────────────────────────────────────────
# matplotlib 차트
# ──────────────────────────────────────────────────────────────────

def save_comparison_chart(
    results: dict[str, CVResult],
    output_path: str,
    gate_large_f1: float = 0.85,
    dpi: int = 150,
) -> None:
    """대/중분류·태그 F1을 Bar chart로 저장한다."""
    model_names = list(results.keys())
    display = {
        "tfidf_lgbm": "TF-IDF\n+LightGBM",
        "fasttext":   "FastText\n+KoNLPy",
        "koelectra":  "KoELECTRA\nMulti-task",
    }
    labels = [display.get(n, n) for n in model_names]

    metrics = {
        "large_F1":  ([r.large_f1  for r in results.values()], [r.large_f1_std  for r in results.values()]),
        "medium_F1": ([r.medium_f1 for r in results.values()], [r.medium_f1_std for r in results.values()]),
        "tag_F1":    ([r.tag_f1    for r in results.values()], [r.tag_f1_std    for r in results.values()]),
    }

    x = np.arange(len(model_names))
    width = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric_name, (vals, stds)) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, vals, width, label=metric_name,
                      yerr=stds, capsize=4, color=colors[i], alpha=0.85)

    # 합격선
    ax.axhline(gate_large_f1, color="red", linestyle="--", linewidth=1.2,
               label=f"Gate ({gate_large_f1})")

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("macro F1", fontsize=12)
    ax.set_title("RE:FRIDGE Phase 1 — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info("차트 저장: %s", output_path)


# ──────────────────────────────────────────────────────────────────
# HTML 리포트
# ──────────────────────────────────────────────────────────────────

def save_html_report(
    results: dict[str, CVResult],
    study_results: dict[str, StudyResult],
    data_summary: dict[str, Any],
    error_examples: dict[str, pd.DataFrame],
    per_class_dfs: dict[str, pd.DataFrame],
    output_path: str,
    chart_path: str,
    gate_large_f1: float = 0.85,
) -> None:
    """
    HTML 리포트를 생성한다 (Jupyter 불필요 — 직접 렌더링).

    Args:
        results:        {"model_name": CVResult}
        study_results:  {"model_name": StudyResult}
        data_summary:   data_utils.data_summary() 반환값
        error_examples: {"model_name": DataFrame}
        per_class_dfs:  {"model_name": DataFrame} — per_class_f1_report 반환값
        output_path:    저장 경로
        chart_path:     bar chart 이미지 경로
        gate_large_f1:  합격 기준
    """
    import base64

    # 차트 이미지를 base64로 임베딩
    chart_b64 = ""
    chart_p = Path(chart_path)
    if chart_p.exists():
        with open(chart_p, "rb") as f:
            chart_b64 = base64.b64encode(f.read()).decode()

    display_names = {
        "tfidf_lgbm": "TF-IDF + LightGBM",
        "fasttext":   "FastText + KoNLPy + ML",
        "koelectra":  "KoELECTRA Multi-task",
    }

    # ── 합격 여부 배지 ──
    def gate_badge(r: CVResult) -> str:
        if r.large_f1 >= gate_large_f1:
            return f'<span class="badge pass">PASS ✓ ({r.large_f1:.4f})</span>'
        return f'<span class="badge fail">FAIL ✗ ({r.large_f1:.4f})</span>'

    # ── 비교 테이블 HTML ──
    rows_html = ""
    for name, r in results.items():
        badge = gate_badge(r)
        rows_html += f"""
        <tr>
          <td>{display_names.get(name, name)}</td>
          <td>{badge}</td>
          <td>{r.large_f1:.4f} ± {r.large_f1_std:.4f}</td>
          <td>{r.medium_f1:.4f} ± {r.medium_f1_std:.4f}</td>
          <td>{r.tag_f1:.4f} ± {r.tag_f1_std:.4f}</td>
          <td>{r.large_acc:.4f}</td>
          <td>{r.train_time:.1f}s</td>
          <td>{r.infer_time_ms:.2f}ms</td>
        </tr>"""

    # ── 하이퍼파라미터 섹션 ──
    hp_sections = ""
    for name, sr in study_results.items():
        hp_json = json.dumps(sr.best_params, indent=2, ensure_ascii=False)
        hp_sections += f"""
        <details>
          <summary><b>{display_names.get(name, name)}</b> — best score: {sr.best_score:.4f}</summary>
          <pre class="code">{hp_json}</pre>
        </details>"""

    # ── Optuna 학습 곡선 ──
    optuna_sections = ""
    for name, sr in study_results.items():
        if sr.all_trials_df is not None and not sr.all_trials_df.empty:
            table_html = sr.all_trials_df[["number", "value", "state"]].head(30).to_html(
                index=False, classes="data-table", border=0
            )
            optuna_sections += f"""
            <h4>{display_names.get(name, name)}</h4>
            {table_html}"""

    # ── 오분류 예시 ──
    error_sections = ""
    for name, err_df in error_examples.items():
        if not err_df.empty:
            error_sections += f"""
            <h4>{display_names.get(name, name)}</h4>
            {err_df.to_html(index=False, classes='data-table', border=0)}"""

    # ── 데이터 요약 ──
    data_html = f"""
    <ul>
      <li>총 샘플: <b>{data_summary.get('n_samples', 0)}</b>개</li>
      <li>대분류: <b>{data_summary.get('n_large', 0)}</b>종</li>
      <li>중분류: <b>{data_summary.get('n_medium', 0)}</b>종</li>
      <li>카테고리태그: <b>{data_summary.get('n_tag', 0)}</b>종</li>
      <li>브랜드 수: <b>{data_summary.get('n_brands', 0)}</b>개 (GroupKFold 그룹)</li>
    </ul>"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>RE:FRIDGE Phase 1 — Model Comparison Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 32px; color: #222; background: #f9f9f9; }}
  h1   {{ color: #1a237e; }}
  h2   {{ color: #283593; border-bottom: 2px solid #3f51b5; padding-bottom: 6px; margin-top: 40px; }}
  h3, h4 {{ color: #37474f; }}
  .badge {{ padding: 3px 10px; border-radius: 12px; font-weight: bold; font-size: 0.9em; }}
  .pass  {{ background: #c8e6c9; color: #1b5e20; }}
  .fail  {{ background: #ffcdd2; color: #b71c1c; }}
  table.data-table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  table.data-table th, table.data-table td {{
    border: 1px solid #cfd8dc; padding: 8px 12px; text-align: left; font-size: 0.9em;
  }}
  table.data-table th {{ background: #e8eaf6; font-weight: bold; }}
  table.data-table tr:nth-child(even) {{ background: #f5f5f5; }}
  pre.code {{ background: #263238; color: #eceff1; padding: 16px; border-radius: 6px; overflow-x: auto; font-size: 0.85em; }}
  details {{ margin: 8px 0; }}
  summary {{ cursor: pointer; padding: 6px 0; }}
  .chart {{ max-width: 800px; margin: 20px 0; }}
  .summary-box {{ background: #e8eaf6; border-left: 4px solid #3f51b5; padding: 12px 20px; border-radius: 4px; margin: 16px 0; }}
</style>
</head>
<body>

<h1>RE:FRIDGE Phase 1 — Model Comparison Report</h1>

<h2>1. Executive Summary</h2>
<div class="summary-box">
  <p><b>합격 기준:</b> 대분류 macro_F1 ≥ {gate_large_f1} (5-Fold GroupKFold)</p>
  <p><b>평가 프로토콜:</b> GroupKFold(n=5, group=brand_name) — 브랜드 기준 데이터 누수 방지</p>
</div>
<table class="data-table">
  <thead>
    <tr>
      <th>Model</th><th>합격 여부</th>
      <th>large_F1 (mean±std)</th><th>medium_F1</th><th>tag_F1</th>
      <th>large_Acc</th><th>train</th><th>infer</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>

<h2>2. Comparison Chart</h2>
{'<img class="chart" src="data:image/png;base64,' + chart_b64 + '" alt="comparison chart">' if chart_b64 else '<p>(차트 없음)</p>'}

<h2>3. Best Hyperparameters</h2>
{hp_sections}

<h2>4. Optuna Optimization History (상위 30 trials)</h2>
{optuna_sections}

<h2>5. Data Info</h2>
{data_html}

<h2>6. Error Analysis (Fold 0, 대분류 오분류 상위 20개)</h2>
{error_sections if error_sections else '<p>오류 분석 데이터 없음</p>'}

</body>
</html>
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("HTML 리포트 저장: %s", output_path)


# ──────────────────────────────────────────────────────────────────
# CSV 저장
# ──────────────────────────────────────────────────────────────────

def save_comparison_csv(results: dict[str, CVResult], output_path: str) -> None:
    rows = []
    for name, r in results.items():
        rows.append({
            "model":         name,
            "large_f1":      r.large_f1,
            "large_f1_std":  r.large_f1_std,
            "medium_f1":     r.medium_f1,
            "medium_f1_std": r.medium_f1_std,
            "tag_f1":        r.tag_f1,
            "tag_f1_std":    r.tag_f1_std,
            "large_acc":     r.large_acc,
            "train_time_s":  r.train_time,
            "infer_time_ms": r.infer_time_ms,
        })
    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("비교표 CSV 저장: %s", output_path)
