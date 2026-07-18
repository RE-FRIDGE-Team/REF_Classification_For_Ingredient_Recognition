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

    # ★수정(2026-07): 하드코딩 model_order 순회 → results 전체 순회.
    #   기존에는 등록된 3개 이름만 표에 실려, 신규 모델(text_pipeline)이
    #   결과가 있어도 빈 표 + "합격 모델 없음"으로 출력되는 버그가 있었다.
    #   알려진 이름은 선호 순서를 유지하고, 그 외 이름도 뒤에 모두 붙인다.
    preferred = ["text_pipeline", "tfidf_lgbm", "fasttext", "koelectra"]
    display_names = {
        "text_pipeline": "Text Pipeline",
        "tfidf_lgbm": "TF-IDF + LightGBM",
        "fasttext":   "FastText + KoNLPy",
        "koelectra":  "KoELECTRA Multi-task",
    }
    model_order = [n for n in preferred if n in results] + \
                  [n for n in results if n not in preferred]

    passed: list[str] = []
    for name in model_order:
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
        "text_pipeline": "Text\nPipeline",
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
    other_stats: dict[str, dict] | None = None,
    run_label: str = "",
) -> None:
    """
    HTML 리포트 생성 (Jupyter 불필요 — 직접 렌더링).

    ★개편(2026-07 2차):
      - Error Analysis: 전 fold OOF + true/pred_medium + error_type 컬럼 포함.
      - Other Statistics 섹션 신설 (hierarchical_error_stats 산출물 렌더링):
        오분류 카테고리 순위(대/중), 계층 오류 3분면 케이스 목록,
        중분류-태그 정합성 검증, F1 외 보조 지표(Acc/BalancedAcc/MCC/Kappa 등).

    Args:
        results:        {"model_name": CVResult}
        study_results:  {"model_name": StudyResult}
        data_summary:   data_utils.data_summary() 반환값
        error_examples: {"model_name": DataFrame} — evaluate.error_examples (OOF판)
        per_class_dfs:  {"model_name": DataFrame} — per_class_f1_report 반환값
        output_path:    저장 경로
        chart_path:     bar chart 이미지 경로
        gate_large_f1:  합격 기준
        other_stats:    {"model_name": hierarchical_error_stats() 반환 dict}
        run_label:      실행 식별 라벨 (모드_타임스탬프) — 제목·프로토콜 표기용
    """
    import base64

    # 차트 이미지를 base64로 임베딩
    chart_b64 = ""
    chart_p = Path(chart_path)
    if chart_p.exists():
        with open(chart_p, "rb") as f:
            chart_b64 = base64.b64encode(f.read()).decode()

    display_names = {
        "text_pipeline": "Text Pipeline",
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

    # ── 오분류 예시 (OOF·중분류 포함) — 화면 표기는 상위 300행 제한 ──
    _MAX_ROWS = 300

    def _df_html(df: pd.DataFrame, max_rows: int = _MAX_ROWS) -> str:
        note = (f'<p class="note">(전체 {len(df)}행 중 상위 {max_rows}행 표시 — '
                f'전량은 동봉 error_analysis CSV 참조)</p>') if len(df) > max_rows else ""
        return df.head(max_rows).to_html(index=False, classes="data-table", border=0) + note

    error_sections = ""
    for name, err_df in error_examples.items():
        if not err_df.empty:
            type_counts = err_df["error_type"].value_counts().to_dict() \
                if "error_type" in err_df.columns else {}
            counts_line = " · ".join(f"{k}: <b>{v}</b>건" for k, v in type_counts.items())
            error_sections += f"""
            <h4>{display_names.get(name, name)} — 총 오류 {len(err_df)}건</h4>
            <p>{counts_line}</p>
            {_df_html(err_df)}"""

    # ── Other Statistics 섹션 ──
    def _metrics_table(metrics: dict[str, dict]) -> str:
        rows = ""
        for task, m in metrics.items():
            rows += (f"<tr><td><b>{task}</b></td>"
                     + "".join(f"<td>{m[k]:.4f}</td>"
                               for k in ["accuracy", "balanced_accuracy", "macro_f1",
                                          "weighted_f1", "mcc", "cohen_kappa"])
                     + "</tr>")
        return f"""
        <table class="data-table">
          <thead><tr><th>Task</th><th>Accuracy</th><th>Balanced Acc</th>
          <th>macro F1</th><th>weighted F1</th><th>MCC</th><th>Cohen κ</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>"""

    def _case_details(title: str, df: pd.DataFrame, important: bool = False) -> str:
        mark = " ⭐최중요" if important else ""
        body = _df_html(df) if not df.empty else "<p>해당 케이스 없음</p>"
        return f"""
        <details {'open' if important else ''}>
          <summary><b>{title}{mark}</b> — {len(df)}건</summary>
          {body}
        </details>"""

    stats_sections = ""
    for name, st in (other_stats or {}).items():
        tag_df = st["tag_inconsistent"]
        tag_html = (
            "<p>✅ 중분류가 정답인 모든 샘플에서 태그가 올바르게 태깅됨 — "
            "태그 오류는 전부 중분류 오류에서만 유래하므로 별도 카운트 불요.</p>"
            if tag_df.empty else
            f"<p>⚠️ 중분류는 정답인데 태그가 불일치한 케이스 <b>{len(tag_df)}</b>건 발견 "
            "(중분류→태그 매핑 테이블 점검 필요):</p>" + _df_html(tag_df)
        )
        stats_sections += f"""
        <h3>{display_names.get(name, name)}</h3>

        <h4>7-1. 가장 많이 오분류한 카테고리 순위</h4>
        <p><b>대분류 기준</b> (오분류 개수 · 최다 혼동 대상):</p>
        {_df_html(st["rank_large"])}
        <p><b>중분류 기준:</b></p>
        {_df_html(st["rank_medium"])}

        <h4>7-2. 계층 오류 케이스 분해</h4>
        {_case_details("대분류 정답 · 중분류 오답", st["case_large_ok_medium_wrong"], important=True)}
        {_case_details("대분류 오답 → 중분류도 오답", st["case_both_wrong"])}
        {_case_details("대분류 오답 · 중분류 정답 (희귀 케이스)", st["case_large_wrong_medium_ok"])}

        <h4>7-3. 중분류 → 카테고리 태그 정합성 검증</h4>
        {tag_html}

        <h4>7-4. F1 외 보조 분류 지표 (전 fold OOF 기준)</h4>
        {_metrics_table(st["metrics"])}
        <p>계층 동시 정답률 — 대·중 모두 정답: <b>{st["hier_exact_large_medium"]:.4f}</b>
           / 대·중·태그 모두 정답: <b>{st["hier_exact_all"]:.4f}</b></p>"""

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
  .note {{ color: #607d8b; font-size: 0.85em; }}
</style>
</head>
<body>

<h1>RE:FRIDGE Phase 1 — Model Comparison Report</h1>
{f'<p class="note">Run: <b>{run_label}</b></p>' if run_label else ''}

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

<h2>6. Error Analysis (전 fold OOF — 대분류·중분류 오류 전량)</h2>
{error_sections if error_sections else '<p>오류 분석 데이터 없음</p>'}

<h2>7. Other Statistics</h2>
{stats_sections if stats_sections else '<p>통계 데이터 없음</p>'}

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