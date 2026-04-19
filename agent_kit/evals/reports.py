"""Report renderers: console (rich), Markdown, HTML diff, regression compare.

The HTML report uses a single self-contained template — no external assets,
so you can upload the file to S3 or email it without breakage.

Example:
    >>> from agent_kit.evals.harness import EvalRun
    >>> from agent_kit.evals.metrics import RunMetrics
    >>> run = EvalRun(
    ...     dataset="demo",
    ...     started_at="2026-04-19T00:00:00Z",
    ...     finished_at="2026-04-19T00:00:01Z",
    ...     results=[],
    ...     metrics=RunMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0),
    ... )
    >>> "Eval Report" in render_markdown(run)
    True
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from jinja2 import Template
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:  # pragma: no cover
    from agent_kit.evals.harness import EvalRun


_HTML_TEMPLATE = Template(
    """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Eval Report — {{ run.dataset }}</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1100px; margin: 2rem auto; color: #1f2937; padding: 0 1rem; }
  h1 { font-size: 1.6rem; }
  .summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.5rem 0; }
  .card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px;
          padding: 1rem; }
  .card .v { font-size: 1.4rem; font-weight: 600; }
  .card .k { color: #6b7280; font-size: 0.85rem; }
  table { width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.92rem; }
  th, td { border-bottom: 1px solid #e5e7eb; padding: 0.5rem 0.6rem; text-align: left;
           vertical-align: top; }
  tr.pass td:first-child { border-left: 4px solid #16a34a; }
  tr.fail td:first-child { border-left: 4px solid #dc2626; }
  details { margin: 0.25rem 0; }
  .badge { display: inline-block; padding: 0.05rem 0.5rem; border-radius: 99px;
           font-size: 0.75rem; font-weight: 600; }
  .b-pass { background: #dcfce7; color: #166534; }
  .b-fail { background: #fee2e2; color: #991b1b; }
  pre { white-space: pre-wrap; word-break: break-word; background: #f3f4f6;
        padding: 0.6rem; border-radius: 6px; font-size: 0.8rem; }
</style>
</head>
<body>
<h1>Eval Report — {{ run.dataset }}</h1>
<p><small>{{ run.started_at }} → {{ run.finished_at }}</small></p>
<div class="summary">
  <div class="card"><div class="v">{{ '%.1f%%' % (m.pass_rate * 100) }}</div><div class="k">pass rate</div></div>
  <div class="card"><div class="v">{{ m.passed }} / {{ m.total }}</div><div class="k">cases</div></div>
  <div class="card"><div class="v">${{ '%.4f' % m.cost_usd }}</div><div class="k">total cost</div></div>
  <div class="card"><div class="v">{{ '%.0fms' % m.latency_p95 }}</div><div class="k">p95 latency</div></div>
</div>
<table>
  <thead><tr><th>#</th><th>case</th><th>judges</th><th>cost</th><th>latency</th></tr></thead>
  <tbody>
  {% for r in run.results %}
    <tr class="{{ 'pass' if r.passed else 'fail' }}">
      <td>{{ loop.index }}</td>
      <td>
        <strong>{{ r.case_id }}</strong>
        <span class="badge {{ 'b-pass' if r.passed else 'b-fail' }}">
          {{ 'PASS' if r.passed else 'FAIL' }}
        </span>
        <details><summary>input</summary><pre>{{ r.input }}</pre></details>
        <details><summary>output</summary><pre>{{ r.output }}</pre></details>
        {% if r.expected %}
          <details><summary>expected</summary><pre>{{ r.expected }}</pre></details>
        {% endif %}
        {% if r.error %}<pre style="color:#b91c1c">{{ r.error }}</pre>{% endif %}
      </td>
      <td>
        {% for j in r.judge_results %}
          <div><span class="badge {{ 'b-pass' if j.passed else 'b-fail' }}">
            {{ j.judge }} {{ '%.2f' % j.score }}
          </span> {{ j.rationale }}</div>
        {% endfor %}
      </td>
      <td>{{ '$%.4f' % (r.cost_usd or 0.0) }}</td>
      <td>{{ '%.0fms' % r.latency_ms }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
</body>
</html>
"""
)


def render_markdown(run: EvalRun) -> str:
    """Render a Markdown summary of an eval run.

    Example:
        >>> # doctest: +SKIP
        >>> md = render_markdown(run)
    """
    m = run.metrics
    lines: list[str] = [
        f"# Eval Report — {run.dataset}",
        "",
        f"- dataset: **{run.dataset}**",
        f"- started: {run.started_at}",
        f"- finished: {run.finished_at}",
        f"- cases: **{m.passed}/{m.total}** pass ({m.pass_rate * 100:.1f}%)",
        f"- mean score: {m.mean_score:.3f}",
        f"- cost: ${m.cost_usd:.4f}",
        f"- latency p50 / p95 / p99: {m.latency_p50:.0f}ms / {m.latency_p95:.0f}ms / {m.latency_p99:.0f}ms",
        "",
        "| # | case | result | mean score | cost | latency |",
        "|---|------|--------|-----------:|-----:|--------:|",
    ]
    for i, r in enumerate(run.results, 1):
        badge = "✅" if r.passed else "❌"
        cost = f"${r.cost_usd:.4f}" if r.cost_usd is not None else "n/a"
        lines.append(
            f"| {i} | `{r.case_id}` | {badge} | {r.mean_score:.3f} | {cost} "
            f"| {r.latency_ms:.0f}ms |"
        )
    lines.append("")
    for r in run.results:
        if r.passed and not r.error:
            continue
        lines.append(f"## ❌ {r.case_id}")
        lines.append("")
        lines.append(f"**input:** {r.input}")
        lines.append("")
        if r.expected:
            lines.append(f"**expected:** {r.expected}")
            lines.append("")
        lines.append(f"**output:** {r.output or '(empty)'}")
        lines.append("")
        if r.error:
            lines.append(f"**error:** `{r.error}`")
            lines.append("")
        for j in r.judge_results:
            lines.append(f"- **{j.judge}**: score {j.score:.2f} — {j.rationale or 'no rationale'}")
        lines.append("")
    return "\n".join(lines)


def render_html(run: EvalRun) -> str:
    """Render a self-contained HTML report for an eval run."""
    return _HTML_TEMPLATE.render(run=run, m=run.metrics)


def render_console(run: EvalRun, console: Console | None = None) -> None:
    """Print a rich summary table to the console."""
    console = console or Console()
    m = run.metrics
    console.rule(f"Eval: {run.dataset}")
    table = Table(show_header=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("case")
    table.add_column("result")
    table.add_column("score", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("latency", justify="right")
    for i, r in enumerate(run.results, 1):
        badge = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        cost = f"${r.cost_usd:.4f}" if r.cost_usd is not None else "n/a"
        table.add_row(
            str(i),
            r.case_id,
            badge,
            f"{r.mean_score:.3f}",
            cost,
            f"{r.latency_ms:.0f}ms",
        )
    console.print(table)
    console.print(
        f"[bold]{m.passed}/{m.total}[/bold] passed "
        f"([bold]{m.pass_rate * 100:.1f}%[/bold]) — "
        f"cost ${m.cost_usd:.4f} — p95 {m.latency_p95:.0f}ms"
    )


def compare_to_baseline(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Compare two serialised runs and return a regression diff.

    The returned dict has two keys:

    * ``metrics``: per-metric ``{baseline, current, delta}`` entries.
    * ``regressed_cases``: list of cases that previously passed and now fail.

    Example:
        >>> base = {"metrics": {"pass_rate": 1.0}, "results": [
        ...     {"case_id": "a", "passed": True}]}
        >>> curr = {"metrics": {"pass_rate": 0.0}, "results": [
        ...     {"case_id": "a", "passed": False}]}
        >>> diff = compare_to_baseline(base, curr)
        >>> diff["regressed_cases"][0]["case_id"]
        'a'
    """
    metrics: dict[str, dict[str, float]] = {}
    base_metrics = baseline.get("metrics", {})
    curr_metrics = current.get("metrics", {})
    for key in sorted(set(base_metrics) | set(curr_metrics)):
        try:
            b = float(base_metrics.get(key, 0.0))
            c = float(curr_metrics.get(key, 0.0))
        except (TypeError, ValueError):
            continue
        metrics[key] = {"baseline": b, "current": c, "delta": c - b}

    base_by_id = {r["case_id"]: r for r in baseline.get("results", [])}
    regressed: list[dict[str, Any]] = []
    for r in current.get("results", []):
        base = base_by_id.get(r["case_id"])
        if base and base.get("passed") and not r.get("passed"):
            regressed.append(
                {
                    "case_id": r["case_id"],
                    "input": r.get("input"),
                    "prev_output": base.get("output"),
                    "new_output": r.get("output"),
                }
            )
    return {"metrics": metrics, "regressed_cases": regressed}


def _metrics_dict(metrics: Any) -> dict[str, Any]:
    """Coerce a ``RunMetrics`` dataclass (or dict) to a plain dict."""
    return metrics if isinstance(metrics, dict) else asdict(metrics)


__all__ = [
    "compare_to_baseline",
    "render_console",
    "render_html",
    "render_markdown",
]
