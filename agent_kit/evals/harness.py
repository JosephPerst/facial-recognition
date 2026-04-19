"""Parallel eval runner with judges, metrics, and CLI entrypoint.

Usage (CLI):

    uv run -m agent_kit.evals.harness run path/to/dataset.jsonl
    uv run -m agent_kit.evals.harness compare baseline.json latest.json

Usage (library):

    >>> # doctest: +SKIP
    >>> harness = EvalHarness(agent_factory=make_agent, judges=[StringMatchJudge()])
    >>> run = await harness.run(dataset)
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.table import Table

from agent_kit.core.agent import Agent
from agent_kit.core.config import Settings
from agent_kit.core.provider import get_provider
from agent_kit.evals.datasets import EvalCase, EvalDataset, load_jsonl
from agent_kit.evals.judges import (
    CorrectnessJudge,
    HelpfulnessJudge,
    Judge,
    JudgeResult,
    SafetyJudge,
    StringMatchJudge,
)
from agent_kit.evals.metrics import RunMetrics, aggregate
from agent_kit.evals.reports import (
    compare_to_baseline,
    render_console,
    render_html,
    render_markdown,
)

AgentFactory = Callable[[EvalCase], Awaitable[Agent]]


class EvalRunResult(BaseModel):
    """Per-case outcome of an eval run."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    input: str
    output: str
    expected: str
    passed: bool
    mean_score: float
    judge_results: list[JudgeResult]
    cost_usd: float | None
    latency_ms: float
    iterations: int
    error: str | None = None
    tags: list[str] = Field(default_factory=list)


@dataclass
class EvalRun:
    """Full result of running a dataset against an agent."""

    dataset: str
    started_at: str
    finished_at: str
    results: list[EvalRunResult]
    metrics: RunMetrics
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "dataset": self.dataset,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "results": [r.model_dump() for r in self.results],
            "metrics": asdict(self.metrics),
            "config": self.config,
        }

    def save(self, path: str | Path) -> Path:
        """Write the run to JSON and return the path written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path


class EvalHarness:
    """Runs a dataset against an agent factory, scoring with judges."""

    def __init__(
        self,
        *,
        agent_factory: AgentFactory,
        judges: Sequence[Judge],
        concurrency: int = 4,
    ) -> None:
        """Build the harness.

        Args:
            agent_factory: Async callable returning a fresh ``Agent`` per case.
            judges: Judges applied to each case's output.
            concurrency: Max number of cases evaluated in parallel.
        """
        if not judges:
            raise ValueError("at least one judge is required")
        self.agent_factory = agent_factory
        self.judges = list(judges)
        self.concurrency = concurrency

    async def run(
        self,
        dataset: EvalDataset,
        *,
        config: dict[str, Any] | None = None,
    ) -> EvalRun:
        """Evaluate every case in ``dataset`` and return a consolidated run."""
        started = datetime.now(UTC).isoformat()
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_one(case: EvalCase) -> EvalRunResult:
            async with semaphore:
                return await self._run_case(case)

        results = await asyncio.gather(*(run_one(c) for c in dataset.cases))
        finished = datetime.now(UTC).isoformat()
        metrics = aggregate(
            [
                {
                    "passed": r.passed,
                    "cost_usd": r.cost_usd,
                    "latency_ms": r.latency_ms,
                    "mean_score": r.mean_score,
                }
                for r in results
            ]
        )
        return EvalRun(
            dataset=dataset.name,
            started_at=started,
            finished_at=finished,
            results=list(results),
            metrics=metrics,
            config=config or {},
        )

    async def _run_case(self, case: EvalCase) -> EvalRunResult:
        """Run a single case end-to-end: agent call + all judges."""
        start = time.perf_counter()
        output = ""
        error: str | None = None
        cost: float | None = None
        iterations = 0
        try:
            agent = await self.agent_factory(case)
            result = await agent.run(case.input)
            output = result.output_text
            cost = result.usage.cost_usd
            iterations = result.iterations
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - start) * 1000

        judge_results: list[JudgeResult] = []
        for judge in self.judges:
            try:
                judge_results.append(await judge.score(case, output))
            except Exception as exc:
                judge_results.append(
                    JudgeResult(
                        judge=judge.name,
                        score=0.0,
                        passed=False,
                        rationale=f"judge raised: {exc}",
                    )
                )

        mean_score = (
            sum(j.score for j in judge_results) / len(judge_results) if judge_results else 0.0
        )
        passed = error is None and all(j.passed for j in judge_results)
        return EvalRunResult(
            case_id=case.id,
            input=case.input,
            output=output,
            expected=case.expected,
            passed=passed,
            mean_score=mean_score,
            judge_results=judge_results,
            cost_usd=cost,
            latency_ms=latency_ms,
            iterations=iterations,
            error=error,
            tags=case.tags,
        )


# ---------- Default factory helpers ----------


def default_agent_factory(
    *,
    system: str | None = None,
    settings: Settings | None = None,
) -> AgentFactory:
    """Return an agent factory that builds a fresh ``Agent`` per case.

    Example:
        >>> f = default_agent_factory(system="Be concise.")
        >>> callable(f)
        True
    """
    s = settings or Settings()
    provider = get_provider(s)

    async def factory(case: EvalCase) -> Agent:
        return Agent(settings=s, system=system, provider=provider)

    return factory


def default_judges(settings: Settings | None = None) -> list[Judge]:
    """Build the shipped trio of LLM judges + a cheap string-match smoke test.

    Example:
        >>> judges = default_judges(Settings())
        >>> {j.name for j in judges} == {"correctness", "helpfulness", "safety", "string_match"}
        True
    """
    s = settings or Settings()
    # Judges use a (potentially) different model for evaluation robustness.
    judge_settings = s.model_copy(update={"model": s.judge_model})
    provider = get_provider(judge_settings)
    return [
        StringMatchJudge(),
        CorrectnessJudge(provider, model=s.judge_model, threshold=0.7),
        HelpfulnessJudge(provider, model=s.judge_model, threshold=0.5),
        SafetyJudge(provider, model=s.judge_model, threshold=0.9),
    ]


# ---------- CLI ----------

app = typer.Typer(add_completion=False, help="Agent Kit evaluation harness.")
_console = Console()


@app.command("run")
def run_cmd(
    dataset: Path = typer.Argument(..., help="Path to a JSONL dataset."),
    system: str = typer.Option(
        "You are a helpful, precise assistant.",
        help="System prompt passed to each agent.",
    ),
    output_dir: Path = typer.Option(
        Path("eval_runs"),
        help="Where to write JSON / markdown / HTML reports.",
    ),
    concurrency: int = typer.Option(4, min=1, max=32),
    baseline: Path | None = typer.Option(None, help="Compare against this previous run JSON."),
    string_match_only: bool = typer.Option(
        False, help="Skip LLM judges — use only the deterministic StringMatchJudge."
    ),
) -> None:
    """Run the eval harness on ``DATASET`` and write reports."""
    settings = Settings()
    judges: list[Judge] = [StringMatchJudge()] if string_match_only else default_judges(settings)
    harness = EvalHarness(
        agent_factory=default_agent_factory(system=system, settings=settings),
        judges=judges,
        concurrency=concurrency,
    )
    ds = load_jsonl(dataset)
    _console.print(f"[bold]Running {len(ds.cases)} cases from {ds.name}…[/bold]")
    run = asyncio.run(
        harness.run(
            ds,
            config={
                "provider": settings.provider,
                "model": settings.model,
                "judge_model": settings.judge_model,
                "concurrency": concurrency,
            },
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = run.save(output_dir / f"{ds.name}.json")
    md_path = output_dir / f"{ds.name}.md"
    html_path = output_dir / f"{ds.name}.html"
    md_path.write_text(render_markdown(run))
    html_path.write_text(render_html(run))

    render_console(run, _console)

    if baseline and baseline.exists():
        previous = json.loads(baseline.read_text())
        diff = compare_to_baseline(previous, run.to_dict())
        _render_regression(diff)

    _console.print(f"\n[green]wrote[/green] {json_path}")
    _console.print(f"[green]wrote[/green] {md_path}")
    _console.print(f"[green]wrote[/green] {html_path}")


@app.command("compare")
def compare_cmd(
    baseline: Path = typer.Argument(...),
    current: Path = typer.Argument(...),
) -> None:
    """Print a regression table comparing two run JSON files."""
    base = json.loads(baseline.read_text())
    curr = json.loads(current.read_text())
    _render_regression(compare_to_baseline(base, curr))


def _render_regression(diff: dict[str, Any]) -> None:
    """Pretty-print a regression diff table."""
    table = Table(title="Regression")
    table.add_column("metric")
    table.add_column("baseline", justify="right")
    table.add_column("current", justify="right")
    table.add_column("delta", justify="right")
    for key, entry in diff["metrics"].items():
        delta = entry["delta"]
        style = "red" if delta < 0 else ("green" if delta > 0 else "")
        table.add_row(
            key,
            f"{entry['baseline']:.4f}",
            f"{entry['current']:.4f}",
            f"[{style}]{delta:+.4f}[/{style}]" if style else f"{delta:+.4f}",
        )
    _console.print(table)
    regressed = diff["regressed_cases"]
    if regressed:
        _console.print(
            f"[bold red]{len(regressed)} case(s) regressed[/bold red]: "
            + ", ".join(c["case_id"] for c in regressed[:10])
        )
    else:
        _console.print("[bold green]no case regressed[/bold green]")


def cli() -> None:
    """Entry point exposed via ``pyproject.toml`` scripts."""
    app()


if __name__ == "__main__":  # pragma: no cover - CLI
    cli()


__all__ = [
    "AgentFactory",
    "EvalHarness",
    "EvalRun",
    "EvalRunResult",
    "cli",
    "default_agent_factory",
    "default_judges",
]
