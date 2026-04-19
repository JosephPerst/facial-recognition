"""Eval harness: golden datasets, LLM judges, regression detection, reports."""

from __future__ import annotations

from agent_kit.evals.datasets import EvalCase, EvalDataset, load_jsonl
from agent_kit.evals.harness import EvalHarness, EvalRun, EvalRunResult
from agent_kit.evals.judges import (
    CorrectnessJudge,
    HelpfulnessJudge,
    Judge,
    JudgeResult,
    SafetyJudge,
)
from agent_kit.evals.metrics import RunMetrics, aggregate
from agent_kit.evals.reports import (
    compare_to_baseline,
    render_console,
    render_html,
    render_markdown,
)

__all__ = [
    "CorrectnessJudge",
    "EvalCase",
    "EvalDataset",
    "EvalHarness",
    "EvalRun",
    "EvalRunResult",
    "HelpfulnessJudge",
    "Judge",
    "JudgeResult",
    "RunMetrics",
    "SafetyJudge",
    "aggregate",
    "compare_to_baseline",
    "load_jsonl",
    "render_console",
    "render_html",
    "render_markdown",
]
