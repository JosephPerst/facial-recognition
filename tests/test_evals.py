"""Tests for the eval harness, judges, datasets, metrics, and reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agent_kit import Agent
from agent_kit.core.config import Settings
from agent_kit.evals.datasets import EvalCase, load_jsonl
from agent_kit.evals.harness import EvalHarness, EvalRun, EvalRunResult
from agent_kit.evals.judges import JudgeResult, StringMatchJudge
from agent_kit.evals.metrics import RunMetrics, aggregate
from agent_kit.evals.reports import (
    compare_to_baseline,
    render_html,
    render_markdown,
)

from tests.conftest import FakeProvider


def test_load_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "cases.jsonl"
    p.write_text(
        json.dumps({"id": "1", "input": "hi", "expected": "hi"})
        + "\n"
        + json.dumps({"id": "2", "input": "bye", "expected": "bye"})
        + "\n"
    )
    ds = load_jsonl(p)
    assert len(ds.cases) == 2
    assert ds.cases[0].id == "1"


def test_load_jsonl_invalid_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text("not json\n")
    with pytest.raises(ValueError):
        load_jsonl(p)


async def test_string_match_judge() -> None:
    j = StringMatchJudge()
    case = EvalCase(id="1", input="q", expected="hello")
    good = await j.score(case, "it says hello world")
    bad = await j.score(case, "goodbye")
    assert good.passed and good.score == 1.0
    assert not bad.passed and bad.score == 0.0


def test_aggregate_empty() -> None:
    m = aggregate([])
    assert m.total == 0
    assert m.pass_rate == 0.0


def test_aggregate_metrics() -> None:
    m = aggregate(
        [
            {"passed": True, "cost_usd": 0.01, "latency_ms": 100, "mean_score": 1.0},
            {"passed": False, "cost_usd": 0.02, "latency_ms": 300, "mean_score": 0.2},
        ]
    )
    assert m.total == 2
    assert m.passed == 1
    assert m.pass_rate == 0.5
    assert round(m.cost_usd, 4) == 0.03
    assert m.latency_p50 == 200
    assert m.mean_score == 0.6


async def test_harness_runs(tmp_path: Path, settings: Settings) -> None:
    cases_file = tmp_path / "ds.jsonl"
    cases_file.write_text(
        json.dumps({"id": "a", "input": "say hi", "expected": "hi"})
        + "\n"
        + json.dumps({"id": "b", "input": "say bye", "expected": "bye"})
        + "\n"
    )
    ds = load_jsonl(cases_file)

    async def factory(case: EvalCase) -> Agent:
        provider = FakeProvider(settings, script=[(case.expected, [])])
        return Agent(settings=settings, provider=provider)

    harness = EvalHarness(
        agent_factory=factory,
        judges=[StringMatchJudge()],
        concurrency=2,
    )
    run = await harness.run(ds)
    assert run.metrics.pass_rate == 1.0
    assert len(run.results) == 2


def test_render_markdown_contains_badges() -> None:
    run = EvalRun(
        dataset="t",
        started_at="s",
        finished_at="f",
        results=[
            EvalRunResult(
                case_id="a",
                input="?",
                output="!",
                expected="!",
                passed=True,
                mean_score=1.0,
                judge_results=[JudgeResult(judge="x", score=1.0, passed=True, rationale="ok")],
                cost_usd=0.001,
                latency_ms=50.0,
                iterations=1,
            ),
            EvalRunResult(
                case_id="b",
                input="?",
                output="wrong",
                expected="right",
                passed=False,
                mean_score=0.0,
                judge_results=[JudgeResult(judge="x", score=0.0, passed=False, rationale="no")],
                cost_usd=0.002,
                latency_ms=80.0,
                iterations=1,
            ),
        ],
        metrics=RunMetrics(2, 1, 0.5, 0.5, 0.003, 50, 80, 80, 130),
    )
    md = render_markdown(run)
    assert "Eval Report" in md
    assert "## ❌ b" in md
    html = render_html(run)
    assert "Eval Report" in html


def test_compare_to_baseline_detects_regression() -> None:
    base = {
        "metrics": {"pass_rate": 1.0, "cost_usd": 0.01},
        "results": [
            {"case_id": "a", "passed": True, "output": "ok"},
            {"case_id": "b", "passed": True, "output": "ok"},
        ],
    }
    curr = {
        "metrics": {"pass_rate": 0.5, "cost_usd": 0.02},
        "results": [
            {"case_id": "a", "passed": True, "output": "ok"},
            {"case_id": "b", "passed": False, "output": "oops"},
        ],
    }
    diff = compare_to_baseline(base, curr)
    assert diff["metrics"]["pass_rate"]["delta"] == -0.5
    assert [c["case_id"] for c in diff["regressed_cases"]] == ["b"]
