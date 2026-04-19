"""Typed dataset loader for JSONL eval suites.

Example:
    >>> import json, tempfile, pathlib
    >>> from agent_kit.evals.datasets import load_jsonl
    >>> with tempfile.TemporaryDirectory() as d:
    ...     p = pathlib.Path(d) / "cases.jsonl"
    ...     _ = p.write_text(json.dumps(
    ...         {"id": "1", "input": "2+2?", "expected": "4"}
    ...     ) + "\\n")
    ...     ds = load_jsonl(p)
    >>> ds.cases[0].input
    '2+2?'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvalCase(BaseModel):
    """A single eval case.

    Attributes:
        id: Stable ID for the case (used in reports and regression diffs).
        input: The user input sent to the agent.
        expected: Free-form expected answer; judges decide how to interpret.
        tags: Optional labels for filtering or per-tag metrics.
        metadata: Arbitrary JSON the judge / harness can key on.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    input: str
    expected: str = ""
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalDataset(BaseModel):
    """A collection of eval cases loaded from a JSONL file."""

    model_config = ConfigDict(extra="forbid")

    name: str
    cases: list[EvalCase]


def load_jsonl(path: str | Path) -> EvalDataset:
    """Load a JSONL file into an :class:`EvalDataset`.

    Each line must be a JSON object matching the :class:`EvalCase` schema.
    The dataset name is derived from the file stem.

    Example:
        >>> # doctest: +SKIP
        >>> ds = load_jsonl("examples/eval_suites/basic.jsonl")
    """
    path = Path(path)
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_num} invalid JSON: {e}") from e
            cases.append(EvalCase.model_validate(record))
    return EvalDataset(name=path.stem, cases=cases)


__all__ = ["EvalCase", "EvalDataset", "load_jsonl"]
