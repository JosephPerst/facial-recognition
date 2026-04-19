"""Aggregate metrics for an eval run: pass rate, cost, latency percentiles.

Example:
    >>> from agent_kit.evals.metrics import aggregate
    >>> metrics = aggregate([
    ...     {"passed": True, "cost_usd": 0.01, "latency_ms": 100},
    ...     {"passed": False, "cost_usd": 0.02, "latency_ms": 200},
    ... ])
    >>> metrics.pass_rate
    0.5
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunMetrics:
    """Aggregated metrics for a single eval run.

    Attributes:
        total: Total number of cases in the run.
        passed: Number of cases that passed (all judges passed).
        pass_rate: ``passed / total`` in [0, 1].
        mean_score: Mean of per-case averaged judge scores.
        cost_usd: Sum of dollar cost across all cases.
        latency_p50: Median end-to-end latency in ms.
        latency_p95: 95th-percentile latency in ms.
        latency_p99: 99th-percentile latency in ms.
        total_ms: Wall-clock total for all cases combined.
    """

    total: int
    passed: int
    pass_rate: float
    mean_score: float
    cost_usd: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    total_ms: float


def _percentile(sorted_values: list[float], p: float) -> float:
    """Return the ``p``-percentile (0.0-1.0) of a sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = p * (len(sorted_values) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    weight = idx - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def aggregate(records: Iterable[dict[str, Any]]) -> RunMetrics:
    """Aggregate a sequence of per-case result records.

    Each record must contain ``passed`` (bool), ``cost_usd`` (float or None),
    and ``latency_ms`` (float). A ``mean_score`` field is optional.

    Example:
        >>> aggregate([]).pass_rate
        0.0
    """
    records = list(records)
    total = len(records)
    if total == 0:
        return RunMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    passed = sum(1 for r in records if r.get("passed"))
    scores = [float(r.get("mean_score", 1.0 if r.get("passed") else 0.0)) for r in records]
    latencies = sorted(float(r.get("latency_ms", 0.0)) for r in records)
    cost = sum(float(r.get("cost_usd") or 0.0) for r in records)
    return RunMetrics(
        total=total,
        passed=passed,
        pass_rate=passed / total,
        mean_score=sum(scores) / total,
        cost_usd=cost,
        latency_p50=_percentile(latencies, 0.5),
        latency_p95=_percentile(latencies, 0.95),
        latency_p99=_percentile(latencies, 0.99),
        total_ms=sum(latencies),
    )


__all__ = ["RunMetrics", "aggregate"]
