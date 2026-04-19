"""Per-provider pricing tables and a thread-safe cost ledger.

Pricing is quoted per-million tokens (USD) and is intentionally conservative
— users should override when new models launch or contract rates change.

Example:
    >>> from agent_kit.core.messages import Usage
    >>> from agent_kit.observability.cost import price_usage
    >>> u = Usage(input_tokens=1000, output_tokens=500)
    >>> priced = price_usage("anthropic", "claude-sonnet-4-6", u)
    >>> priced.cost_usd is not None
    True
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from agent_kit.core.messages import Usage


@dataclass(frozen=True)
class Price:
    """USD cost per million tokens for a given model."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, usage: Usage) -> float:
        """Return the cost in USD for a given ``Usage`` object."""
        return (
            usage.input_tokens * self.input_per_mtok / 1_000_000
            + usage.output_tokens * self.output_per_mtok / 1_000_000
        )


# Conservative published prices as of 2026-Q1. Update as needed; unknown
# models fall back to ``_DEFAULT`` so agents never silently bill $0.
_PRICES: dict[tuple[str, str], Price] = {
    ("anthropic", "claude-opus-4-7"): Price(15.0, 75.0),
    ("anthropic", "claude-sonnet-4-6"): Price(3.0, 15.0),
    ("anthropic", "claude-haiku-4-5"): Price(1.0, 5.0),
    ("anthropic", "claude-haiku-4-5-20251001"): Price(1.0, 5.0),
    ("openai", "gpt-5"): Price(5.0, 20.0),
    ("openai", "gpt-5-mini"): Price(0.5, 2.0),
    ("openai", "gpt-4o"): Price(2.5, 10.0),
    ("openai", "gpt-4o-mini"): Price(0.15, 0.6),
    ("openai", "o3"): Price(10.0, 40.0),
    ("openai", "o4-mini"): Price(1.1, 4.4),
}
_DEFAULT = Price(3.0, 15.0)


def pricing_for(provider: str, model: str) -> Price:
    """Return the ``Price`` for a (provider, model) pair with a safe default."""
    return _PRICES.get((provider, model), _DEFAULT)


def price_usage(provider: str, model: str, usage: Usage) -> Usage:
    """Return a new ``Usage`` with ``cost_usd`` populated.

    Example:
        >>> from agent_kit.core.messages import Usage
        >>> u = price_usage("openai", "gpt-4o-mini", Usage(100, 100))
        >>> round(u.cost_usd or 0, 6)
        7.5e-05
    """
    if provider == "ollama":  # local model — no dollar cost
        return Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=0.0,
        )
    price = pricing_for(provider, model)
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=price.cost(usage),
    )


@dataclass
class CostLedger:
    """Thread-safe ledger tallying usage across an agent run.

    Example:
        >>> ledger = CostLedger()
        >>> ledger.record(Usage(10, 10, cost_usd=0.001))
        >>> ledger.total.total_tokens
        20
    """

    total: Usage = field(default_factory=Usage)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, usage: Usage) -> None:
        """Add ``usage`` to the running totals in a thread-safe way."""
        with self._lock:
            self.total = self.total + usage

    def reset(self) -> None:
        """Clear the ledger."""
        with self._lock:
            self.total = Usage()


__all__ = ["CostLedger", "Price", "price_usage", "pricing_for"]
