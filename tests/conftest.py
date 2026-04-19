"""Shared pytest fixtures: a fake provider that never hits the network."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from agent_kit.core.config import ProviderName, Settings
from agent_kit.core.messages import Message, Role, ToolCall, Usage
from agent_kit.core.provider import Provider, StreamEvent


class FakeProvider(Provider):
    """Deterministic provider for tests.

    Scripted responses are a list of ``(text, tool_calls)`` tuples; each call
    to ``complete`` / ``stream`` consumes the next one. When exhausted, the
    provider returns an empty assistant message so loops terminate cleanly.
    """

    name: ProviderName = "anthropic"

    def __init__(
        self,
        settings: Settings | None = None,
        script: list[tuple[str, list[ToolCall]]] | None = None,
    ) -> None:
        """Create the fake provider with an optional pre-scripted response queue."""
        super().__init__(settings or Settings())
        self._script = list(script or [])
        self.calls: list[dict[str, Any]] = []

    def _next(self) -> tuple[str, list[ToolCall]]:
        if not self._script:
            return "", []
        return self._script.pop(0)

    async def complete(
        self,
        *,
        system: str | None,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[Message, Usage]:
        """Return the next scripted response as a single Message."""
        self.calls.append(
            {
                "system": system,
                "messages": list(messages),
                "tools": tools,
                "model": model,
            }
        )
        text, tool_calls = self._next()
        usage = Usage(input_tokens=10, output_tokens=20, cost_usd=0.001)
        return (
            Message(role=Role.ASSISTANT, content=text, tool_calls=tool_calls),
            usage,
        )

    async def stream(
        self,
        *,
        system: str | None,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield the next scripted response as a stream of events."""
        self.calls.append(
            {
                "system": system,
                "messages": list(messages),
                "tools": tools,
                "model": model,
                "stream": True,
            }
        )
        text, tool_calls = self._next()
        for ch in text:
            yield StreamEvent(kind="text", text=ch)
        for tc in tool_calls:
            yield StreamEvent(kind="tool_call", tool_call=tc)
        yield StreamEvent(
            kind="message_stop",
            usage=Usage(input_tokens=10, output_tokens=20, cost_usd=0.001),
        )


@pytest.fixture
def fake_provider() -> FakeProvider:
    """Yield a pre-built FakeProvider."""
    return FakeProvider()


@pytest.fixture
def settings(tmp_path: Any) -> Settings:
    """Return a Settings instance that isolates file paths to ``tmp_path``."""
    return Settings(
        memory_backend="inmemory",
        sqlite_path=tmp_path / "db.sqlite",
        chroma_path=tmp_path / "chroma",
        trace_file=tmp_path / "traces.jsonl",
        max_iterations=4,
        max_cost_usd=100.0,
        retry_max_attempts=1,
    )
