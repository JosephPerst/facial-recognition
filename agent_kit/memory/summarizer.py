"""Auto-summarising conversation memory.

Wraps a ``Memory`` backend and replaces the oldest window of messages with
an LLM-generated summary once an approximate token threshold is exceeded.

Example:
    >>> import asyncio
    >>> from agent_kit.memory.conversation import InMemoryConversation
    >>> from agent_kit.memory.summarizer import AutoSummarizingMemory, Summarizer
    >>> from agent_kit.core.messages import Message
    >>> class DummySummarizer(Summarizer):
    ...     async def summarize(self, messages):  # noqa: D401
    ...         return "summary of " + str(len(messages)) + " msgs"
    >>> async def demo() -> None:
    ...     base = InMemoryConversation()
    ...     mem = AutoSummarizingMemory(
    ...         base, summarizer=DummySummarizer(), max_tokens=1, keep_last=0
    ...     )
    ...     await mem.append(Message.user("a" * 20))
    ...     await mem.append(Message.user("b" * 20))
    ...     history = await mem.history()
    ...     assert any("summary of" in m.content for m in history)
    >>> asyncio.run(demo())
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from agent_kit.core.messages import Message, Role
from agent_kit.memory.base import Memory

if TYPE_CHECKING:  # pragma: no cover
    from agent_kit.core.provider import Provider


def _approx_tokens(messages: Sequence[Message]) -> int:
    """Approximate token count using the 4-chars-per-token heuristic."""
    total = 0
    for m in messages:
        total += len(m.content) // 4
        for tc in m.tool_calls:
            total += (len(tc.name) + len(str(tc.arguments))) // 4
        for tr in m.tool_results:
            total += len(tr.content) // 4
    return total


class Summarizer(ABC):
    """Produces a single summary string from a window of messages."""

    @abstractmethod
    async def summarize(self, messages: Sequence[Message]) -> str:
        """Return a concise summary of ``messages``."""


class LLMSummarizer(Summarizer):
    """LLM-backed summariser using any Agent Kit provider."""

    _PROMPT = (
        "You are a summarisation engine. Produce a terse bullet-point summary "
        "of the conversation so far that preserves: user goals, decisions, "
        "tool outcomes, and any open questions. Do NOT add commentary."
    )

    def __init__(self, provider: Provider, model: str | None = None) -> None:
        """Summarise using ``provider`` and the given ``model``."""
        self.provider = provider
        self.model = model

    async def summarize(self, messages: Sequence[Message]) -> str:
        """Summarise ``messages`` via the LLM provider."""
        body = "\n".join(
            f"{m.role.value.upper()}: {m.content}" for m in messages if m.content.strip()
        )
        resp, _ = await self.provider.complete(
            system=self._PROMPT,
            messages=[Message.user(body)],
            tools=None,
            model=self.model,
            temperature=0.2,
            max_tokens=800,
        )
        return resp.content


class AutoSummarizingMemory(Memory):
    """Memory wrapper that rolls up old turns once token budget is exceeded."""

    def __init__(
        self,
        inner: Memory,
        *,
        summarizer: Summarizer,
        max_tokens: int = 6000,
        keep_last: int = 8,
    ) -> None:
        """Wrap ``inner`` with auto-summarisation.

        Args:
            inner: The underlying memory backend.
            summarizer: Produces the rollup summary.
            max_tokens: Approx token budget at which a rollup is triggered.
            keep_last: Most recent messages to preserve verbatim.
        """
        self._inner = inner
        self._summarizer = summarizer
        self._max_tokens = max_tokens
        self._keep_last = keep_last

    async def append(self, message: Message) -> None:
        """Append and maybe summarise."""
        await self._inner.append(message)
        await self._maybe_summarize()

    async def history(self) -> list[Message]:
        """Return the (possibly summarised) history."""
        return await self._inner.history()

    async def clear(self) -> None:
        """Reset the underlying memory."""
        await self._inner.clear()

    async def _maybe_summarize(self) -> None:
        messages = await self._inner.history()
        if _approx_tokens(messages) <= self._max_tokens:
            return
        if len(messages) <= self._keep_last + 1:
            return
        # Preserve any existing leading system message.
        head: list[Message] = []
        body = list(messages)
        if body and body[0].role == Role.SYSTEM:
            head.append(body.pop(0))
        to_summarize = body[: max(0, len(body) - self._keep_last)]
        tail = body[len(to_summarize) :]
        if not to_summarize:
            return
        summary_text = await self._summarizer.summarize(to_summarize)
        summary_msg = Message(
            role=Role.SYSTEM,
            content=f"[summary of prior conversation]\n{summary_text}",
            metadata={"auto_summary": True, "replaced": len(to_summarize)},
        )
        await self._inner.clear()
        await self._inner.extend(head + [summary_msg, *tail])


__all__ = ["AutoSummarizingMemory", "LLMSummarizer", "Summarizer"]
