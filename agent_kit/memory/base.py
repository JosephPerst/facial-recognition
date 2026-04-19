"""Abstract memory types: conversation history and vector recall.

``Memory`` is the short-term working set (the conversation passed to the
model on each turn). ``VectorMemory`` is the long-term, semantic store that
lets you recall snippets across sessions.

Both are intentionally small interfaces so you can roll your own backend
without touching the agent loop.

Example:
    >>> import asyncio
    >>> from agent_kit.memory.conversation import InMemoryConversation
    >>> from agent_kit.core.messages import Message
    >>> async def demo() -> None:
    ...     mem = InMemoryConversation()
    ...     await mem.append(Message.user("hi"))
    ...     assert (await mem.history())[0].content == "hi"
    >>> asyncio.run(demo())
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from agent_kit.core.messages import Message


class Memory(ABC):
    """Conversation-level memory; backs the history fed to the provider."""

    @abstractmethod
    async def append(self, message: Message) -> None:
        """Append a message to the history."""

    @abstractmethod
    async def history(self) -> list[Message]:
        """Return a copy of the full history (ordered, oldest first)."""

    @abstractmethod
    async def clear(self) -> None:
        """Reset the history."""

    async def extend(self, messages: list[Message]) -> None:
        """Append many messages; default implementation loops."""
        for m in messages:
            await self.append(m)


@dataclass
class VectorRecord:
    """A record stored in a vector memory backend.

    Attributes:
        id: Stable identifier.
        text: The original text.
        embedding: Dense vector representation.
        metadata: Arbitrary JSON metadata.
        score: Populated during search with a similarity score in [0, 1].
    """

    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any]
    score: float = 0.0


class VectorMemory(ABC):
    """Long-term semantic memory: embed, store, and query text."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return an embedding for ``text``."""

    @abstractmethod
    async def upsert(
        self,
        *,
        id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a record keyed by ``id``."""

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorRecord]:
        """Return the top ``k`` records most similar to ``query``."""

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete records by id."""


__all__ = ["Memory", "VectorMemory", "VectorRecord"]
