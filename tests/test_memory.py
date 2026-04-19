"""Tests for conversation backends and the auto-summariser wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from agent_kit.core.messages import Message, Role
from agent_kit.memory.conversation import InMemoryConversation, SQLiteConversation
from agent_kit.memory.summarizer import AutoSummarizingMemory, Summarizer


async def test_inmemory_append_history() -> None:
    m = InMemoryConversation()
    await m.append(Message.user("hi"))
    await m.append(Message.assistant("hello"))
    history = await m.history()
    assert [msg.content for msg in history] == ["hi", "hello"]


async def test_inmemory_clear() -> None:
    m = InMemoryConversation()
    await m.append(Message.user("one"))
    await m.clear()
    assert await m.history() == []


async def test_sqlite_round_trip(tmp_path: Path) -> None:
    m = SQLiteConversation(tmp_path / "c.db", session_id="s1")
    await m.append(Message.user("hi"))
    await m.append(Message.assistant("hello"))
    history = await m.history()
    assert [h.content for h in history] == ["hi", "hello"]
    m.close()


async def test_sqlite_isolates_sessions(tmp_path: Path) -> None:
    a = SQLiteConversation(tmp_path / "c.db", session_id="a")
    b = SQLiteConversation(tmp_path / "c.db", session_id="b")
    await a.append(Message.user("aaa"))
    await b.append(Message.user("bbb"))
    assert [m.content for m in await a.history()] == ["aaa"]
    assert [m.content for m in await b.history()] == ["bbb"]
    a.close()
    b.close()


class _StubSummarizer(Summarizer):
    async def summarize(self, messages: Sequence[Message]) -> str:
        return f"summary of {len(messages)} messages"


async def test_auto_summarizer_rolls_up() -> None:
    base = InMemoryConversation()
    mem = AutoSummarizingMemory(
        base,
        summarizer=_StubSummarizer(),
        max_tokens=1,  # tiny budget so it fires instantly
        keep_last=1,
    )
    await mem.append(Message.user("a" * 200))
    await mem.append(Message.user("b" * 200))
    await mem.append(Message.user("c" * 200))
    history = await mem.history()
    assert any(m.role == Role.SYSTEM and "summary of" in m.content for m in history)
    # The last message should still be present verbatim.
    assert any("c" * 200 in m.content for m in history)
