"""In-memory and SQLite conversation-history backends.

``InMemoryConversation`` is the zero-dependency default. ``SQLiteConversation``
persists history across process restarts and is safe for local dev or a
single-host deployment.

Example:
    >>> import asyncio, tempfile
    >>> from pathlib import Path
    >>> from agent_kit.core.messages import Message
    >>> from agent_kit.memory.conversation import SQLiteConversation
    >>> async def demo() -> None:
    ...     with tempfile.TemporaryDirectory() as d:
    ...         m = SQLiteConversation(Path(d) / "c.db", session_id="s1")
    ...         await m.append(Message.user("hi"))
    ...         assert len(await m.history()) == 1
    >>> asyncio.run(demo())
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agent_kit.core.messages import Message, Role, ToolCall, ToolResult
from agent_kit.memory.base import Memory


class InMemoryConversation(Memory):
    """List-backed conversation history; loses contents on process exit."""

    def __init__(self, messages: Iterable[Message] | None = None) -> None:
        """Optionally seed with an initial list of messages."""
        self._messages: list[Message] = list(messages or [])
        self._lock = asyncio.Lock()

    async def append(self, message: Message) -> None:
        """Append a message to the in-memory list."""
        async with self._lock:
            self._messages.append(message)

    async def history(self) -> list[Message]:
        """Return a shallow copy of the current message list."""
        async with self._lock:
            return list(self._messages)

    async def clear(self) -> None:
        """Reset the history."""
        async with self._lock:
            self._messages.clear()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_calls TEXT NOT NULL,
    tool_results TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""


class SQLiteConversation(Memory):
    """Conversation memory persisted to a SQLite database file."""

    def __init__(self, path: str | Path, session_id: str = "default") -> None:
        """Open (or create) a SQLite file and initialise the schema."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self._lock = asyncio.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    async def append(self, message: Message) -> None:
        """Serialise and persist ``message``."""
        async with self._lock:
            await asyncio.to_thread(self._append_sync, message)

    def _append_sync(self, message: Message) -> None:
        self._conn.execute(
            """
            INSERT INTO messages (
                session_id, message_id, role, content, tool_calls,
                tool_results, created_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.session_id,
                message.id,
                message.role.value,
                message.content,
                json.dumps([tc.model_dump() for tc in message.tool_calls]),
                json.dumps([tr.model_dump() for tr in message.tool_results]),
                message.created_at.isoformat(),
                json.dumps(message.metadata, default=str),
            ),
        )
        self._conn.commit()

    async def history(self) -> list[Message]:
        """Return all messages for the current session in creation order."""
        async with self._lock:
            rows = await asyncio.to_thread(self._history_sync)
        return [_row_to_message(r) for r in rows]

    def _history_sync(self) -> list[dict[str, Any]]:
        cursor = self._conn.execute(
            """
            SELECT message_id, role, content, tool_calls, tool_results,
                   created_at, metadata
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (self.session_id,),
        )
        cols = [c[0] for c in cursor.description]
        return [dict(zip(cols, row, strict=False)) for row in cursor.fetchall()]

    async def clear(self) -> None:
        """Delete all rows for the current session."""
        async with self._lock:
            await asyncio.to_thread(self._clear_sync)

    def _clear_sync(self) -> None:
        self._conn.execute(
            "DELETE FROM messages WHERE session_id = ?",
            (self.session_id,),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()


def _row_to_message(row: dict[str, Any]) -> Message:
    """Deserialise a SQLite row into a ``Message``."""
    from datetime import datetime

    tool_calls = [ToolCall.model_validate(d) for d in json.loads(row["tool_calls"])]
    tool_results = [ToolResult.model_validate(d) for d in json.loads(row["tool_results"])]
    return Message(
        id=row["message_id"],
        role=Role(row["role"]),
        content=row["content"],
        tool_calls=tool_calls,
        tool_results=tool_results,
        created_at=datetime.fromisoformat(row["created_at"]),
        metadata=json.loads(row["metadata"]),
    )


__all__ = ["InMemoryConversation", "SQLiteConversation"]
