"""Typed message primitives shared across providers.

This module defines the canonical message format used by Agent Kit. Every
provider translates to and from these Pydantic models so that the rest of
the system can be provider-agnostic.

Design notes:
    * ``content`` is always a string — tool calls and results live in
      separate fields so downstream consumers don't have to parse
      provider-specific content blocks.
    * ``Usage`` captures input / output tokens and (optionally) the USD cost
      computed by the observability layer.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    """The role of a message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A tool invocation requested by the model.

    Attributes:
        id: Provider-assigned identifier for pairing with a ``ToolResult``.
        name: The name of the tool to invoke.
        arguments: JSON-serialisable argument object.

    Example:
        >>> call = ToolCall(id="call_1", name="web_search", arguments={"q": "hi"})
        >>> call.name
        'web_search'
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The outcome of executing a ``ToolCall``.

    Attributes:
        tool_call_id: ID of the originating ``ToolCall``.
        content: Result serialised as a string (JSON for structured data).
        is_error: True if the tool raised or returned an error payload.

    Example:
        >>> ToolResult(tool_call_id="call_1", content="ok").is_error
        False
    """

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    content: str
    is_error: bool = False


class Usage(BaseModel):
    """Token and cost usage for a single provider call or an accumulated run.

    Attributes:
        input_tokens: Tokens counted on the prompt side.
        output_tokens: Tokens counted on the completion side.
        cost_usd: Dollar cost in USD; ``None`` if the provider's pricing is
            unknown (e.g. Ollama running locally).
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        """Return the sum of input and output tokens."""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        """Add two usage records, summing tokens and costs conservatively."""
        if not isinstance(other, Usage):  # pragma: no cover - defensive
            return NotImplemented
        cost: float | None
        if self.cost_usd is None and other.cost_usd is None:
            cost = None
        else:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=cost,
        )


class Message(BaseModel):
    """A single turn in an agent conversation.

    Attributes:
        id: Stable identifier (useful for memory and tracing).
        role: Speaker role.
        content: Plain-text content. May be empty for pure tool-call turns.
        tool_calls: Tool calls requested by an assistant turn.
        tool_results: Tool results attached to a tool-role turn.
        created_at: UTC timestamp when the message was created.
        name: Optional name (used by some providers for multi-agent setups).
        metadata: Free-form metadata for downstream consumers (e.g. trace IDs).

    Example:
        >>> m = Message(role=Role.USER, content="hello")
        >>> m.role
        <Role.USER: 'user'>
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def system(cls, content: str) -> Message:
        """Build a system message."""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        """Build a user message."""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(
        cls,
        content: str = "",
        tool_calls: list[ToolCall] | None = None,
    ) -> Message:
        """Build an assistant message, optionally with tool calls."""
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls or [],
        )

    @classmethod
    def tool(cls, results: list[ToolResult]) -> Message:
        """Build a tool-role message carrying one or more tool results."""
        return cls(role=Role.TOOL, tool_results=results)


StreamEventKind = Literal[
    "text",
    "tool_call",
    "message_stop",
    "error",
]


__all__ = [
    "Message",
    "Role",
    "StreamEventKind",
    "ToolCall",
    "ToolResult",
    "Usage",
]
