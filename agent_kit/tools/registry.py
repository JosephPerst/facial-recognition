"""Tool registry: registration, schema export, parallel execution.

The registry is the boundary between the agent loop and individual tools.
It validates names, enforces uniqueness, surfaces provider-ready schemas,
and executes tool calls (with error wrapping) in parallel via ``asyncio``.

Example:
    >>> from agent_kit.tools.base import tool
    >>> from agent_kit.tools.registry import ToolRegistry
    >>> @tool()
    ... async def ping(msg: str) -> str:
    ...     '''Echo a message.'''
    ...     return f"pong: {msg}"
    >>> r = ToolRegistry([ping])
    >>> "ping" in r
    True
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator
from typing import Any

from agent_kit.core.messages import ToolCall, ToolResult
from agent_kit.tools.base import Tool


class DuplicateToolError(ValueError):
    """Raised when two tools are registered under the same name."""


class ToolNotFoundError(KeyError):
    """Raised when a tool call targets an unknown tool."""


class ToolRegistry:
    """Mutable map of ``name -> Tool`` with provider-ready schema export.

    Example:
        >>> from agent_kit.tools.base import tool
        >>> @tool()
        ... async def add(a: int, b: int) -> int:
        ...     '''Add two numbers.'''
        ...     return a + b
        >>> r = ToolRegistry([add])
        >>> [s["name"] for s in r.schemas()]
        ['add']
    """

    def __init__(self, tools: Iterable[Tool[Any]] | None = None) -> None:
        """Initialise the registry, optionally with a starter set of tools."""
        self._tools: dict[str, Tool[Any]] = {}
        for t in tools or ():
            self.register(t)

    def register(self, tool_instance: Tool[Any]) -> None:
        """Add a tool. Raises :class:`DuplicateToolError` on name collision."""
        if not tool_instance.name:
            raise ValueError("tool has no name")
        if tool_instance.name in self._tools:
            raise DuplicateToolError(tool_instance.name)
        self._tools[tool_instance.name] = tool_instance

    def unregister(self, name: str) -> None:
        """Remove a tool by name; no-op if missing."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool[Any]:
        """Return the tool registered under ``name``."""
        try:
            return self._tools[name]
        except KeyError as e:
            raise ToolNotFoundError(name) from e

    def names(self) -> list[str]:
        """Return all registered tool names (sorted for determinism)."""
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """Return provider-agnostic tool specs for every registered tool."""
        return [self._tools[n].to_spec() for n in self.names()]

    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call, catching errors into the result."""
        try:
            tool_instance = self.get(call.name)
        except ToolNotFoundError:
            return ToolResult(
                tool_call_id=call.id,
                content=f"Tool {call.name!r} is not registered.",
                is_error=True,
            )
        try:
            content = await tool_instance.invoke(call.arguments)
            return ToolResult(tool_call_id=call.id, content=content, is_error=False)
        except Exception as exc:
            return ToolResult(
                tool_call_id=call.id,
                content=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )

    async def execute_many(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Execute several tool calls in parallel, preserving order."""
        return list(await asyncio.gather(*(self.execute(c) for c in calls)))

    def __contains__(self, name: object) -> bool:
        """Membership by tool name."""
        return isinstance(name, str) and name in self._tools

    def __iter__(self) -> Iterator[Tool[Any]]:
        """Iterate over registered tools in deterministic order."""
        for name in self.names():
            yield self._tools[name]

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)


__all__ = ["DuplicateToolError", "ToolNotFoundError", "ToolRegistry"]
