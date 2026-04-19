"""Agent Kit — a production-grade Python starter for building AI agents.

Agent Kit gives you an opinionated, typed, async foundation for shipping agent
products: provider-agnostic LLM access, tool calling with parallel execution,
memory, evals, observability, and deploy targets.

Example:
    >>> import asyncio
    >>> from agent_kit import Agent, Settings
    >>> async def main() -> None:
    ...     agent = Agent(settings=Settings())
    ...     async for token in agent.stream("Hello, what's 2 + 2?"):
    ...         print(token, end="")
    >>> asyncio.run(main())  # doctest: +SKIP
"""

from __future__ import annotations

from agent_kit.core.agent import Agent, AgentResult
from agent_kit.core.config import Settings
from agent_kit.core.messages import Message, Role, ToolCall, ToolResult
from agent_kit.core.provider import Provider, get_provider
from agent_kit.tools.base import Tool
from agent_kit.tools.registry import ToolRegistry

__all__ = [
    "Agent",
    "AgentResult",
    "Message",
    "Provider",
    "Role",
    "Settings",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "get_provider",
]

__version__ = "0.1.0"
