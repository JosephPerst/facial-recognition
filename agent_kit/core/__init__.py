"""Core agent primitives: messages, config, providers, agent loop."""

from __future__ import annotations

from agent_kit.core.agent import Agent, AgentResult
from agent_kit.core.config import Settings
from agent_kit.core.messages import Message, Role, ToolCall, ToolResult, Usage
from agent_kit.core.provider import Provider, StreamEvent, get_provider

__all__ = [
    "Agent",
    "AgentResult",
    "Message",
    "Provider",
    "Role",
    "Settings",
    "StreamEvent",
    "ToolCall",
    "ToolResult",
    "Usage",
    "get_provider",
]
