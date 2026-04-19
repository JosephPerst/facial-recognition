"""Tests for the Agent class: run, stream, hooks, limits, tool calls."""

from __future__ import annotations

import pytest
from agent_kit import Agent
from agent_kit.core.agent import BudgetExceeded
from agent_kit.core.config import Settings
from agent_kit.core.messages import Message, ToolCall
from agent_kit.memory.conversation import InMemoryConversation
from agent_kit.tools.base import tool
from agent_kit.tools.registry import ToolRegistry

from tests.conftest import FakeProvider


async def test_agent_run_simple(settings: Settings) -> None:
    provider = FakeProvider(settings, script=[("Hello world", [])])
    agent = Agent(settings=settings, provider=provider)
    result = await agent.run("hi")
    assert result.output_text == "Hello world"
    assert result.iterations == 1
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 20


async def test_agent_stream_yields_tokens(settings: Settings) -> None:
    provider = FakeProvider(settings, script=[("abc", [])])
    agent = Agent(settings=settings, provider=provider)
    tokens = [t async for t in agent.stream("go")]
    assert "".join(tokens) == "abc"


async def test_agent_runs_tool_and_loops(settings: Settings) -> None:
    @tool()
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    registry = ToolRegistry([add])
    call = ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})
    provider = FakeProvider(
        settings,
        script=[("", [call]), ("5", [])],
    )
    agent = Agent(settings=settings, provider=provider, tools=registry)
    result = await agent.run("what is 2+3?")
    assert result.output_text == "5"
    assert result.iterations == 2


async def test_agent_parallel_tool_calls(settings: Settings) -> None:
    @tool()
    async def square(x: int) -> int:
        """Square a number."""
        return x * x

    registry = ToolRegistry([square])
    calls = [
        ToolCall(id="c1", name="square", arguments={"x": 2}),
        ToolCall(id="c2", name="square", arguments={"x": 3}),
    ]
    provider = FakeProvider(settings, script=[("", calls), ("done", [])])
    agent = Agent(settings=settings, provider=provider, tools=registry)
    result = await agent.run("square 2 and 3")
    # Tool message holds both results
    tool_messages = [m for m in result.messages if m.role.value == "tool"]
    assert len(tool_messages) == 1
    assert len(tool_messages[0].tool_results) == 2


async def test_agent_hooks_fire(settings: Settings) -> None:
    provider = FakeProvider(settings, script=[("hi", [])])
    agent = Agent(settings=settings, provider=provider)
    seen: list[str] = []

    @agent.on_message
    async def _m(msg: Message) -> None:
        seen.append(msg.role.value)

    await agent.run("hey")
    assert "assistant" in seen


async def test_agent_max_iterations(settings: Settings) -> None:
    loop_call = ToolCall(id="c1", name="square", arguments={"x": 1})

    @tool()
    async def square(x: int) -> int:
        """Square."""
        return x * x

    provider = FakeProvider(
        settings,
        script=[("", [loop_call])] * 20,
    )
    agent = Agent(
        settings=settings,
        provider=provider,
        tools=ToolRegistry([square]),
    )
    result = await agent.run("loop")
    assert result.stop_reason == "max_iterations"
    assert result.iterations == settings.max_iterations


async def test_agent_cost_limit(settings: Settings) -> None:
    s = settings.model_copy(update={"max_cost_usd": 0.0005})
    provider = FakeProvider(s, script=[("bye", [])])
    agent = Agent(settings=s, provider=provider)
    with pytest.raises(BudgetExceeded):
        await agent.run("over budget")


async def test_agent_conversation_memory_persists(settings: Settings) -> None:
    memory = InMemoryConversation()
    provider = FakeProvider(settings, script=[("one", []), ("two", [])])
    agent = Agent(settings=settings, provider=provider, memory=memory)
    await agent.run("first")
    await agent.run("second")
    history = await memory.history()
    # user1, assistant1, user2, assistant2
    assert len(history) == 4
