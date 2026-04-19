"""The Agent: the core async loop that ties providers, tools, and memory.

An ``Agent`` runs an LLM conversation until the model stops calling tools,
enforcing hard limits on iterations, tokens, and USD spend. It supports:

* full message-at-a-time ``run``,
* token-by-token ``stream``,
* parallel tool-call execution,
* provider swaps via ``settings.provider``,
* lifecycle hooks (``on_message``, ``on_tool_call``, ``on_tool_result``,
  ``on_error``),
* automatic retry on transient provider errors.

Example:
    >>> import asyncio
    >>> from agent_kit import Agent
    >>> async def main() -> None:
    ...     agent = Agent(system="You are helpful.")
    ...     result = await agent.run("Say hi in one word.")
    ...     print(result.output_text)
    >>> asyncio.run(main())  # doctest: +SKIP
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agent_kit.core.config import Settings
from agent_kit.core.messages import Message, Role, ToolCall, ToolResult, Usage
from agent_kit.core.provider import Provider, StreamEvent, get_provider
from agent_kit.memory.base import Memory
from agent_kit.memory.conversation import InMemoryConversation
from agent_kit.observability.cost import CostLedger
from agent_kit.observability.tracing import Tracer, get_logger, get_tracer
from agent_kit.tools.registry import ToolRegistry

# Hook signatures. All hooks are async; sync callers can wrap with
# ``asyncio.to_thread``.
OnMessage = Callable[[Message], Awaitable[None]]
OnToolCall = Callable[[ToolCall], Awaitable[None]]
OnToolResult = Callable[[ToolCall, ToolResult], Awaitable[None]]
OnError = Callable[[Exception], Awaitable[None]]


class AgentError(RuntimeError):
    """Raised when the agent cannot make further progress."""


class BudgetExceeded(AgentError):
    """Raised when an iteration, token, or cost limit is hit."""


class MaxIterationsExceeded(BudgetExceeded):
    """Raised when the agent exceeds ``max_iterations`` without stopping."""


class CostLimitExceeded(BudgetExceeded):
    """Raised when the cumulative cost would exceed ``max_cost_usd``."""


@dataclass
class AgentResult:
    """Result of an :meth:`Agent.run` invocation.

    Attributes:
        output_text: Final assistant text (last non-tool-call turn).
        messages: Every message produced during the run (excluding initial history).
        usage: Cumulative usage for the run.
        iterations: Number of provider calls performed.
        stop_reason: One of ``"done" | "max_iterations" | "cost_limit"``.
    """

    output_text: str
    messages: list[Message]
    usage: Usage
    iterations: int
    stop_reason: str = "done"


class AgentConfig(BaseModel):
    """Per-run overrides to the global ``Settings`` defaults."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_iterations: int | None = None
    max_cost_usd: float | None = None


@dataclass
class _Hooks:
    on_message: list[OnMessage] = field(default_factory=list)
    on_tool_call: list[OnToolCall] = field(default_factory=list)
    on_tool_result: list[OnToolResult] = field(default_factory=list)
    on_error: list[OnError] = field(default_factory=list)


class Agent:
    """An LLM agent with tools, memory, streaming, hooks, and budgets.

    The agent is stateless across ``run`` calls unless you pass a ``Memory``
    backend — the conversation history always lives in the memory object.

    Example:
        >>> import asyncio
        >>> from agent_kit import Agent
        >>> async def demo() -> None:
        ...     agent = Agent(system="Be concise.")
        ...     result = await agent.run("What's 2+2?")
        ...     assert "4" in result.output_text
        >>> asyncio.run(demo())  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        system: str | None = None,
        provider: Provider | None = None,
        tools: ToolRegistry | None = None,
        memory: Memory | None = None,
        tracer: Tracer | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        """Construct an agent.

        Args:
            settings: Runtime settings; defaults to ``Settings()`` (env-driven).
            system: System prompt; overrides any default.
            provider: Pre-built provider; otherwise built from ``settings``.
            tools: Tool registry; defaults to empty.
            memory: Memory backend; defaults to in-process conversation store.
            tracer: Tracer to use; defaults to the process-wide singleton.
            config: Per-agent overrides of model/temperature/limits.
        """
        self.settings = settings or Settings()
        self.config = config or AgentConfig()
        self.system = system
        self.provider = provider or get_provider(self.settings)
        self.tools = tools or ToolRegistry()
        self.memory = memory or InMemoryConversation()
        self.tracer = tracer or get_tracer(self.settings.trace_file)
        self.logger = get_logger(
            "agent_kit.agent",
            level=self.settings.log_level,
            json_logs=self.settings.log_json,
        )
        self.ledger = CostLedger()
        self._hooks = _Hooks()

    # ---- Hook registration ----

    def on_message(self, fn: OnMessage) -> OnMessage:
        """Register a coroutine invoked for every message produced."""
        self._hooks.on_message.append(fn)
        return fn

    def on_tool_call(self, fn: OnToolCall) -> OnToolCall:
        """Register a coroutine invoked before each tool call runs."""
        self._hooks.on_tool_call.append(fn)
        return fn

    def on_tool_result(self, fn: OnToolResult) -> OnToolResult:
        """Register a coroutine invoked after each tool call completes."""
        self._hooks.on_tool_result.append(fn)
        return fn

    def on_error(self, fn: OnError) -> OnError:
        """Register a coroutine invoked when the agent hits an error."""
        self._hooks.on_error.append(fn)
        return fn

    # ---- Public runs ----

    async def run(
        self,
        prompt: str | None = None,
        *,
        extra_messages: Sequence[Message] | None = None,
    ) -> AgentResult:
        """Run the agent loop to completion and return the final result.

        Args:
            prompt: Convenience; appended as a user message if provided.
            extra_messages: Additional messages to append before running.

        Returns:
            ``AgentResult`` with the final text, all messages, and usage.
        """
        if prompt is not None:
            await self.memory.append(Message.user(prompt))
        for m in extra_messages or []:
            await self.memory.append(m)

        produced: list[Message] = []
        iterations = 0
        max_iter = self.config.max_iterations or self.settings.max_iterations
        stop_reason = "done"

        with self.tracer.span(
            "agent.run",
            {"provider": self.provider.name, "model": self._model()},
        ) as span:
            while True:
                if iterations >= max_iter:
                    stop_reason = "max_iterations"
                    await self._fire_error(MaxIterationsExceeded(max_iter))
                    break
                iterations += 1
                history = await self.memory.history()
                assistant_msg, usage = await self._call_provider(history)
                self.ledger.record(usage)
                self._enforce_cost_limit(usage)
                produced.append(assistant_msg)
                await self.memory.append(assistant_msg)
                await self._fire_message(assistant_msg)

                if not assistant_msg.tool_calls:
                    break

                tool_msg = await self._run_tool_calls(assistant_msg.tool_calls)
                produced.append(tool_msg)
                await self.memory.append(tool_msg)
                await self._fire_message(tool_msg)

            span.set_attribute("iterations", iterations)
            span.set_attribute("input_tokens", self.ledger.total.input_tokens)
            span.set_attribute("output_tokens", self.ledger.total.output_tokens)
            span.set_attribute("cost_usd", self.ledger.total.cost_usd or 0.0)
            span.set_attribute("stop_reason", stop_reason)

        output_text = _last_text(produced)
        return AgentResult(
            output_text=output_text,
            messages=produced,
            usage=self.ledger.total,
            iterations=iterations,
            stop_reason=stop_reason,
        )

    async def stream(
        self,
        prompt: str | None = None,
        *,
        extra_messages: Sequence[Message] | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens as they arrive from the provider.

        Tool calls are executed between provider turns; tokens from tool
        results are not streamed. Use ``stream_events`` for full fidelity.

        Example:
            >>> import asyncio
            >>> async def demo() -> None:
            ...     agent = Agent()
            ...     async for t in agent.stream("Say hi"):
            ...         print(t, end="")
            >>> asyncio.run(demo())  # doctest: +SKIP
        """
        async for event in self.stream_events(prompt, extra_messages=extra_messages):
            if event.kind == "text":
                yield event.text

    async def stream_events(
        self,
        prompt: str | None = None,
        *,
        extra_messages: Sequence[Message] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream ``StreamEvent``s covering text, tool calls, and stop events."""
        if prompt is not None:
            await self.memory.append(Message.user(prompt))
        for m in extra_messages or []:
            await self.memory.append(m)

        iterations = 0
        max_iter = self.config.max_iterations or self.settings.max_iterations

        while True:
            if iterations >= max_iter:
                await self._fire_error(MaxIterationsExceeded(max_iter))
                yield StreamEvent(kind="error", error="max_iterations")
                return
            iterations += 1

            history = await self.memory.history()
            assembled_text = ""
            tool_calls: list[ToolCall] = []

            async for event in self._stream_provider(history):
                if event.kind == "text":
                    assembled_text += event.text
                    yield event
                elif event.kind == "tool_call" and event.tool_call is not None:
                    tool_calls.append(event.tool_call)
                    yield event
                elif event.kind == "message_stop":
                    if event.usage is not None:
                        self.ledger.record(event.usage)
                        self._enforce_cost_limit(event.usage)
                    yield event
                elif event.kind == "error":
                    await self._fire_error(RuntimeError(event.error or "provider error"))
                    yield event
                    return

            assistant_msg = Message.assistant(assembled_text, tool_calls=tool_calls)
            await self.memory.append(assistant_msg)
            await self._fire_message(assistant_msg)

            if not tool_calls:
                return

            tool_msg = await self._run_tool_calls(tool_calls)
            await self.memory.append(tool_msg)
            await self._fire_message(tool_msg)

    # ---- Internal helpers ----

    def _model(self) -> str:
        return self.config.model or self.settings.model

    def _retry(self) -> AsyncRetrying:
        """Build a tenacity retry policy from settings."""
        return AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(self.settings.retry_max_attempts),
            wait=wait_exponential(
                multiplier=self.settings.retry_initial_delay,
                max=self.settings.retry_max_delay,
            ),
            retry=retry_if_exception_type(_RETRYABLE_EXC),
        )

    async def _call_provider(self, history: list[Message]) -> tuple[Message, Usage]:
        tool_schemas = self.tools.schemas() or None
        try:
            async for attempt in self._retry():
                with (
                    attempt,
                    self.tracer.span(
                        "provider.complete",
                        {"provider": self.provider.name, "model": self._model()},
                    ) as span,
                ):
                    msg, usage = await self.provider.complete(
                        system=self.system,
                        messages=history,
                        tools=tool_schemas,
                        model=self.config.model,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )
                    span.set_attribute("input_tokens", usage.input_tokens)
                    span.set_attribute("output_tokens", usage.output_tokens)
                    return msg, usage
        except RetryError as e:  # pragma: no cover - tenacity re-raises cause
            await self._fire_error(e)
            raise
        raise AgentError("retry loop exited unexpectedly")  # pragma: no cover

    async def _stream_provider(self, history: list[Message]) -> AsyncIterator[StreamEvent]:
        tool_schemas = self.tools.schemas() or None
        try:
            async for attempt in self._retry():
                with attempt:
                    with self.tracer.span(
                        "provider.stream",
                        {"provider": self.provider.name, "model": self._model()},
                    ):
                        async for event in self.provider.stream(
                            system=self.system,
                            messages=history,
                            tools=tool_schemas,
                            model=self.config.model,
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                        ):
                            yield event
                    return
        except RetryError as e:  # pragma: no cover - tenacity re-raises cause
            await self._fire_error(e)
            raise

    async def _run_tool_calls(self, calls: list[ToolCall]) -> Message:
        """Execute tool calls in parallel and wrap results in a tool message."""
        if not calls:
            return Message.tool([])

        async def run_one(call: ToolCall) -> ToolResult:
            await self._fire_tool_call(call)
            with self.tracer.span("tool.execute", {"tool": call.name}) as span:
                try:
                    result = await self.tools.execute(call)
                    span.set_attribute("is_error", result.is_error)
                except Exception as exc:
                    await self._fire_error(exc)
                    result = ToolResult(
                        tool_call_id=call.id,
                        content=f"Tool {call.name!r} raised: {exc}",
                        is_error=True,
                    )
            await self._fire_tool_result(call, result)
            return result

        results = await asyncio.gather(*(run_one(c) for c in calls))
        return Message.tool(list(results))

    def _enforce_cost_limit(self, usage: Usage) -> None:
        limit = self.config.max_cost_usd or self.settings.max_cost_usd
        total = self.ledger.total.cost_usd or 0.0
        if total > limit:
            err = CostLimitExceeded(f"cumulative cost ${total:.4f} exceeds limit ${limit:.2f}")
            self.logger.warning(
                "cost.limit_exceeded",
                total_usd=total,
                limit_usd=limit,
                last_call_usd=usage.cost_usd,
            )
            raise err

    async def _fire_message(self, message: Message) -> None:
        for hook in self._hooks.on_message:
            try:
                await hook(message)
            except Exception as exc:
                self.logger.error("hook.on_message.error", error=str(exc))

    async def _fire_tool_call(self, call: ToolCall) -> None:
        for hook in self._hooks.on_tool_call:
            try:
                await hook(call)
            except Exception as exc:
                self.logger.error("hook.on_tool_call.error", error=str(exc))

    async def _fire_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        for hook in self._hooks.on_tool_result:
            try:
                await hook(call, result)
            except Exception as exc:
                self.logger.error("hook.on_tool_result.error", error=str(exc))

    async def _fire_error(self, error: Exception) -> None:
        for hook in self._hooks.on_error:
            try:
                await hook(error)
            except Exception as exc:
                self.logger.error("hook.on_error.error", error=str(exc))


def _last_text(messages: Sequence[Message]) -> str:
    """Return the final non-empty assistant text in a list."""
    for m in reversed(messages):
        if m.role == Role.ASSISTANT and m.content:
            return m.content
    return ""


# Pulled out for readability: which exceptions should trigger retry.
try:
    from anthropic import APIConnectionError as _AnthropicConnError
    from anthropic import APIStatusError as _AnthropicStatusError
    from anthropic import RateLimitError as _AnthropicRateLimitError
    from openai import APIConnectionError as _OpenAIConnError
    from openai import APIStatusError as _OpenAIStatusError
    from openai import RateLimitError as _OpenAIRateLimitError

    _RETRYABLE_EXC: tuple[type[BaseException], ...] = (
        _AnthropicConnError,
        _AnthropicRateLimitError,
        _AnthropicStatusError,
        _OpenAIConnError,
        _OpenAIRateLimitError,
        _OpenAIStatusError,
        asyncio.TimeoutError,
        ConnectionError,
    )
except ImportError:  # pragma: no cover - SDKs are hard deps
    _RETRYABLE_EXC = (asyncio.TimeoutError, ConnectionError)


__all__ = [
    "Agent",
    "AgentConfig",
    "AgentError",
    "AgentResult",
    "BudgetExceeded",
    "CostLimitExceeded",
    "MaxIterationsExceeded",
]
