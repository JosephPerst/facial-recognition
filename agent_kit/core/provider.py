"""Provider abstraction + Anthropic / OpenAI / Ollama implementations.

Providers translate between Agent Kit's canonical ``Message`` model and the
wire format expected by the underlying SDK. Every provider exposes the same
two coroutines:

* ``complete`` — non-streaming, returns a final assistant ``Message``.
* ``stream`` — async iterator of ``StreamEvent`` chunks.

Both return a ``Usage`` record so the agent can enforce budgets.

Example:
    >>> from agent_kit.core.config import Settings
    >>> from agent_kit.core.provider import get_provider
    >>> provider = get_provider(Settings())  # doctest: +SKIP
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from agent_kit.core.config import ProviderName, Settings
from agent_kit.core.messages import Message, Role, ToolCall, Usage
from agent_kit.observability.cost import price_usage


@dataclass
class StreamEvent:
    """Incremental event emitted while streaming a completion.

    Attributes:
        kind: ``"text"`` for a token delta, ``"tool_call"`` for a fully-assembled
            tool invocation, ``"message_stop"`` when the turn completes,
            ``"error"`` for a terminal error.
        text: The text delta for ``kind == "text"``.
        tool_call: A ``ToolCall`` for ``kind == "tool_call"``.
        usage: Final usage totals on ``kind == "message_stop"``.
        error: Human-readable message on ``kind == "error"``.
    """

    kind: Literal["text", "tool_call", "message_stop", "error"]
    text: str = ""
    tool_call: ToolCall | None = None
    usage: Usage | None = None
    error: str | None = None


class Provider(ABC):
    """Abstract provider: translate Agent Kit messages to/from a vendor SDK."""

    name: ProviderName

    def __init__(self, settings: Settings) -> None:
        """Store settings; subclasses build the SDK client lazily."""
        self.settings = settings

    @abstractmethod
    async def complete(
        self,
        *,
        system: str | None,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[Message, Usage]:
        """Return a complete assistant ``Message`` and its ``Usage``."""

    @abstractmethod
    def stream(
        self,
        *,
        system: str | None,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield incremental ``StreamEvent``s."""

    def _price(self, usage: Usage, model: str) -> Usage:
        """Attach a USD cost to ``usage`` using pricing tables."""
        return price_usage(self.name, model, usage)


# ---------- Anthropic ----------


class AnthropicProvider(Provider):
    """Anthropic Claude provider via the official async SDK."""

    name: ProviderName = "anthropic"

    def __init__(self, settings: Settings) -> None:
        """Initialise the Anthropic async client."""
        super().__init__(settings)
        self._client = AsyncAnthropic(
            api_key=settings.anthropic_key(),
            timeout=settings.request_timeout,
        )

    async def complete(
        self,
        *,
        system: str | None,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[Message, Usage]:
        """Non-streaming completion via Anthropic's Messages API."""
        kwargs = self._build_kwargs(
            system=system,
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = await self._client.messages.create(**kwargs)
        text = ""
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input) if isinstance(block.input, dict) else {},
                    )
                )
        usage = self._price(
            Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            kwargs["model"],
        )
        return (
            Message(role=Role.ASSISTANT, content=text, tool_calls=tool_calls),
            usage,
        )

    async def stream(
        self,
        *,
        system: str | None,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming completion via Anthropic's Messages API."""
        kwargs = self._build_kwargs(
            system=system,
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                kind = getattr(event, "type", None)
                if kind == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta is not None and getattr(delta, "type", None) == "text_delta":
                        yield StreamEvent(kind="text", text=delta.text)
            final = await stream.get_final_message()
        for block in final.content:
            if block.type == "tool_use":
                yield StreamEvent(
                    kind="tool_call",
                    tool_call=ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input) if isinstance(block.input, dict) else {},
                    ),
                )
        usage = self._price(
            Usage(
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
            ),
            kwargs["model"],
        )
        yield StreamEvent(kind="message_stop", usage=usage)

    def _build_kwargs(
        self,
        *,
        system: str | None,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        converted = _to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": model or self.settings.model,
            "messages": converted,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "temperature": (temperature if temperature is not None else self.settings.temperature),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _tools_to_anthropic(tools)
        return kwargs


def _to_anthropic_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Translate Agent Kit messages into Anthropic-format messages."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == Role.SYSTEM:
            # Anthropic uses a top-level ``system`` parameter, not a message.
            continue
        if m.role == Role.USER:
            out.append({"role": "user", "content": m.content})
        elif m.role == Role.ASSISTANT:
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )
            out.append({"role": "assistant", "content": blocks or m.content})
        elif m.role == Role.TOOL:
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": r.tool_call_id,
                            "content": r.content,
                            "is_error": r.is_error,
                        }
                        for r in m.tool_results
                    ],
                }
            )
    return out


def _tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Agent Kit tool schemas to Anthropic's tool format."""
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t["parameters"],
        }
        for t in tools
    ]


# ---------- OpenAI ----------


class OpenAIProvider(Provider):
    """OpenAI Chat Completions provider (tool-calling format)."""

    name: ProviderName = "openai"

    def __init__(self, settings: Settings) -> None:
        """Initialise the OpenAI async client."""
        super().__init__(settings)
        self._client = AsyncOpenAI(
            api_key=settings.openai_key(),
            timeout=settings.request_timeout,
        )

    async def complete(
        self,
        *,
        system: str | None,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[Message, Usage]:
        """Non-streaming completion via OpenAI's Chat Completions API."""
        kwargs = self._build_kwargs(
            system=system,
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls: list[ToolCall] = []
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage = self._price(
            Usage(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
            ),
            kwargs["model"],
        )
        return (
            Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls),
            usage,
        )

    async def stream(
        self,
        *,
        system: str | None,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming completion via OpenAI's Chat Completions API."""
        kwargs = self._build_kwargs(
            system=system,
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        tool_buffers: dict[int, dict[str, Any]] = {}
        final_usage = Usage()
        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.usage is not None:
                final_usage = Usage(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                )
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield StreamEvent(kind="text", text=delta.content)
            for tc in delta.tool_calls or []:
                buf = tool_buffers.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                if tc.id:
                    buf["id"] = tc.id
                if tc.function and tc.function.name:
                    buf["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    buf["arguments"] += tc.function.arguments
        for buf in tool_buffers.values():
            if not buf["id"] or not buf["name"]:
                continue
            try:
                args = json.loads(buf["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {"_raw": buf["arguments"]}
            yield StreamEvent(
                kind="tool_call",
                tool_call=ToolCall(id=buf["id"], name=buf["name"], arguments=args),
            )
        yield StreamEvent(
            kind="message_stop",
            usage=self._price(final_usage, kwargs["model"]),
        )

    def _build_kwargs(
        self,
        *,
        system: str | None,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        converted = _to_openai_messages(messages, system=system)
        kwargs: dict[str, Any] = {
            "model": model or self.settings.model,
            "messages": converted,
            "temperature": (temperature if temperature is not None else self.settings.temperature),
            "max_tokens": max_tokens or self.settings.max_tokens,
        }
        if tools:
            kwargs["tools"] = _tools_to_openai(tools)
        return kwargs


def _to_openai_messages(messages: Sequence[Message], *, system: str | None) -> list[dict[str, Any]]:
    """Translate Agent Kit messages to OpenAI's message schema."""
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m.role == Role.SYSTEM:
            out.append({"role": "system", "content": m.content})
        elif m.role == Role.USER:
            out.append({"role": "user", "content": m.content})
        elif m.role == Role.ASSISTANT:
            entry: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(entry)
        elif m.role == Role.TOOL:
            for r in m.tool_results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": r.tool_call_id,
                        "content": r.content,
                    }
                )
    return out


def _tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wrap Agent Kit tool schemas in OpenAI's function-calling envelope."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ]


# ---------- Ollama ----------


class OllamaProvider(Provider):
    """Local Ollama provider for self-hosted models."""

    name: ProviderName = "ollama"

    def __init__(self, settings: Settings) -> None:
        """Initialise the Ollama async client."""
        super().__init__(settings)
        from ollama import AsyncClient  # local import — keeps top-level imports light

        self._client = AsyncClient(host=settings.ollama_host)

    async def complete(
        self,
        *,
        system: str | None,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[Message, Usage]:
        """Non-streaming completion via Ollama."""
        converted = _to_openai_messages(messages, system=system)
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        kwargs: dict[str, Any] = {
            "model": model or self.settings.model,
            "messages": converted,
            "options": options,
        }
        if tools:
            kwargs["tools"] = _tools_to_openai(tools)
        response = await self._client.chat(**kwargs)
        msg = response["message"] if isinstance(response, dict) else response.message
        content = _ollama_field(msg, "content") or ""
        raw_tool_calls = _ollama_field(msg, "tool_calls") or []
        tool_calls: list[ToolCall] = []
        for i, tc in enumerate(raw_tool_calls):
            fn = _ollama_field(tc, "function") or {}
            name = _ollama_field(fn, "name") or ""
            args = _ollama_field(fn, "arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append(ToolCall(id=f"ollama_{i}", name=name, arguments=args))
        usage = self._price(
            Usage(
                input_tokens=_resp_field(response, "prompt_eval_count") or 0,
                output_tokens=_resp_field(response, "eval_count") or 0,
            ),
            kwargs["model"],
        )
        return (
            Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls),
            usage,
        )

    async def stream(
        self,
        *,
        system: str | None,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming completion via Ollama (tool calls are delivered at end)."""
        converted = _to_openai_messages(messages, system=system)
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        kwargs: dict[str, Any] = {
            "model": model or self.settings.model,
            "messages": converted,
            "options": options,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = _tools_to_openai(tools)
        prompt_tokens = 0
        output_tokens = 0
        async for chunk in await self._client.chat(**kwargs):
            msg = _ollama_field(chunk, "message") or {}
            delta = _ollama_field(msg, "content") or ""
            if delta:
                yield StreamEvent(kind="text", text=delta)
            for i, tc in enumerate(_ollama_field(msg, "tool_calls") or []):
                fn = _ollama_field(tc, "function") or {}
                name = _ollama_field(fn, "name") or ""
                args = _ollama_field(fn, "arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                yield StreamEvent(
                    kind="tool_call",
                    tool_call=ToolCall(id=f"ollama_{i}", name=name, arguments=args),
                )
            prompt_tokens = _ollama_field(chunk, "prompt_eval_count") or prompt_tokens
            output_tokens = _ollama_field(chunk, "eval_count") or output_tokens
        yield StreamEvent(
            kind="message_stop",
            usage=self._price(
                Usage(input_tokens=prompt_tokens, output_tokens=output_tokens),
                kwargs["model"],
            ),
        )


def _ollama_field(obj: Any, key: str) -> Any:
    """Fetch ``key`` from an Ollama response shape (dict or attrs object)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _resp_field(response: Any, key: str) -> Any:
    """Read a top-level field from an Ollama response object."""
    return _ollama_field(response, key)


# ---------- Factory ----------


def get_provider(settings: Settings) -> Provider:
    """Return the ``Provider`` configured in ``settings``.

    Example:
        >>> from agent_kit.core.config import Settings
        >>> isinstance(get_provider(Settings(provider="anthropic")), AnthropicProvider)
        True
    """
    match settings.provider:
        case "anthropic":
            return AnthropicProvider(settings)
        case "openai":
            return OpenAIProvider(settings)
        case "ollama":
            return OllamaProvider(settings)


__all__ = [
    "AnthropicProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "Provider",
    "StreamEvent",
    "get_provider",
]
