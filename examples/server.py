"""FastAPI wrapper exposing an Agent Kit agent over HTTP with SSE streaming.

Run:

    uv run agent-kit-server
    # or
    uv run uvicorn examples.server:app --reload

Endpoints:

* ``GET  /healthz``      — readiness probe.
* ``POST /v1/agent``     — non-streaming run; returns final text + usage.
* ``POST /v1/agent/stream`` — Server-Sent Events stream.

All routes accept JSON ``{"prompt": str, "session_id": str}`` and are
authenticated via the optional ``X-Api-Key`` header matched against the
``AGENT_KIT_API_KEYS`` env var (comma-separated).
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

from agent_kit import Agent, Settings
from agent_kit.memory.conversation import SQLiteConversation
from agent_kit.tools.builtin import default_tools
from agent_kit.tools.registry import ToolRegistry
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse


class AgentRequest(BaseModel):
    """Request body shared by the two agent endpoints."""

    prompt: str = Field(..., min_length=1)
    session_id: str = "default"
    system: str | None = None


class AgentResponse(BaseModel):
    """Non-streaming response payload."""

    output: str
    iterations: int
    cost_usd: float | None
    input_tokens: int
    output_tokens: int
    session_id: str


def _api_keys() -> set[str]:
    """Load comma-separated API keys from ``AGENT_KIT_API_KEYS``."""
    raw = os.environ.get("AGENT_KIT_API_KEYS", "").strip()
    return {k.strip() for k in raw.split(",") if k.strip()}


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency enforcing optional API-key auth."""
    allowed = _api_keys()
    if not allowed:
        return  # auth disabled
    if x_api_key not in allowed:
        raise HTTPException(status_code=401, detail="invalid api key")


def build_agent(settings: Settings, session_id: str, system: str | None) -> Agent:
    """Build an agent with persistent SQLite memory keyed by ``session_id``."""
    memory = SQLiteConversation(settings.sqlite_path, session_id=session_id)
    tools = ToolRegistry(default_tools())
    return Agent(settings=settings, system=system, tools=tools, memory=memory)


app = FastAPI(title="Agent Kit", version="0.1.0")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness / readiness probe."""
    return {"status": "ok"}


@app.post("/v1/agent", response_model=AgentResponse)
async def run_agent(
    body: AgentRequest,
    _: None = Depends(require_api_key),
) -> AgentResponse:
    """Non-streaming agent invocation."""
    settings = Settings()
    agent = build_agent(settings, body.session_id, body.system)
    result = await agent.run(body.prompt)
    usage = result.usage
    return AgentResponse(
        output=result.output_text,
        iterations=result.iterations,
        cost_usd=usage.cost_usd,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        session_id=body.session_id,
    )


@app.post("/v1/agent/stream")
async def stream_agent(
    body: AgentRequest,
    _: None = Depends(require_api_key),
) -> EventSourceResponse:
    """SSE endpoint streaming tokens and a final ``done`` event."""
    settings = Settings()
    agent = build_agent(settings, body.session_id, body.system)

    async def events() -> AsyncIterator[dict[str, str]]:
        async for event in agent.stream_events(body.prompt):
            if event.kind == "text":
                yield {"event": "token", "data": event.text}
            elif event.kind == "tool_call" and event.tool_call is not None:
                yield {
                    "event": "tool_call",
                    "data": json.dumps(
                        {"name": event.tool_call.name, "arguments": event.tool_call.arguments}
                    ),
                }
            elif event.kind == "message_stop":
                payload = {
                    "input_tokens": event.usage.input_tokens if event.usage else 0,
                    "output_tokens": event.usage.output_tokens if event.usage else 0,
                    "cost_usd": event.usage.cost_usd if event.usage else None,
                }
                yield {"event": "message_stop", "data": json.dumps(payload)}
        total = agent.ledger.total
        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "cost_usd": total.cost_usd,
                    "input_tokens": total.input_tokens,
                    "output_tokens": total.output_tokens,
                }
            ),
        }

    return EventSourceResponse(events())


def main() -> None:
    """Launch the server via ``uvicorn``. Used by ``agent-kit-server`` script."""
    import uvicorn

    uvicorn.run(
        "examples.server:app",
        host=os.environ.get("AGENT_KIT_HOST", "0.0.0.0"),
        port=int(os.environ.get("AGENT_KIT_PORT", "8000")),
        log_level=os.environ.get("AGENT_KIT_LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
