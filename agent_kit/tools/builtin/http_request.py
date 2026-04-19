"""HTTP request tool for GETing / POSTing arbitrary URLs.

Use this tool for simple API fetches. It blocks non-HTTP(S) schemes and
(optionally) a deny-list of hosts.

Example:
    >>> t = HTTPRequestTool()
    >>> t.name
    'http_request'
"""

from __future__ import annotations

from typing import Any, ClassVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from agent_kit.tools.base import Tool


class HTTPRequestArgs(BaseModel):
    """Arguments for :class:`HTTPRequestTool`."""

    url: str = Field(..., description="Fully-qualified http(s) URL.")
    method: str = Field("GET", description="HTTP method.")
    headers: dict[str, str] = Field(default_factory=dict)
    json_body: dict[str, Any] | None = Field(default=None)
    params: dict[str, str] = Field(default_factory=dict)
    timeout: float = Field(30.0, ge=0.1, le=120.0)


class HTTPRequestTool(Tool[HTTPRequestArgs]):
    """Minimal HTTP client exposed as a tool."""

    name: ClassVar[str] = "http_request"
    description: ClassVar[str] = "Make an HTTP(S) request and return {status, headers, body}."
    args_model: ClassVar[type[BaseModel]] = HTTPRequestArgs

    _MAX_BODY_BYTES = 500_000

    def __init__(self, denied_hosts: list[str] | None = None) -> None:
        """Build the tool with an optional host deny-list."""
        self.denied_hosts = set(denied_hosts or [])

    async def run(self, args: HTTPRequestArgs) -> dict[str, Any]:
        """Perform the HTTP request."""
        parsed = urlparse(args.url)
        if parsed.scheme not in {"http", "https"}:
            return {"error": f"unsupported scheme: {parsed.scheme}"}
        if parsed.hostname in self.denied_hosts:
            return {"error": f"host {parsed.hostname!r} is denied"}

        async with httpx.AsyncClient(timeout=args.timeout) as client:
            response = await client.request(
                args.method.upper(),
                args.url,
                headers=args.headers,
                params=args.params,
                json=args.json_body,
            )
            body = response.content[: self._MAX_BODY_BYTES].decode("utf-8", errors="replace")
            truncated = len(response.content) > self._MAX_BODY_BYTES
            return {
                "status": response.status_code,
                "headers": dict(response.headers),
                "body": body,
                "truncated": truncated,
            }


__all__ = ["HTTPRequestArgs", "HTTPRequestTool"]
