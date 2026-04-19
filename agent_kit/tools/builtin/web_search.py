"""Web search tool via DuckDuckGo's zero-config HTML endpoint.

DuckDuckGo is used because it doesn't require an API key — ideal for a
starter kit. Swap in Tavily, Brave, or Serper by subclassing ``WebSearchTool``
and overriding ``fetch``.

Example:
    >>> tool = WebSearchTool()
    >>> tool.name
    'web_search'
"""

from __future__ import annotations

import re
import urllib.parse
from typing import ClassVar

import httpx
from pydantic import BaseModel, Field

from agent_kit.tools.base import Tool


class WebSearchArgs(BaseModel):
    """Arguments for :class:`WebSearchTool`."""

    query: str = Field(..., description="Search query.")
    max_results: int = Field(5, ge=1, le=20, description="Maximum result count.")


class WebSearchResult(BaseModel):
    """A single search result."""

    title: str
    url: str
    snippet: str


_RESULT_RE = re.compile(
    r'<a rel="nofollow" class="result__a" href="([^"]+)">(.*?)</a>.*?'
    r'<a class="result__snippet".*?>(.*?)</a>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


class WebSearchTool(Tool[WebSearchArgs]):
    """Keyword search against DuckDuckGo's HTML endpoint."""

    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = (
        "Search the web and return a list of {title, url, snippet} results."
    )
    args_model: ClassVar[type[BaseModel]] = WebSearchArgs

    def __init__(self, timeout: float = 15.0) -> None:
        """Create a web search tool with a per-request timeout."""
        self._timeout = timeout

    async def run(self, args: WebSearchArgs) -> list[dict[str, str]]:
        """Execute the search and return parsed results."""
        html = await self._fetch(args.query)
        return [r.model_dump() for r in _parse(html)[: args.max_results]]

    async def _fetch(self, query: str) -> str:
        """Fetch the raw HTML response for ``query``."""
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                url,
                data=params,
                headers={"User-Agent": "agent-kit/0.1 (+https://agent-kit.dev)"},
            )
            response.raise_for_status()
            return response.text


def _parse(html: str) -> list[WebSearchResult]:
    """Extract search result cards from DuckDuckGo HTML output."""
    results: list[WebSearchResult] = []
    for url, title, snippet in _RESULT_RE.findall(html):
        if url.startswith("/"):
            qs = urllib.parse.urlparse(url).query
            real = urllib.parse.parse_qs(qs).get("uddg", [""])[0]
            url = urllib.parse.unquote(real) or url
        results.append(
            WebSearchResult(
                title=_strip(title),
                url=url.strip(),
                snippet=_strip(snippet),
            )
        )
    return results


def _strip(text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    return re.sub(r"\s+", " ", _TAG_RE.sub("", text)).strip()


__all__ = ["WebSearchArgs", "WebSearchResult", "WebSearchTool"]
