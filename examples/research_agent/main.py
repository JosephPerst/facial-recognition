"""A minimal multi-step research agent.

The agent can call ``web_search`` and ``http_request`` to look things up on
the live web, then returns a cited summary. Streaming is piped to stdout so
you can watch it think in real time.

Run:

    uv run examples/research_agent "who won the 2024 Eurovision?"
"""

from __future__ import annotations

import asyncio
import sys

from agent_kit import Agent, Settings
from agent_kit.core.messages import ToolCall
from agent_kit.tools.builtin import HTTPRequestTool, WebSearchTool
from agent_kit.tools.registry import ToolRegistry
from rich.console import Console

SYSTEM_PROMPT = """You are a careful research assistant.

Workflow:
1. Read the user's question carefully.
2. Plan 1–3 targeted web searches.
3. Call `web_search` for each. Prefer authoritative sources.
4. Optionally `http_request` a page to read more detail.
5. Synthesise an answer. Cite sources inline as [1], [2] and list them at the end.

Never fabricate. If sources disagree, say so. Keep the final answer under 250 words.
"""


def build_agent() -> Agent:
    """Build the research agent with its tools wired up."""
    settings = Settings()
    tools = ToolRegistry([WebSearchTool(), HTTPRequestTool()])
    return Agent(settings=settings, system=SYSTEM_PROMPT, tools=tools)


async def research(question: str) -> None:
    """Stream an answer to ``question`` to stdout."""
    console = Console()
    agent = build_agent()

    @agent.on_tool_call
    async def _announce(call: ToolCall) -> None:
        console.print(f"[dim]→ tool[/dim] [cyan]{call.name}[/cyan] {call.arguments}")

    console.rule(f"[bold]Q:[/bold] {question}")
    async for token in agent.stream(question):
        sys.stdout.write(token)
        sys.stdout.flush()
    console.print()
    console.rule(
        f"[dim]cost ${agent.ledger.total.cost_usd or 0.0:.4f}  "
        f"tokens {agent.ledger.total.total_tokens}[/dim]"
    )


def main() -> None:
    """Synchronous CLI wrapper."""
    if len(sys.argv) < 2:
        print("Usage: uv run examples/research_agent 'your question'", file=sys.stderr)
        sys.exit(2)
    question = " ".join(sys.argv[1:])
    asyncio.run(research(question))


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
