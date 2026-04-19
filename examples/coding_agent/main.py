"""A minimal coding agent.

The agent is sandboxed to a working directory. It can:

* read files (``file_read``),
* write files (``file_write``),
* run shell commands (``shell``) such as ``pytest`` or ``ruff``.

Run:

    uv run examples/coding_agent --workdir ./scratch \\
        "add a function add(a, b) with tests in scratch/"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from agent_kit import Agent, Settings
from agent_kit.core.messages import ToolCall, ToolResult
from agent_kit.tools.builtin import FileReadTool, FileWriteTool, ShellTool
from agent_kit.tools.registry import ToolRegistry
from rich.console import Console

SYSTEM_PROMPT = """You are a careful coding assistant.

Ground rules:
* You MAY read files, write files, and run shell commands — all sandboxed to
  the working directory provided.
* Before writing, skim the existing files so you don't clobber structure.
* After writing code, run tests (pytest) or lints (ruff) when available.
* Keep changes minimal. Don't rewrite files you didn't need to touch.
* When done, summarise what you changed and any follow-ups.
"""


def build_agent(workdir: Path) -> Agent:
    """Build the coding agent with sandboxed IO tools."""
    workdir.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    tools = ToolRegistry(
        [
            FileReadTool(root=workdir),
            FileWriteTool(root=workdir),
            ShellTool(workdir=workdir),
        ]
    )
    return Agent(settings=settings, system=SYSTEM_PROMPT, tools=tools)


async def code(task: str, workdir: Path) -> None:
    """Run the coding agent on ``task`` inside ``workdir``."""
    console = Console()
    agent = build_agent(workdir)

    @agent.on_tool_call
    async def _log_call(call: ToolCall) -> None:
        console.print(f"[dim]→[/dim] [cyan]{call.name}[/cyan] {call.arguments}")

    @agent.on_tool_result
    async def _log_result(call: ToolCall, result: ToolResult) -> None:
        head = result.content if len(result.content) < 200 else result.content[:200] + "…"
        tone = "red" if result.is_error else "green"
        console.print(f"[{tone}]← {call.name}[/{tone}] {head}")

    console.rule(f"[bold]Task:[/bold] {task}")
    async for token in agent.stream(task):
        sys.stdout.write(token)
        sys.stdout.flush()
    console.print()
    console.rule(
        f"[dim]iterations {agent.ledger.total.total_tokens} tokens  "
        f"cost ${agent.ledger.total.cost_usd or 0.0:.4f}[/dim]"
    )


def main() -> None:
    """Synchronous CLI wrapper."""
    parser = argparse.ArgumentParser(description="Agent Kit coding agent")
    parser.add_argument("task", nargs="+")
    parser.add_argument("--workdir", type=Path, default=Path("./scratch"))
    args = parser.parse_args()
    asyncio.run(code(" ".join(args.task), args.workdir))


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
