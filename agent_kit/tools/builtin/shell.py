"""Shell-exec tool with timeout and optional command allow-list.

By default the tool will run any command. Production users should override
``allow`` (a list of allowed argv[0] binaries) or set
``deny`` (a list of argv[0] binaries that should never run). An agent hitting
a denied command receives a structured error string, not an exception.

Example:
    >>> import asyncio
    >>> t = ShellTool()
    >>> out = asyncio.run(t.invoke({"command": "echo hi"}))
    >>> "hi" in out
    True
"""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from agent_kit.tools.base import Tool


class ShellArgs(BaseModel):
    """Arguments for :class:`ShellTool`."""

    command: str = Field(..., description="Shell command to execute (argv-style).")
    timeout: float = Field(60.0, ge=0.1, le=600.0, description="Timeout in seconds.")


class ShellResult(BaseModel):
    """Structured result returned by :class:`ShellTool`."""

    exit_code: int
    stdout: str
    stderr: str
    command: str


class ShellTool(Tool[ShellArgs]):
    """Execute a shell command with a timeout and optional filtering."""

    name: ClassVar[str] = "shell"
    description: ClassVar[str] = (
        "Run a shell command. Returns {exit_code, stdout, stderr}. "
        "Prefer specific argv commands over shell pipelines when possible."
    )
    args_model: ClassVar[type[BaseModel]] = ShellArgs

    def __init__(
        self,
        workdir: str | Path = ".",
        allow: list[str] | None = None,
        deny: list[str] | None = None,
        max_output_bytes: int = 100_000,
    ) -> None:
        """Create the shell tool.

        Args:
            workdir: Working directory for commands.
            allow: Optional allow-list of argv[0] binaries; if set, any other
                command is rejected.
            deny: Optional deny-list of argv[0] binaries that are never run.
            max_output_bytes: Maximum bytes to return per stdout/stderr stream.
        """
        self.workdir = Path(workdir)
        self.allow = set(allow) if allow else None
        self.deny = set(deny or {"rm", "mkfs", "shutdown", "reboot"})
        self.max_output_bytes = max_output_bytes

    async def run(self, args: ShellArgs) -> dict[str, object]:
        """Execute the command subject to allow/deny policy."""
        parts = shlex.split(args.command)
        if not parts:
            return ShellResult(
                exit_code=1,
                stdout="",
                stderr="empty command",
                command=args.command,
            ).model_dump()
        head = Path(parts[0]).name
        if self.allow is not None and head not in self.allow:
            return ShellResult(
                exit_code=126,
                stdout="",
                stderr=f"command {head!r} not in allow-list",
                command=args.command,
            ).model_dump()
        if head in self.deny:
            return ShellResult(
                exit_code=126,
                stdout="",
                stderr=f"command {head!r} is denied by policy",
                command=args.command,
            ).model_dump()

        proc = await asyncio.create_subprocess_shell(
            args.command,
            cwd=str(self.workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=args.timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return ShellResult(
                exit_code=124,
                stdout="",
                stderr=f"timed out after {args.timeout}s",
                command=args.command,
            ).model_dump()
        return ShellResult(
            exit_code=proc.returncode or 0,
            stdout=_clip(stdout_b, self.max_output_bytes),
            stderr=_clip(stderr_b, self.max_output_bytes),
            command=args.command,
        ).model_dump()


def _clip(data: bytes, limit: int) -> str:
    """Decode and truncate output bytes to ``limit`` bytes."""
    if len(data) <= limit:
        return data.decode("utf-8", errors="replace")
    head = data[:limit].decode("utf-8", errors="replace")
    return head + f"\n... [{len(data) - limit} more bytes truncated]"


__all__ = ["ShellArgs", "ShellResult", "ShellTool"]
