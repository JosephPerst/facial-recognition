"""Sandboxed file-read / file-write tools.

Both tools refuse any path outside their ``root`` directory so an agent
cannot poke around ``/etc/`` or overwrite ``/home``.

Example:
    >>> import tempfile, asyncio
    >>> from agent_kit.tools.builtin.file_io import FileWriteTool, FileReadTool
    >>> with tempfile.TemporaryDirectory() as d:
    ...     _ = asyncio.run(FileWriteTool(root=d).invoke(
    ...         {"path": "note.txt", "content": "hi"}
    ...     ))
    ...     out = asyncio.run(FileReadTool(root=d).invoke({"path": "note.txt"}))
    >>> "hi" in out
    True
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from agent_kit.tools.base import Tool

_MAX_READ_BYTES = 1_000_000  # 1 MiB


class PathEscapeError(ValueError):
    """Raised when a requested path escapes the sandbox root."""


def _safe_path(root: Path, rel: str) -> Path:
    """Resolve ``rel`` against ``root`` and reject traversal outside root."""
    root = root.resolve()
    candidate = (root / rel).resolve()
    if root != candidate and root not in candidate.parents:
        raise PathEscapeError(f"{rel!r} escapes sandbox root {root}")
    return candidate


class FileReadArgs(BaseModel):
    """Arguments for :class:`FileReadTool`."""

    path: str = Field(..., description="Path relative to the sandbox root.")
    max_bytes: int = Field(_MAX_READ_BYTES, ge=1, le=_MAX_READ_BYTES)


class FileReadTool(Tool[FileReadArgs]):
    """Read a UTF-8 text file from within a sandbox root."""

    name: ClassVar[str] = "file_read"
    description: ClassVar[str] = "Read a UTF-8 text file relative to the agent's working directory."
    args_model: ClassVar[type[BaseModel]] = FileReadArgs

    def __init__(self, root: str | Path = ".") -> None:
        """Create the tool with a root directory constraint."""
        self.root = Path(root)

    async def run(self, args: FileReadArgs) -> str:
        """Return the file contents as a string."""
        target = _safe_path(self.root, args.path)
        if not target.exists():
            raise FileNotFoundError(args.path)
        if not target.is_file():
            raise IsADirectoryError(args.path)
        data = target.read_bytes()[: args.max_bytes]
        return data.decode("utf-8", errors="replace")


class FileWriteArgs(BaseModel):
    """Arguments for :class:`FileWriteTool`."""

    path: str = Field(..., description="Path relative to the sandbox root.")
    content: str = Field(..., description="UTF-8 content to write.")
    append: bool = Field(False, description="Append to file instead of overwriting.")


class FileWriteTool(Tool[FileWriteArgs]):
    """Write a UTF-8 text file inside a sandbox root."""

    name: ClassVar[str] = "file_write"
    description: ClassVar[str] = (
        "Create or overwrite a UTF-8 text file inside the agent's working "
        "directory. Use append=true to append instead of overwrite."
    )
    args_model: ClassVar[type[BaseModel]] = FileWriteArgs

    def __init__(self, root: str | Path = ".") -> None:
        """Create the tool with a root directory constraint."""
        self.root = Path(root)

    async def run(self, args: FileWriteArgs) -> dict[str, object]:
        """Write to the file and return byte/ line counts."""
        target = _safe_path(self.root, args.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append else "w"
        with target.open(mode, encoding="utf-8") as f:
            f.write(args.content)
        return {
            "path": str(target.relative_to(self.root.resolve())),
            "bytes": len(args.content.encode("utf-8")),
            "mode": mode,
        }


__all__ = [
    "FileReadArgs",
    "FileReadTool",
    "FileWriteArgs",
    "FileWriteTool",
    "PathEscapeError",
]
