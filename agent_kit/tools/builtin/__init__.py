"""Built-in tools: web search, file IO, shell, HTTP."""

from __future__ import annotations

from typing import Any

from agent_kit.tools.base import Tool
from agent_kit.tools.builtin.file_io import FileReadTool, FileWriteTool
from agent_kit.tools.builtin.http_request import HTTPRequestTool
from agent_kit.tools.builtin.shell import ShellTool
from agent_kit.tools.builtin.web_search import WebSearchTool


def default_tools(workdir: str = ".") -> list[Tool[Any]]:
    """Return a list of default tool instances rooted at ``workdir``.

    Example:
        >>> tools = default_tools(".")
        >>> len(tools) >= 5
        True
    """
    return [
        WebSearchTool(),
        FileReadTool(root=workdir),
        FileWriteTool(root=workdir),
        ShellTool(workdir=workdir),
        HTTPRequestTool(),
    ]


__all__ = [
    "FileReadTool",
    "FileWriteTool",
    "HTTPRequestTool",
    "ShellTool",
    "WebSearchTool",
    "default_tools",
]
