"""Tests for tool base class, registry, and built-in tools."""

from __future__ import annotations

from pathlib import Path

import pytest
from agent_kit.core.messages import ToolCall
from agent_kit.tools.base import Tool, tool
from agent_kit.tools.builtin.file_io import (
    FileReadTool,
    FileWriteTool,
    PathEscapeError,
    _safe_path,
)
from agent_kit.tools.builtin.http_request import HTTPRequestTool
from agent_kit.tools.builtin.shell import ShellTool
from agent_kit.tools.registry import DuplicateToolError, ToolNotFoundError, ToolRegistry
from pydantic import BaseModel

# ---------- decorator form ----------


@tool()
async def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def test_decorator_tool_round_trip() -> None:
    assert add.name == "add"
    out = await add.invoke({"a": 1, "b": 2})
    assert out == "3"


async def test_decorator_tool_schema_has_fields() -> None:
    schema = add.json_schema()
    assert "properties" in schema
    assert "a" in schema["properties"]
    assert "b" in schema["properties"]


async def test_decorator_validates_args() -> None:
    with pytest.raises(Exception):  # noqa: B017
        await add.invoke({"a": "oops", "b": 2})


# ---------- subclass form ----------


class PingArgs(BaseModel):
    msg: str


class PingTool(Tool[PingArgs]):
    name = "ping"
    description = "echo"
    args_model = PingArgs

    async def run(self, args: PingArgs) -> str:
        return f"pong: {args.msg}"


async def test_subclass_tool_round_trip() -> None:
    out = await PingTool().invoke({"msg": "hi"})
    assert out == "pong: hi"


# ---------- registry ----------


async def test_registry_register_and_execute() -> None:
    r = ToolRegistry([add])
    result = await r.execute(ToolCall(id="1", name="add", arguments={"a": 1, "b": 4}))
    assert result.content == "5"
    assert not result.is_error


async def test_registry_duplicate_raises() -> None:
    r = ToolRegistry([add])
    with pytest.raises(DuplicateToolError):
        r.register(add)


async def test_registry_unknown_tool_returns_error() -> None:
    r = ToolRegistry()
    res = await r.execute(ToolCall(id="1", name="nope", arguments={}))
    assert res.is_error


async def test_registry_raises_on_get_missing() -> None:
    r = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        r.get("nope")


async def test_registry_execute_many_parallel() -> None:
    r = ToolRegistry([add])
    calls = [ToolCall(id=str(i), name="add", arguments={"a": i, "b": 1}) for i in range(5)]
    results = await r.execute_many(calls)
    assert [r.content for r in results] == ["1", "2", "3", "4", "5"]


# ---------- file IO ----------


async def test_file_write_read(tmp_path: Path) -> None:
    w = FileWriteTool(root=tmp_path)
    r = FileReadTool(root=tmp_path)
    await w.invoke({"path": "a.txt", "content": "hello"})
    out = await r.invoke({"path": "a.txt"})
    assert out == "hello"


async def test_file_read_escape(tmp_path: Path) -> None:
    with pytest.raises(PathEscapeError):
        _safe_path(tmp_path, "../etc/passwd")


# ---------- shell ----------


async def test_shell_runs_echo(tmp_path: Path) -> None:
    t = ShellTool(workdir=tmp_path)
    out = await t.invoke({"command": "echo hello"})
    assert "hello" in out
    assert '"exit_code": 0' in out


async def test_shell_deny_list(tmp_path: Path) -> None:
    t = ShellTool(workdir=tmp_path, deny=["rm"])
    out = await t.invoke({"command": "rm -rf /"})
    assert "denied" in out


async def test_shell_allow_list(tmp_path: Path) -> None:
    t = ShellTool(workdir=tmp_path, allow=["echo"])
    blocked = await t.invoke({"command": "true"})
    assert "allow-list" in blocked


async def test_shell_timeout(tmp_path: Path) -> None:
    t = ShellTool(workdir=tmp_path)
    out = await t.invoke({"command": "sleep 3", "timeout": 0.2})
    assert "timed out" in out


# ---------- http ----------


async def test_http_rejects_scheme() -> None:
    t = HTTPRequestTool()
    out = await t.invoke({"url": "file:///etc/passwd"})
    assert "unsupported scheme" in out
