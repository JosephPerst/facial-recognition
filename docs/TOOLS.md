# Writing Custom Tools

Tools are the agent's only way to affect the outside world. Agent Kit gives
you two ways to define one — the right one depends on how structured the
tool's arguments are.

## Option A: the `@tool` decorator (fastest)

```python
from agent_kit.tools.base import tool

@tool()
async def geocode(address: str, country: str = "US") -> dict[str, float]:
    """Convert a postal address into lat/lon. Returns {"lat": ..., "lng": ...}."""
    ...
```

The decorator introspects your function signature and creates a Pydantic
args model behind the scenes:

- Parameter types → schema types (strict validation at call time).
- Default values → schema defaults.
- Docstring → tool description (so the LLM knows what it does).
- Extra args that weren't in the schema → `ValidationError` before your
  code ever runs.

Use this style when your tool takes scalar / simple-dict arguments.

## Option B: subclass `Tool` (when args have structure)

```python
from pydantic import BaseModel, Field
from agent_kit.tools.base import Tool

class RunSqlArgs(BaseModel):
    """Arguments for the run_sql tool."""
    query: str = Field(..., description="A single SQL SELECT statement.")
    database: str = Field("primary", description="Connection name to use.")
    timeout_s: float = Field(30.0, ge=1.0, le=120.0)

class RunSqlTool(Tool[RunSqlArgs]):
    name = "run_sql"
    description = "Execute a read-only SQL query and return rows as JSON."
    args_model = RunSqlArgs

    def __init__(self, pool):
        self.pool = pool

    async def run(self, args: RunSqlArgs) -> list[dict]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(args.query)
            rows = await cur.fetchall()
            return [dict(r) for r in rows]
```

This style buys you:

- A dedicated `BaseModel` you can reuse in tests.
- `Field(...)` descriptions that the LLM sees in the schema.
- Validators, conditional types, discriminated unions, everything pydantic.

## Registering tools

```python
from agent_kit.tools.registry import ToolRegistry
from agent_kit.tools.builtin import WebSearchTool, HTTPRequestTool

registry = ToolRegistry([
    WebSearchTool(),
    HTTPRequestTool(),
    RunSqlTool(pool=my_pool),
    geocode,  # decorator-built tool is already a Tool instance
])

agent = Agent(tools=registry)
```

Duplicate names throw `DuplicateToolError`. Missing tools return structured
errors in the tool message so the model can recover.

## Parallel execution & error handling

When the model emits multiple tool calls in one turn, the registry runs
them via `asyncio.gather`. Failures in one tool do **not** fail the other
calls — the failing one comes back as `is_error=True` so the model can
decide how to react.

```python
# Under the hood, effectively:
results = await asyncio.gather(*[registry.execute(c) for c in calls])
```

## Hooks: watch or inject side effects

```python
@agent.on_tool_call
async def audit_call(call):
    await db.insert("audit", {"tool": call.name, "args": call.arguments})

@agent.on_tool_result
async def track_result(call, result):
    metrics.increment(f"tool.{call.name}.{'error' if result.is_error else 'ok'}")
```

## Safety defaults

- **File IO tools** are sandboxed to a `root` directory; path escapes raise.
- **Shell tool** ships with a deny-list of destructive commands
  (`rm`, `mkfs`, `shutdown`, `reboot`). Override via the constructor.
- **HTTP tool** rejects non-`http(s)` schemes and lets you pass a host
  deny-list.

Always think about the worst case: what happens if the model calls your
tool with the scariest argument it can dream up? If you can't answer, add
guardrails before you ship.

## Testing tools

The Agent Kit test suite shows the pattern (`tests/test_tools.py`):

```python
async def test_my_tool() -> None:
    out = await MyTool().invoke({"arg": "value"})
    assert "expected" in out
```

`invoke` returns a string because that's what goes back into the model's
context. If you want strongly-typed results for your own code paths, call
`await tool.run(args)` directly.
