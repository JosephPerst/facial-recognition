# Agent Kit

**Ship production AI agents in a weekend, not a month.**

Agent Kit is an opinionated Python starter kit for building LLM agents. It
bundles the plumbing every agent product eventually needs — provider-swap,
tool calling, memory, streaming, cost tracking, evals, observability, HTTP
server, and deploy targets — so you can focus on your product, not your
infra.

- **Provider-agnostic** — Anthropic, OpenAI, Ollama. Switch with one env var.
- **Typed end-to-end** — Pydantic v2 models for every message, tool, and eval.
- **Async-first** — streaming, parallel tool calls, cancellation.
- **Evals built in** — JSONL golden sets, LLM-as-judge, regression compare,
  Markdown + HTML reports. Don't ship without them.
- **Batteries included** — web search, file IO, shell, HTTP tools; Chroma
  and pgvector memory; SSE streaming server.
- **Deploy anywhere** — Docker, Modal, Fly.io, Railway configs ready to go.

---

## 60-second quickstart

```bash
# 1. Install (uv manages venv + lockfile)
uv sync

# 2. Copy env + add your key
cp .env.example .env
# edit .env -> ANTHROPIC_API_KEY=sk-ant-...

# 3. Run the research agent
uv run examples/research_agent "who won the 2024 Nobel prize in physics?"

# 4. Run the coding agent (sandboxed to ./scratch)
uv run examples/coding_agent --workdir ./scratch "write a function sum(n) that sums 1..n with tests"

# 5. Start the HTTP server (SSE streaming)
uv run agent-kit-server
# → POST http://localhost:8000/v1/agent/stream  { "prompt": "..." }

# 6. Run the eval harness
uv run agent-kit-evals run examples/eval_suites/basic.jsonl
# → eval_runs/basic.{json,md,html}
```

That's it. Everything you see below is already wired.

---

## A tiny agent

```python
import asyncio
from agent_kit import Agent, Settings
from agent_kit.tools.base import tool
from agent_kit.tools.registry import ToolRegistry


@tool()
async def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def main() -> None:
    agent = Agent(
        settings=Settings(),
        system="You are a pedantic calculator.",
        tools=ToolRegistry([add]),
    )
    async for token in agent.stream("What is 17 + 25?"):
        print(token, end="", flush=True)
    print(f"\n\ncost ${agent.ledger.total.cost_usd:.4f}")


asyncio.run(main())
```

Swap to OpenAI by setting `AGENT_KIT_PROVIDER=openai`. Swap to Ollama by
setting `AGENT_KIT_PROVIDER=ollama` and `AGENT_KIT_MODEL=llama3.1`. Nothing
else changes.

---

## What you get

| Module             | What it does                                                        |
|--------------------|---------------------------------------------------------------------|
| `agent_kit.core`   | `Agent` loop, `Provider` abstraction, `Message`/`Tool*` primitives. |
| `agent_kit.tools`  | Typed `Tool` base + `@tool` decorator + built-ins.                  |
| `agent_kit.memory` | In-memory, SQLite, Chroma, pgvector; auto-summariser for long runs. |
| `agent_kit.evals`  | Golden datasets, LLM judges, regression compare, HTML reports.      |
| `agent_kit.observability` | Structured logs, OTel-shaped spans, per-call cost ledger.    |
| `examples/`        | Research agent, coding agent, FastAPI + SSE server.                 |
| `deploy/`          | `Dockerfile`, `modal_app.py`, `fly.toml`, `railway.json`.           |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design decisions, and
[EVALS.md](EVALS.md) for the eval workflow — this is the differentiator.

---

## Quality gates

```bash
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy --strict agent_kit examples tests
uv run pytest              # unit tests (no network needed)
```

CI should fail on any of the above.

---

## License

Agent Kit ships under a [commercial starter-kit license](LICENSE.md). You
get full source, unlimited projects, and modification rights. You can't
relicense or resell the template itself.

Questions? Open an issue or email the maintainer.
