# Architecture

This document explains the *shape* of Agent Kit and the trade-offs we took.

## Design goals

1. **One brain, many bodies.** The agent loop doesn't know or care which
   provider, memory, or tool runtime you're using. That's why `Agent`
   depends only on `Provider`, `ToolRegistry`, and `Memory` abstractions.
2. **Typed end-to-end.** Every message is a Pydantic model. Every tool
   call is a Pydantic model. Every eval case is a Pydantic model. This is
   what lets `mypy --strict` pass without `# type: ignore` noise.
3. **Streaming, parallel, async-first.** The loop runs `provider.stream()`
   and fans out tool execution via `asyncio.gather`. No thread pools, no
   sync shims.
4. **Budgets, not billing surprises.** Every turn updates a `CostLedger`
   that the agent enforces against `max_cost_usd`.
5. **Evals as a first-class product.** A "starter kit" without evals is a
   demo. See `EVALS.md`.

## Module map

```
agent_kit/
├── core/
│   ├── messages.py       # Message, ToolCall, ToolResult, Usage
│   ├── config.py         # Settings (env-driven, pydantic-settings)
│   ├── provider.py       # Provider ABC + Anthropic/OpenAI/Ollama impls
│   └── agent.py          # Agent loop, hooks, budgets, streaming
├── tools/
│   ├── base.py           # Tool ABC + @tool decorator (schema from pydantic)
│   ├── registry.py       # ToolRegistry: register, validate, parallel exec
│   └── builtin/          # web_search, file_read, file_write, shell, http
├── memory/
│   ├── base.py           # Memory + VectorMemory ABCs
│   ├── conversation.py   # InMemoryConversation, SQLiteConversation
│   ├── vector.py         # ChromaVectorMemory, PgVectorMemory
│   └── summarizer.py     # AutoSummarizingMemory (rollup past turns)
├── evals/
│   ├── datasets.py       # EvalCase / EvalDataset, JSONL loader
│   ├── judges.py         # StringMatch + Correctness/Helpfulness/Safety
│   ├── harness.py        # EvalHarness, CLI, regression compare
│   ├── metrics.py        # pass rate, percentiles, cost
│   └── reports.py        # Markdown + HTML + console rendering
└── observability/
    ├── tracing.py        # structlog + OTel-shaped Span type
    └── cost.py           # per-(provider, model) price table + ledger
```

## Message flow (happy path)

```
user prompt  ──▶  Memory.append(user)
                  │
                  ▼
            ┌──────────────┐
            │  Agent.run   │◀──── on_message hooks
            └──────────────┘
                  │
                  ▼
            Provider.complete(history, tools)
                  │
                  ▼
        assistant msg (maybe with tool_calls)
                  │
                  ▼                      ┌──────────────┐
         any tool_calls? ─── yes ───▶    │ ToolRegistry │── parallel
                  │                      │  .execute    │    asyncio.gather
                  │ no                   └──────────────┘
                  ▼                             │
              final text                        ▼
                                          tool results
                                                │
                                                ▼
                                      Memory.append(tool)
                                                │
                                                ▼
                                          (loop)
```

## Why these choices

- **Anthropic, OpenAI, Ollama.** Covers the "frontier closed-source,
  cheapest closed-source, self-hosted OSS" triangle. Adding Mistral or
  Groq is ~50 lines.
- **Pydantic for tool schemas.** The provider expects JSON Schema; we
  generate it from the same `args_model` we validate with. One source of
  truth.
- **Chroma locally, pgvector in prod.** Chroma has zero-config persistence
  for weekend builds. Postgres + pgvector is the safe default for production
  because you already run it for everything else.
- **SSE for the HTTP server.** Simpler than WebSockets for pure
  server-push, and every browser speaks it natively.
- **Structured logs + OTel-shaped spans.** We don't import the OTel SDK
  (heavy, evolving), but our span dict is a strict subset so you can ship
  it through an OTel collector with a 20-line bridge.
- **Tenacity for retries.** Exponential backoff, retry-on-subclass,
  battle-tested. Don't reinvent this.

## What's *not* here (on purpose)

- **Agent orchestration / multi-agent.** We'd rather you compose multiple
  `Agent` instances yourself than bake in a half-baked graph framework.
- **A UI.** You have React / htmx / Phoenix LiveView. The HTTP server
  gives you a clean API to plug into.
- **Vendor-locked deploy.** Modal / Fly / Railway configs ship, but
  nothing depends on any of them.

## Extending

- **New provider:** implement `Provider.complete` + `Provider.stream`,
  add an entry to `get_provider`.
- **New tool:** subclass `Tool` or decorate with `@tool`. Register via
  `ToolRegistry`.
- **New judge:** subclass `Judge`, return a `JudgeResult`.
- **New memory backend:** subclass `Memory` or `VectorMemory`.

Every extension point is an ABC; every concrete class is ≤ ~200 LOC.
