# Show HN: Agent Kit — production Python starter for AI agents ($99)

**Title (≤80 chars):**
`Show HN: Agent Kit – Python starter for AI agents (provider-swap, evals, deploy)`

**URL:** `{{LANDING_URL}}`

**Body:**

Hey HN, I built Agent Kit because every time I (or a friend) wanted to ship
an AI agent product, the first two weeks were the same miserable plumbing:
provider-specific streaming parsers, retry loops, cost tracking, tool-call
validation, a toy memory layer, and — if you were conscientious — an eval
harness bolted on at the end.

It's a typed, opinionated Python 3.12 starter kit:

- **Core:** async Agent loop with provider-swap (Anthropic / OpenAI / Ollama),
  streaming, parallel tool calls via `asyncio.gather`, retries with
  exponential backoff, hard `max_cost_usd` / `max_iterations` caps, lifecycle
  hooks (`on_message`, `on_tool_call`, `on_tool_result`, `on_error`).
- **Tools:** a `Tool` ABC + `@tool` decorator that generates the JSON schema
  from a Pydantic args model. Five sandboxed built-ins (web search, file read
  / write, shell with allow-/deny-lists, HTTP).
- **Memory:** in-memory → SQLite → Chroma → pgvector, plus an
  auto-summariser that rolls up old turns when the context gets long.
- **Evals (the reason this exists):** JSONL datasets, LLM-as-judge for
  correctness / helpfulness / safety, a `StringMatchJudge` for deterministic
  smoke tests, parallel runner, regression compare vs. a baseline run,
  standalone Markdown + HTML reports you can drop into a PR.
- **Observability:** structlog + OpenTelemetry-shaped spans, JSONL trace
  sink, per-(provider, model) pricing table feeding a thread-safe cost
  ledger.
- **Server:** FastAPI wrapper with Server-Sent Events streaming, session
  memory keyed by `session_id`, optional `X-Api-Key` auth.
- **Deploy:** Dockerfile, Modal app, Fly.io toml, Railway json.

Everything passes `mypy --strict`, `ruff check`, and a real pytest suite
(37 tests, no network needed).

It's sold as a one-time $99 starter kit (team seat $399) under a
commercial license. I'm posting here to validate whether this is
actually useful — happy to answer questions about design trade-offs
(why these providers, why Chroma + pgvector, how the eval compare
works, etc.) or get roasted for the ones I got wrong.

Docs / demo: {{LANDING_URL}}

---

**Notes before posting:**
- Post Tue–Thu, 8–10am PT (best HN traffic window).
- Respond to every comment within 2 hours for the first 6 hours.
- Do NOT upvote your own post or ask others to.
- Expect harsh feedback on: pricing, license ("why not OSS?"),
  evals-as-moat framing, OTel dependency absence. Have answers ready.
