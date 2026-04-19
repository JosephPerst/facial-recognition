# Twitter / X launch thread

## Tweet 1 (hook — this is the only one that matters)

Every AI agent project starts the same way:

week 1: streaming parser
week 2: retries, tool calls, cost tracker
week 3: a memory layer
week 4: "should we write evals?" (no)
week 5: ship something half-broken

I built the thing that skips to week 5. Agent Kit. 👇

---

## Tweet 2

It's a Python 3.12 starter kit. Opinionated, typed, async, tested.

Anthropic / OpenAI / Ollama — one env var to swap.
Streaming, parallel tool calls, retries, cost caps — done.
FastAPI + SSE server — done.
Deploy configs for Docker / Modal / Fly / Railway — done.

[screenshot: `examples/my_agent.py` code block]

---

## Tweet 3

The unfair advantage: the eval harness.

JSONL golden datasets.
LLM-as-judge (correctness / helpfulness / safety).
Regression compare against a baseline.
Self-contained HTML reports you can paste into a PR.

Writing evals is the step every team skips. This makes skipping harder than doing.

[screenshot: `eval_runs/basic.html`]

---

## Tweet 4

Tools = typed pydantic models.

```python
@tool()
async def run_sql(query: str, db: str = "main") -> list[dict]:
    ...
```

Decorator introspects the signature, generates the JSON schema, validates the LLM's args before your code ever runs. One source of truth.

---

## Tweet 5

Memory scales with you:

- in-memory (dev)
- SQLite (single-host prod)
- Chroma (local RAG)
- pgvector (real prod)

+ an auto-summariser that rolls up old turns when context gets long, so your bills don't explode on turn 40.

---

## Tweet 6

Quality gates that actually ran on this repo:

✅ `uv sync && uv run pytest` — 37 tests pass, no API key needed
✅ `mypy --strict` — zero errors across 42 files
✅ `ruff check` — clean
✅ 60-second quickstart that actually works

Not a demo. A product.

---

## Tweet 7

Who this is for:

→ indie devs shipping AI SaaS
→ consultants billing $10k for an agent who want to skip week 1
→ startup engineers who need a working demo by Friday

Who it's NOT for:

→ no-code / flow-builder crowd
→ multi-agent research (use LangGraph)

---

## Tweet 8 (CTA)

$99 one-time. Lifetime code. 12 months of updates. 14-day refund.

Team license (5 seats) is $399.

{{LANDING_URL}}

Built by {{HANDLE}}. Reply with Qs, I'll answer all of them.

---

## Tweet 9 (optional — pin this after launch if it gets traction)

Pre-orders so far: {{NUMBER}}.
Ship notes, eval deltas, and "things I learned selling a starter kit on Twitter" going into the Discord. Link for buyers.

---

**Thread-writing notes:**
- Thread length 7–9 tweets hits the sweet spot.
- Tweet 1 = 90% of the engagement. Iterate on it before posting.
- Include ≥2 screenshots. Code is always ignored without an image next to it.
- Post 9–11am in your target buyer's timezone (PT for devs).
- Reply to *every* reply for the first 4 hours. Algorithm rewards it.
- Don't tweet a discount in the launch thread — save for day 3 "last
  chance $79" push if momentum stalls.
