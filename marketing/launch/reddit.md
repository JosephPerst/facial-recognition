# Reddit posts

## r/SideProject

**Title:** I got tired of rebuilding agent plumbing every time, so I built a starter kit and I'm selling it for $99

**Body:**

Every time I started a new AI agent project it was the same week of
plumbing: streaming, retries, tool-call schemas, a toy memory layer, a
cost tracker, an eval loop bolted on at the end. Decided to build it once
properly and sell the template.

It's called Agent Kit. Python 3.12, Pydantic v2, FastAPI, async
everywhere. Swaps between Anthropic / OpenAI / Ollama with one env var.
Ships with a real eval harness (LLM-as-judge + regression compare + HTML
reports), sandboxed tool built-ins, SQLite + pgvector memory, and deploy
configs for Docker / Modal / Fly / Railway.

37 pytest tests green, `mypy --strict` clean, `ruff` clean.

**Why I'm posting here:** before I build more I want to know if this
saves anyone else the two weeks it saved me. If you've ever thought
"ugh, I don't want to wire up streaming and tool calls again," this is
for you.

$99 one-time, $399 for a 5-seat team license, 14-day refund.

Link: {{LANDING_URL}}

Roast me / ask me anything.

---

## r/LocalLLaMA

**Title:** Python agent starter kit that runs against Ollama (and Anthropic / OpenAI) with identical code

**Body:**

Built a Python starter kit for AI agents where the provider is swappable
with one env var — so you can dev against local Llama via Ollama and flip
to Claude / GPT in prod (or vice versa). Same `Agent` class, same tool
interface, same streaming API.

Other stuff that matters for local-LLM folks:

- No OpenAI lock-in. Ollama path is fully tested, not a bolt-on.
- Cost tracking correctly reports $0 for Ollama (still tracks tokens).
- Built-in tools: web search (DuckDuckGo HTML, no API key), file IO,
  sandboxed shell, HTTP request. All typed via Pydantic.
- Evals harness doesn't care which provider generated the output.
- FastAPI + SSE server, Docker / Modal / Fly / Railway deploy configs.

$99 one-time license. Not OSS because I want to actually maintain it
for a while. Commercial use included.

Link: {{LANDING_URL}}

---

**Notes before posting:**
- r/SideProject — post on self-promo day, lead with the problem not the
  price.
- r/LocalLLaMA — lean into provider-agnostic angle, don't bury the
  Ollama support.
- Do NOT cross-post identical text; each sub gets its own framing.
- r/MachineLearning has strict self-promo rules — only post if you can
  frame as a technical write-up.
