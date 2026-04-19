# Deploying Agent Kit

All four deploy targets wrap the same `examples.server:app` FastAPI
application. Pick the one that matches your preference:

## Docker (anywhere)

```bash
docker build -f deploy/Dockerfile -t agent-kit .
docker run --rm -p 8000:8000 \
    -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
    agent-kit
```

- Multi-stage build keeps the runtime image < 300 MB.
- Runs as UID 10001 (`agent`), not root.
- Health check probes `/healthz` every 30s.

## Modal

```bash
pip install modal
modal setup  # one-time
modal secret create agent-kit ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
modal deploy deploy/modal_app.py
```

Modal will print the public URL. Scales to zero by default (`min_containers=0`).

## Fly.io

```bash
fly launch --copy-config --dockerfile deploy/Dockerfile
fly secrets set ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
fly deploy
```

- Auto-stop / auto-start keeps cost near zero for low-traffic apps.
- Scales to 1 machine on first request.
- Edit `[[vm]]` block in `deploy/fly.toml` for larger instances.

## Railway

Connect the GitHub repo in the Railway UI and it'll pick up
`deploy/railway.json` automatically. Set `ANTHROPIC_API_KEY` in the
environment variables tab. Health checks and restart policy are already
wired.

## Environment variables (production must-set)

| Variable                   | Required | Description                                  |
|----------------------------|----------|----------------------------------------------|
| `ANTHROPIC_API_KEY`        | one of   | Anthropic provider key                       |
| `OPENAI_API_KEY`           | these    | OpenAI provider key                          |
| `OLLAMA_HOST`              |          | Ollama URL (if `AGENT_KIT_PROVIDER=ollama`)  |
| `AGENT_KIT_PROVIDER`       | no       | `anthropic` (default), `openai`, or `ollama` |
| `AGENT_KIT_MODEL`          | no       | Model ID; default `claude-sonnet-4-6`        |
| `AGENT_KIT_MAX_COST_USD`   | no       | Hard budget per request                      |
| `AGENT_KIT_MAX_ITERATIONS` | no       | Max provider turns per request               |
| `AGENT_KIT_MEMORY_BACKEND` | no       | `inmemory` / `sqlite` / `chroma` / `pgvector`|
| `AGENT_KIT_PGVECTOR_DSN`   | if used  | pgvector connection string                   |
| `AGENT_KIT_LOG_JSON`       | no       | `true` for JSON logs (pipe to Loki / DataDog)|
| `AGENT_KIT_API_KEYS`       | no       | Comma-separated allow-list for `X-Api-Key`   |

## Going to prod checklist

- [ ] `AGENT_KIT_API_KEYS` set (or wire your own auth in front).
- [ ] `AGENT_KIT_MEMORY_BACKEND=pgvector` with a real DB, **not** SQLite,
      so multiple replicas share conversations.
- [ ] `AGENT_KIT_LOG_JSON=true` and ship logs to a central collector.
- [ ] Set conservative `AGENT_KIT_MAX_COST_USD` per request (e.g. `0.50`).
- [ ] Expose `/healthz` to your load balancer.
- [ ] Rate-limit in front of the service (Fly Cloudflare, Modal's rate
      limiter, or a sidecar proxy).
- [ ] Run the eval suite in CI before every deploy.

## Vertical scaling hints

The agent loop is IO-bound. A single 1-vCPU machine happily holds
100+ concurrent streaming sessions thanks to `asyncio`. The bottleneck is
almost always the upstream provider's rate limit.

For horizontal scaling, back the `Memory` with pgvector (or any shared
store) so any replica can resume any session.
