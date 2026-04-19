# syntax=docker/dockerfile:1.7
# Multi-stage build: builder installs deps with uv; runner ships a thin image.

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml .python-version ./
COPY agent_kit ./agent_kit
COPY examples ./examples
COPY docs ./docs
RUN uv sync --no-dev --frozen || uv sync --no-dev

FROM python:3.12-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    AGENT_KIT_HOST=0.0.0.0 \
    AGENT_KIT_PORT=8000

RUN useradd --create-home --uid 10001 agent
WORKDIR /app

COPY --from=builder --chown=agent:agent /app /app

USER agent
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "examples.server:app", "--host", "0.0.0.0", "--port", "8000"]
