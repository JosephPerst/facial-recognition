"""Modal deployment: exposes the FastAPI server behind a Modal web endpoint.

Deploy:

    modal deploy deploy/modal_app.py

Then hit the URL printed by Modal. Environment variables configured in Modal
override `.env` values.
"""

from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .workdir("/app")
    .add_local_dir(".", remote_path="/app", copy=True)
    .run_commands("uv sync --no-dev")
)

app = modal.App("agent-kit", image=image)

SECRETS = [
    modal.Secret.from_name("agent-kit", required_keys=["ANTHROPIC_API_KEY"]),
]


@app.function(
    secrets=SECRETS,
    timeout=600,
    memory=1024,
    min_containers=0,
)
@modal.asgi_app()
def fastapi_app():
    """Return the FastAPI ASGI app for Modal to serve."""
    from examples.server import app as api

    return api
