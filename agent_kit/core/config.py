"""Environment-driven configuration for Agent Kit.

Every knob that a user might flip from ops (provider choice, limits, memory
backend, log level) lives here. Values can come from constructor kwargs,
environment variables, or a `.env` file — in that order.

Example:
    >>> from agent_kit.core.config import Settings
    >>> s = Settings(provider="openai", model="gpt-5")  # doctest: +SKIP
    >>> s.provider  # doctest: +SKIP
    'openai'
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["anthropic", "openai", "ollama"]
MemoryBackend = Literal["inmemory", "sqlite", "chroma", "pgvector"]


class Settings(BaseSettings):
    """Runtime configuration for Agent Kit, sourced from env vars.

    All fields are overridable via constructor kwargs. Provider API keys are
    wrapped in ``SecretStr`` so they don't leak into logs or repr strings.

    Example:
        >>> Settings(provider="anthropic", model="claude-sonnet-4-6").model
        'claude-sonnet-4-6'
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AGENT_KIT_",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Provider & model ---
    provider: ProviderName = "anthropic"
    model: str = "claude-sonnet-4-6"
    judge_model: str = "claude-sonnet-4-6"

    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")

    # --- Agent runtime limits ---
    max_iterations: int = 25
    max_cost_usd: float = 5.0
    max_tokens: int = 4096
    request_timeout: float = 120.0
    temperature: float = 0.7

    # --- Retry behaviour ---
    retry_max_attempts: int = 5
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 30.0

    # --- Memory ---
    memory_backend: MemoryBackend = "inmemory"
    sqlite_path: Path = Path("./.agent_kit/conversations.db")
    chroma_path: Path = Path("./.agent_kit/chroma")
    pgvector_dsn: str = "postgresql://postgres:postgres@localhost:5432/agent_kit"
    memory_summarize_after_tokens: int = 8000

    # --- Observability ---
    log_level: str = "INFO"
    log_json: bool = False
    trace_file: Path | None = Path("./.agent_kit/traces.jsonl")

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        """Ensure log levels are upper-case."""
        return v.upper()

    def anthropic_key(self) -> str | None:
        """Return the Anthropic API key as a plain string, if set."""
        return self.anthropic_api_key.get_secret_value() if self.anthropic_api_key else None

    def openai_key(self) -> str | None:
        """Return the OpenAI API key as a plain string, if set."""
        return self.openai_api_key.get_secret_value() if self.openai_api_key else None


__all__ = ["MemoryBackend", "ProviderName", "Settings"]
