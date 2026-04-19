"""Structured logging + OpenTelemetry-compatible span tracing.

We don't take a hard dependency on the full OTel SDK, but the shape of our
``Span`` object (start/end times, attributes, events, status) is a strict
subset of the OTel spec so it's trivial to swap in an OTel exporter later.

Example:
    >>> tracer = get_tracer()
    >>> with tracer.span("demo", {"k": "v"}) as span:
    ...     span.set_attribute("extra", 1)
    >>> tracer.completed[-1].name
    'demo'
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import structlog
from structlog.typing import Processor

_logger_configured = False
_logger_lock = threading.Lock()


def _configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure structlog once per process."""
    global _logger_configured
    with _logger_lock:
        if _logger_configured:
            return
        processors: list[Processor] = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        if json_logs:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()))
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
            cache_logger_on_first_use=True,
        )
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stderr,
            level=getattr(logging, level),
        )
        _logger_configured = True


def get_logger(
    name: str = "agent_kit",
    level: str = "INFO",
    json_logs: bool = False,
) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog logger.

    Example:
        >>> log = get_logger()
        >>> hasattr(log, "info")
        True
    """
    _configure_logging(level=level, json_logs=json_logs)
    return structlog.get_logger(name)  # type: ignore[no-any-return]


SpanStatus = Literal["ok", "error"]


@dataclass
class Span:
    """A single unit of traced work.

    Attributes match OpenTelemetry's span model so this can be adapted to an
    OTel exporter with a thin bridge.
    """

    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start_time_ns: int
    end_time_ns: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: SpanStatus = "ok"
    error: str | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach an arbitrary attribute to the span."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Record an event on the span."""
        self.events.append(
            {
                "name": name,
                "time_ns": time.time_ns(),
                "attributes": attributes or {},
            }
        )

    @property
    def duration_ms(self) -> float | None:
        """Duration in milliseconds once the span has ended."""
        if self.end_time_ns is None:
            return None
        return (self.end_time_ns - self.start_time_ns) / 1_000_000

    def to_dict(self) -> dict[str, Any]:
        """Serialise the span to a JSON-compatible dict."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
            "error": self.error,
        }


class Tracer:
    """In-process span collector with optional JSONL sink.

    Example:
        >>> t = Tracer()
        >>> with t.span("unit"):
        ...     pass
        >>> len(t.completed)
        1
    """

    def __init__(
        self,
        sink: Path | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        """Create a new tracer.

        Args:
            sink: JSONL file that receives each completed span, if provided.
            logger: Optional structlog logger; defaults to ``get_logger()``.
        """
        self.sink = sink
        self.logger = logger or get_logger()
        self.completed: list[Span] = []
        self._stack: list[Span] = []
        self._lock = threading.Lock()
        if self.sink is not None:
            self.sink.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        """Start a new span that auto-closes when the context exits."""
        parent = self._stack[-1] if self._stack else None
        span = Span(
            name=name,
            trace_id=parent.trace_id if parent else uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
            start_time_ns=time.time_ns(),
            attributes=dict(attributes or {}),
        )
        with self._lock:
            self._stack.append(span)
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end_time_ns = time.time_ns()
            with self._lock:
                self._stack.pop()
                self.completed.append(span)
            self._emit(span)

    def _emit(self, span: Span) -> None:
        """Flush a completed span to logger and JSONL sink."""
        self.logger.info(
            "span.completed",
            span=span.name,
            duration_ms=span.duration_ms,
            status=span.status,
            **{f"attr.{k}": v for k, v in span.attributes.items()},
        )
        if self.sink is not None:
            with self.sink.open("a", encoding="utf-8") as f:
                f.write(json.dumps(span.to_dict(), default=str) + "\n")


_default_tracer: Tracer | None = None
_tracer_lock = threading.Lock()


def get_tracer(sink: Path | None = None) -> Tracer:
    """Return a process-wide default tracer, creating it on first use."""
    global _default_tracer
    with _tracer_lock:
        if _default_tracer is None:
            _default_tracer = Tracer(sink=sink)
        return _default_tracer


__all__ = ["Span", "SpanStatus", "Tracer", "get_logger", "get_tracer"]
