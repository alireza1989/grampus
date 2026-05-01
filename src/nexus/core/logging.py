"""Structured logging setup using structlog with correlation ID support."""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import IO, Any

import structlog

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def get_logger(name: str) -> Any:
    """Return a structlog bound logger for the given name."""
    return structlog.get_logger(name)


def bind_correlation_id(correlation_id: str | None = None) -> Any:
    """Bind a correlation ID to the current context. Auto-generates one if not provided."""
    cid = correlation_id or str(uuid.uuid4())
    token = _correlation_id.set(cid)
    structlog.contextvars.bind_contextvars(correlation_id=cid)
    return token


def _add_correlation_id(
    logger: Any, method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    cid = _correlation_id.get("")
    if cid:
        event_dict.setdefault("correlation_id", cid)
    return event_dict


def configure_logging(
    *,
    dev: bool = False,
    level: str = "INFO",
    stream: IO[str] | None = None,
) -> None:
    """Configure structlog for the application.

    Args:
        dev: Use human-readable console output when True; JSON when False.
        level: Standard log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        stream: Optional output stream (defaults to sys.stdout).
    """
    output_stream = stream or sys.stdout

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _add_correlation_id,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", key="timestamp"),
        structlog.processors.StackInfoRenderer(),
    ]

    if dev:
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors
        + [
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=output_stream),  # type: ignore[arg-type]
        cache_logger_on_first_use=False,
    )
