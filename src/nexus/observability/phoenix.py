"""Arize Phoenix integration helpers for Nexus observability."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator
from typing import Any

from pydantic import BaseModel

from nexus.observability.tracer import NexusTracer


class PhoenixConfig(BaseModel):
    """Configuration for Arize Phoenix integration."""

    endpoint: str = "http://localhost:6006"
    otlp_grpc_endpoint: str = "http://localhost:4317"
    project_name: str = "nexus"
    enabled: bool = True


def configure_phoenix_tracer(
    config: PhoenixConfig,
    *,
    service_name: str = "nexus-agent",
    agent_id: str = "unknown",
    session_id: str | None = None,
) -> NexusTracer:
    """Return a NexusTracer pre-configured to export to Phoenix via OTLP.

    When config.enabled is False, returns a NoOp NexusTracer (no OTLP export).

    Args:
        config: Phoenix connection settings.
        service_name: OTEL service.name resource attribute.
        agent_id: Default agent_id attached to every span.
        session_id: Optional session identifier attached to every span.

    Returns:
        A configured NexusTracer instance.
    """
    otlp_endpoint = config.otlp_grpc_endpoint if config.enabled else None
    extra: dict[str, str] = {"phoenix.project.name": config.project_name}
    return NexusTracer(
        service_name=service_name,
        otlp_endpoint=otlp_endpoint,
        agent_id=agent_id,
        session_id=session_id,
        extra_resource_attrs=extra,
    )


def _config_from_env() -> PhoenixConfig:
    """Build a PhoenixConfig by reading environment variables."""
    kwargs: dict[str, Any] = {}
    if endpoint := os.environ.get("PHOENIX_ENDPOINT"):
        kwargs["endpoint"] = endpoint
    if otlp := os.environ.get("PHOENIX_OTLP_ENDPOINT"):
        kwargs["otlp_grpc_endpoint"] = otlp
    if project := os.environ.get("PHOENIX_PROJECT_NAME"):
        kwargs["project_name"] = project
    return PhoenixConfig(**kwargs)


@contextlib.contextmanager
def phoenix_tracer(
    config: PhoenixConfig | None = None,
    **kwargs: Any,
) -> Generator[NexusTracer, None, None]:
    """Context manager that yields a Phoenix-connected NexusTracer.

    When config is None, reads PHOENIX_ENDPOINT, PHOENIX_OTLP_ENDPOINT, and
    PHOENIX_PROJECT_NAME from the environment.

    Args:
        config: Optional explicit Phoenix configuration. Reads env vars when None.
        **kwargs: Forwarded to configure_phoenix_tracer (service_name, agent_id, session_id).

    Yields:
        A NexusTracer configured to export to Phoenix.
    """
    resolved = config if config is not None else _config_from_env()
    tracer = configure_phoenix_tracer(resolved, **kwargs)
    yield tracer
