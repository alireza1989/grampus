"""FastAPI application factory for the Nexus REST API."""

from __future__ import annotations

import importlib.metadata
from typing import Any

try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False

from nexus.core.errors import NexusError
from nexus.core.types import AgentDefinition


def _pkg_version() -> str:
    try:
        return importlib.metadata.version("nexus-ai")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def create_app(
    runner: Any,
    agent_def: AgentDefinition,
    *,
    memory_manager: Any | None = None,
    webhook_registry: Any | None = None,
    schedule_store: Any | None = None,
    agent_registry: Any | None = None,
) -> Any:
    """Create and configure the FastAPI application.

    Args:
        runner: An AgentRunner instance.
        agent_def: The AgentDefinition this server exposes.
        memory_manager: Optional MemoryManager for the /memory endpoints.
        webhook_registry: Optional WebhookRegistry; a new one is created if omitted.
        schedule_store: Optional ScheduleStore for the /job callback endpoints.

    Returns:
        Configured FastAPI app with all routes mounted.

    Raises:
        ImportError: When fastapi is not installed.
    """
    if not _HAS_FASTAPI:
        raise ImportError(  # pragma: no cover
            "FastAPI is not installed. Install server deps: pip install nexus-ai[server]"
        )

    from nexus.server.openai_compat import create_openai_router
    from nexus.server.routes import create_router
    from nexus.server.webhook import WebhookRegistry

    app = FastAPI(title="Nexus Agent API", version=_pkg_version())

    # CORS is intentionally open — this is a local/self-hosted server, not a public API.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from nexus.orchestration.handoff import AgentRegistry

    app.state.runner = runner
    app.state.agent_def = agent_def
    app.state.memory_manager = memory_manager
    app.state.webhook_registry = (
        webhook_registry if webhook_registry is not None else WebhookRegistry()
    )
    app.state.schedule_store = schedule_store
    app.state.agent_registry = agent_registry if agent_registry is not None else AgentRegistry()

    @app.exception_handler(NexusError)
    async def _nexus_error(request: Request, exc: NexusError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": str(exc), "code": exc.code, "hint": exc.hint},
        )

    @app.exception_handler(Exception)
    async def _generic_error(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "code": "INTERNAL_ERROR"},
        )

    app.include_router(create_router(memory_manager is not None))
    app.include_router(create_openai_router(), prefix="/v1")
    return app
