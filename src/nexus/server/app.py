"""FastAPI application factory for the Nexus REST API."""

from __future__ import annotations

import importlib.metadata
from typing import Any

try:
    from fastapi import FastAPI, Request
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
) -> Any:
    """Create and configure the FastAPI application.

    Args:
        runner: An AgentRunner instance.
        agent_def: The AgentDefinition this server exposes.
        memory_manager: Optional MemoryManager for the /memory endpoints.

    Returns:
        Configured FastAPI app with all routes mounted.

    Raises:
        ImportError: When fastapi is not installed.
    """
    if not _HAS_FASTAPI:
        raise ImportError(  # pragma: no cover
            "FastAPI is not installed. Install server deps: pip install nexus-ai[server]"
        )

    from nexus.server.routes import create_router

    app = FastAPI(title="Nexus Agent API", version=_pkg_version())

    app.state.runner = runner
    app.state.agent_def = agent_def
    app.state.memory_manager = memory_manager

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
    return app
