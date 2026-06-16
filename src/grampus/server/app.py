"""FastAPI application factory for the Nexus REST API."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False

from grampus.core.errors import GrampusError
from grampus.core.types import AgentDefinition


def _ui_static_path() -> str:
    return str(Path(__file__).parent / "ui" / "static")


def _pkg_version() -> str:
    try:
        return importlib.metadata.version("grampus-ai")
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
    grampus_metrics: Any | None = None,
    alert_evaluator: Any | None = None,
    eval_run_store: Any | None = None,
    a2a_executor: Any | None = None,
    a2a_task_store: Any | None = None,
    a2a_api_key: str | None = None,
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
            "FastAPI is not installed. Install server deps: pip install grampus-ai[server]"
        )

    from grampus.orchestration.handoff import AgentRegistry as LegacyAgentRegistry
    from grampus.server.openai_compat import create_openai_router
    from grampus.server.routes import build_a2a_router, create_router
    from grampus.server.ui.router import router as ui_router
    from grampus.server.webhook import WebhookRegistry

    app = FastAPI(title="Grampus Agent API", version=_pkg_version())

    # CORS is intentionally open — this is a local/self-hosted server, not a public API.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from collections import deque

    app.state.runner = runner
    app.state.agent_def = agent_def
    app.state.memory_manager = memory_manager
    app.state.webhook_registry = (
        webhook_registry if webhook_registry is not None else WebhookRegistry()
    )
    app.state.schedule_store = schedule_store
    app.state.agent_registry = (
        agent_registry if agent_registry is not None else LegacyAgentRegistry()
    )
    app.state.grampus_metrics = grampus_metrics
    app.state.alert_evaluator = alert_evaluator
    app.state.alert_history = deque(maxlen=500)
    app.state.eval_run_store = eval_run_store
    app.state.a2a_executor = a2a_executor
    app.state.a2a_task_store = a2a_task_store
    app.state.a2a_api_key = a2a_api_key

    @app.exception_handler(GrampusError)
    async def _grampus_error(request: Request, exc: GrampusError) -> JSONResponse:
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

    app.mount("/ui/static", StaticFiles(directory=_ui_static_path()), name="ui-static")
    app.include_router(create_router(memory_manager is not None))
    app.include_router(create_openai_router(), prefix="/v1")
    app.include_router(ui_router, prefix="/ui")

    a2a_router = build_a2a_router(
        agent_def=agent_def,
        a2a_executor=a2a_executor,
        a2a_task_store=a2a_task_store,
        agent_registry=app.state.agent_registry,
        api_key=a2a_api_key,
    )
    app.include_router(a2a_router)

    return app
