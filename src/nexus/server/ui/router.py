"""FastAPI router for all /ui/* routes — dashboard, memory inspector, partials."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

from nexus.core.logging import get_logger

_log = get_logger(__name__)

_HERE = Path(__file__).parent
_templates = Jinja2Templates(directory=str(_HERE / "templates"))

router = APIRouter(include_in_schema=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metrics_context(request: Request) -> dict[str, Any]:
    metrics = getattr(request.app.state, "nexus_metrics", None)
    if metrics is None:
        return {
            "active_agents": None,
            "total_llm_calls": None,
            "total_cost_usd": None,
            "error_count": None,
        }
    snap = metrics.snapshot()
    return {
        "active_agents": snap.active_agents,
        "total_llm_calls": snap.llm_call_count,
        "total_cost_usd": snap.total_cost_usd,
        "error_count": snap.total_errors,
    }


async def _list_memory(
    request: Request,
    *,
    agent_id: str,
    memory_type: str,
    q: str,
    min_trust: float,
    page: int,
    limit: int,
) -> list[dict[str, Any]]:
    manager = getattr(request.app.state, "memory_manager", None)
    if manager is None:
        return []
    try:
        return await manager.list_records(  # type: ignore[no-any-return]
            agent_id=agent_id or None,
            memory_type=memory_type or None,
            query=q or None,
            min_trust=min_trust,
            limit=limit,
            offset=page * limit,
        )
    except Exception:  # noqa: BLE001
        _log.warning("ui_list_records_failed")
        return []


async def _find_entry(request: Request, entry_id: str) -> dict[str, Any] | None:
    manager = getattr(request.app.state, "memory_manager", None)
    if manager is None:
        return None
    try:
        records: list[dict[str, Any]] = await manager.list_records(limit=10_000)
        for rec in records:
            if rec.get("id") == entry_id:
                return rec
    except Exception:  # noqa: BLE001
        _log.warning("ui_find_entry_failed", entry_id=entry_id)
    return None


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@router.get("/")
async def dashboard(request: Request) -> Response:
    """Main dashboard page."""
    alerts = list(getattr(request.app.state, "alert_history", []))[-20:]
    return _templates.TemplateResponse(
        request,
        "dashboard.html",
        {"active_page": "dashboard", "recent_alerts": alerts},
    )


@router.get("/_stats")
async def dashboard_stats(request: Request) -> Response:
    """HTMX partial: stat cards that auto-refresh."""
    ctx = _metrics_context(request)
    return _templates.TemplateResponse(
        request,
        "_dashboard_stats.html",
        ctx,
    )


# ---------------------------------------------------------------------------
# Memory inspector
# ---------------------------------------------------------------------------


@router.get("/memory/")
async def memory_page(request: Request) -> Response:
    """Memory inspector full page."""
    return _templates.TemplateResponse(
        request,
        "memory.html",
        {"active_page": "memory"},
    )


@router.get("/memory/_rows")
async def memory_rows(
    request: Request,
    agent_id: str = "",
    memory_type: str = "",
    q: str = "",
    min_trust: float = 0.0,
    page: int = 0,
    limit: int = 50,
) -> Response:
    """HTMX partial: <tr> rows for the memory table."""
    entries = await _list_memory(
        request,
        agent_id=agent_id,
        memory_type=memory_type,
        q=q,
        min_trust=min_trust,
        page=page,
        limit=limit,
    )
    return _templates.TemplateResponse(
        request,
        "_memory_rows.html",
        {"entries": entries},
    )


@router.get("/memory/_detail/{entry_id}")
async def memory_detail(request: Request, entry_id: str) -> Response:
    """HTMX partial: full detail side-panel for one entry."""
    entry = await _find_entry(request, entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    return _templates.TemplateResponse(
        request,
        "_memory_detail.html",
        {"entry": entry},
    )


@router.delete("/memory/{entry_id}")
async def memory_delete(request: Request, entry_id: str, memory_type: str = "") -> Response:
    """Delete a memory entry; HTMX swaps out the row."""
    manager = getattr(request.app.state, "memory_manager", None)
    if manager is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    mtype = memory_type or "episodic"
    try:
        await manager.forget(entry_id, memory_type=mtype)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Entry not found") from exc
    return Response(status_code=200, content="")
