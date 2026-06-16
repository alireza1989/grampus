"""FastAPI router for all /ui/* routes — dashboard, memory inspector, partials."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

from grampus.core.logging import get_logger

_log = get_logger(__name__)

_HERE = Path(__file__).parent
_templates = Jinja2Templates(directory=str(_HERE / "templates"))

router = APIRouter(include_in_schema=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metrics_context(request: Request) -> dict[str, Any]:
    metrics = getattr(request.app.state, "grampus_metrics", None)
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


# ---------------------------------------------------------------------------
# Evals dashboard
# ---------------------------------------------------------------------------


def _enrich_runs(runs: list[Any]) -> list[dict[str, Any]]:
    """Add regression flag to each run (newest-first input)."""
    # Build per-suite chronological history from the newest-first list
    by_suite: dict[str, list[Any]] = {}
    for r in reversed(runs):  # oldest first
        by_suite.setdefault(r.suite_name, []).append(r)

    prev_map: dict[str, float | None] = {}
    for suite_runs in by_suite.values():
        for i, r in enumerate(suite_runs):
            prev_map[r.run_id] = suite_runs[i - 1].pass_rate if i > 0 else None

    result: list[dict[str, Any]] = []
    for run in runs:
        prev_rate = prev_map.get(run.run_id)
        regression = prev_rate is not None and (prev_rate - run.pass_rate) > 0.05
        result.append({"run": run, "regression": regression})
    return result


@router.get("/evals/")
async def evals_page(request: Request) -> Response:
    """Evals dashboard full page."""
    store = getattr(request.app.state, "eval_run_store", None)
    suite_names: list[str] = store.list_suite_names() if store is not None else []
    return _templates.TemplateResponse(
        request,
        "evals.html",
        {"active_page": "evals", "suite_names": suite_names},
    )


@router.get("/evals/_runs")
async def evals_runs(
    request: Request,
    suite_name: str = "",
    limit: int = 50,
) -> Response:
    """HTMX partial: runs table rows."""
    store = getattr(request.app.state, "eval_run_store", None)
    if store is None:
        return _templates.TemplateResponse(request, "_evals_runs.html", {"runs": []})
    raw = store.list_runs(suite_name=suite_name or None, limit=limit)
    runs = _enrich_runs(raw)
    return _templates.TemplateResponse(request, "_evals_runs.html", {"runs": runs})


@router.get("/evals/_detail/{run_id}")
async def evals_detail(request: Request, run_id: str) -> Response:
    """HTMX partial: case-level breakdown for one run."""
    store = getattr(request.app.state, "eval_run_store", None)
    if store is None:
        raise HTTPException(status_code=404, detail="Run not found")
    record = store.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _templates.TemplateResponse(request, "_evals_detail.html", {"record": record})


@router.get("/evals/_trend")
async def evals_trend(
    request: Request,
    suite_name: str = "",
    limit: int = 20,
) -> Response:
    """HTMX partial: ASCII trend chart, oldest→newest."""
    store = getattr(request.app.state, "eval_run_store", None)
    trend_runs: list[Any] = []
    if store is not None:
        newest_first = store.list_runs(suite_name=suite_name or None, limit=limit)
        trend_runs = list(reversed(newest_first))  # oldest first for chart
    return _templates.TemplateResponse(
        request, "_evals_trend.html", {"trend_runs": trend_runs, "suite_name": suite_name}
    )


# ---------------------------------------------------------------------------
# Cost analytics
# ---------------------------------------------------------------------------


@router.get("/cost/")
async def cost_page(request: Request) -> Response:
    """Cost analytics full page."""
    return _templates.TemplateResponse(
        request,
        "cost.html",
        {"active_page": "cost"},
    )


@router.get("/cost/_summary")
async def cost_summary(request: Request) -> Response:
    """HTMX partial: cost stat cards + model/agent tables (auto-refreshes every 30s)."""
    metrics = getattr(request.app.state, "grampus_metrics", None)
    summary = metrics.get_cost_summary() if metrics is not None else None
    return _templates.TemplateResponse(
        request,
        "_cost_summary.html",
        {"summary": summary},
    )
