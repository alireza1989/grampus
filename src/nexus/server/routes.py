"""Route handlers for the Nexus REST API."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json as _json
import secrets
from collections.abc import AsyncGenerator
from typing import Any, cast
from uuid import uuid4

from pydantic import BaseModel

try:
    from fastapi import APIRouter, Request
    from fastapi.responses import HTMLResponse, StreamingResponse
except ImportError as _exc:  # pragma: no cover
    raise ImportError("Install server deps: pip install nexus-ai[server]") from _exc

from nexus.core.logging import get_logger
from nexus.core.types import AgentDefinition, Role, StreamEvent, StreamEventType
from nexus.dapr.schedule_store import ScheduleStore
from nexus.observability.events import AgentEvent, EventLog
from nexus.orchestration.handoff import AgentRegistry
from nexus.server.models import (
    AgentStateResponse,
    HealthResponse,
    MemoryRecallRequest,
    MemoryRecallResponse,
    PendingSession,
    PendingSessionsResponse,
    ResumeRequest,
    ResumeResponse,
    RunRequest,
    RunResponse,
    StreamChunkResponse,
    WebhookAcceptedResponse,
    WebhookListResponse,
    WebhookRegisterRequest,
    WebhookResponse,
    WebhookTriggerResponse,
)
from nexus.server.trace_ui import TRACE_HTML
from nexus.server.ui import UI_HTML
from nexus.server.webhook import WebhookConfig, WebhookRegistry, extract_input, verify_signature

_log = get_logger(__name__)


class _A2ATaskRequest(BaseModel):
    """A2A Protocol task submission body."""

    id: str | None = None
    message: dict[str, Any]


class _A2ATaskResponse(BaseModel):
    """A2A Protocol task submission response."""

    id: str
    status: str = "submitted"
    message: str = "Task received. Use /a2a/tasks/{id} to poll status."


def _mask_secret(data: dict[str, Any]) -> dict[str, Any]:
    """Replace secret with masked value for listing."""
    masked = dict(data)
    if masked.get("secret"):
        masked["secret"] = "***"
    return masked


async def _run_and_callback(
    runner: Any,
    agent_def: AgentDefinition,
    user_input: str,
    session_id: str,
    config: WebhookConfig,
) -> None:
    """Background task: run the agent and POST result to callback_url if set."""
    from nexus.core.logging import get_logger

    log = get_logger(__name__)
    try:
        result = await runner.run(agent_def, user_input, session_id=session_id)
        if config.callback_url:
            import httpx

            payload = {
                "session_id": session_id,
                "output": result.output,
                "status": str(result.status),
                "steps_taken": result.steps_taken,
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(config.callback_url, json=payload)
    except Exception as exc:
        log.error("webhook_async_run_failed", session_id=session_id, error=str(exc))
        if config.callback_url:
            import httpx

            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(
                        config.callback_url,
                        json={"session_id": session_id, "error": str(exc), "status": "failed"},
                    )
            except Exception:
                pass


async def _run_scheduled_job(
    runner: Any,
    agent_def: AgentDefinition,
    input_text: str,
    session_id: str,
    job_name: str,
    schedule_store: ScheduleStore | None,
) -> None:
    """Background task: run agent for a scheduled job and update trigger metadata."""
    from nexus.core.logging import get_logger as _get_logger

    log = _get_logger(__name__)
    try:
        result = await runner.run(agent_def, input_text, session_id=session_id)
        log.info(
            "scheduled_job_completed",
            job=job_name,
            session_id=session_id,
            status=str(result.status),
        )
        if schedule_store is not None:
            cfg = await schedule_store.get(job_name)
            if cfg is not None:
                from datetime import UTC, datetime

                updated = cfg.model_copy(
                    update={
                        "last_triggered_at": datetime.now(UTC),
                        "trigger_count": cfg.trigger_count + 1,
                    }
                )
                await schedule_store.save(updated)
    except Exception as exc:
        log.error(
            "scheduled_job_failed",
            job=job_name,
            session_id=session_id,
            error=str(exc),
        )


def _pkg_version() -> str:
    try:
        return importlib.metadata.version("nexus-ai")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _apply_overrides(agent_def: AgentDefinition, body: RunRequest) -> AgentDefinition:
    data = agent_def.model_dump()
    if body.agent_name is not None:
        data["name"] = body.agent_name
    if body.temperature is not None:
        data["temperature"] = body.temperature
    if body.max_iterations is not None:
        data["max_iterations"] = body.max_iterations
    return AgentDefinition(**data)


def _event_to_chunk(event: StreamEvent) -> StreamChunkResponse:
    etype = str(event.event_type)
    if event.event_type == StreamEventType.TOKEN:
        return StreamChunkResponse(
            event_type=etype,
            delta=event.chunk.delta if event.chunk else "",
        )
    if event.event_type == StreamEventType.TOOL_CALL_START:
        return StreamChunkResponse(
            event_type=etype,
            tool_name=event.tool_call.name if event.tool_call else None,
        )
    if event.event_type == StreamEventType.TOOL_CALL_END:
        return StreamChunkResponse(
            event_type=etype,
            tool_name=event.tool_call.name if event.tool_call else None,
            tool_result=str(event.tool_result.output) if event.tool_result else None,
        )
    if event.event_type == StreamEventType.AGENT_END:
        return StreamChunkResponse(
            event_type=etype,
            token_usage=event.chunk.token_usage if event.chunk else None,
        )
    return StreamChunkResponse(event_type=etype, message=event.message)


def create_router(has_memory: bool) -> APIRouter:
    """Build and return the API router, optionally including memory endpoints."""
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        agent_def = cast(AgentDefinition, request.app.state.agent_def)
        return HealthResponse(
            status="ok",
            version=_pkg_version(),
            agent_name=agent_def.name,
        )

    @router.post("/run", response_model=RunResponse)
    async def run_agent(body: RunRequest, request: Request) -> RunResponse:
        runner = request.app.state.runner
        agent_def = cast(AgentDefinition, request.app.state.agent_def)
        session_id = body.session_id or f"api-{uuid4().hex[:8]}"
        effective_def = _apply_overrides(agent_def, body)
        result = await runner.run(effective_def, body.input, session_id=session_id)
        return RunResponse(
            output=result.output,
            session_id=session_id,
            steps_taken=result.steps_taken,
            tool_calls_made=result.tool_calls_made,
            token_usage=result.token_usage,
            duration_seconds=result.duration_seconds,
            status=str(result.status),
        )

    @router.post("/stream")
    async def stream_agent(body: RunRequest, request: Request) -> StreamingResponse:
        runner = request.app.state.runner
        agent_def = cast(AgentDefinition, request.app.state.agent_def)
        session_id = body.session_id or f"api-{uuid4().hex[:8]}"
        effective_def = _apply_overrides(agent_def, body)

        async def _generate() -> AsyncGenerator[str, None]:
            try:
                async for event in runner.stream(effective_def, body.input, session_id=session_id):
                    chunk = _event_to_chunk(event)
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    await asyncio.sleep(0)
            except Exception as exc:
                err = StreamChunkResponse(event_type=str(StreamEventType.ERROR), message=str(exc))
                yield f"data: {err.model_dump_json()}\n\n"

        return StreamingResponse(_generate(), media_type="text/event-stream")

    @router.get("/agents/pending", response_model=PendingSessionsResponse)
    async def list_pending(request: Request) -> PendingSessionsResponse:
        runner = request.app.state.runner
        agent_def = cast(AgentDefinition, request.app.state.agent_def)
        session_ids = runner.list_pending_sessions(agent_def.name)
        sessions: list[PendingSession] = []
        for sid in session_ids:
            try:
                state = await runner.get_state(agent_def.name, sid)
                last = next((m for m in reversed(state.messages) if m.role != Role.SYSTEM), None)
                sessions.append(
                    PendingSession(
                        session_id=sid,
                        agent_id=agent_def.name,
                        last_message=(last.content[:200] if last and last.content else ""),
                        waiting_since=state.updated_at.isoformat(),
                    )
                )
            except Exception:
                pass
        return PendingSessionsResponse(sessions=sessions, count=len(sessions))

    @router.get("/agents/{session_id}/state", response_model=AgentStateResponse)
    async def get_agent_state(session_id: str, request: Request) -> AgentStateResponse:
        from nexus.core.errors import OrchestrationError

        runner = request.app.state.runner
        agent_def = cast(AgentDefinition, request.app.state.agent_def)
        try:
            state = await runner.get_state(agent_def.name, session_id)
        except OrchestrationError as exc:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail=str(exc)) from exc
        messages = [
            {
                "role": str(m.role),
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
            }
            for m in state.messages
            if m.role != Role.SYSTEM
        ]
        return AgentStateResponse(
            session_id=session_id,
            agent_id=state.agent_id,
            status=str(state.status),
            message_count=len(messages),
            messages=messages,
        )

    @router.post("/agents/{session_id}/resume", response_model=ResumeResponse)
    async def resume_agent(
        session_id: str, body: ResumeRequest, request: Request
    ) -> ResumeResponse:
        runner = request.app.state.runner
        agent_def = cast(AgentDefinition, request.app.state.agent_def)
        result = await runner.resume(agent_def.name, session_id, body.input)
        return ResumeResponse(
            session_id=session_id,
            output=result.output,
            status=str(result.status),
            steps_taken=result.steps_taken,
            token_usage=result.token_usage,
            still_waiting=(str(result.status) == "waiting_for_human"),
        )

    @router.get("/ui", include_in_schema=False)
    async def ui() -> HTMLResponse:
        return HTMLResponse(UI_HTML)

    @router.get("/ui/events", include_in_schema=False)
    async def ui_events(request: Request) -> StreamingResponse:
        runner = request.app.state.runner
        agent_def = cast(AgentDefinition, request.app.state.agent_def)

        async def _generate() -> AsyncGenerator[str, None]:
            import json as _json

            while True:
                if await request.is_disconnected():
                    break
                session_ids = runner.list_pending_sessions(agent_def.name)
                sessions = [{"session_id": sid, "agent_id": agent_def.name} for sid in session_ids]
                payload = _json.dumps({"sessions": sessions})
                yield f"data: {payload}\n\n"
                await asyncio.sleep(2)

        return StreamingResponse(_generate(), media_type="text/event-stream")

    @router.get("/trace", include_in_schema=False)
    async def trace_ui() -> HTMLResponse:
        return HTMLResponse(TRACE_HTML)

    @router.get("/trace/{session_id}/history")
    async def trace_history(session_id: str, request: Request) -> dict[str, Any]:
        runner = request.app.state.runner
        agent_def = cast(AgentDefinition, request.app.state.agent_def)
        state_store = getattr(runner, "_state_store", None)
        event_log = await EventLog.open(
            agent_id=agent_def.name,
            session_id=session_id,
            state_store=state_store,
        )
        events = await event_log.replay()
        return {
            "session_id": session_id,
            "agent_id": agent_def.name,
            "events": [e.model_dump(mode="json") for e in events],
            "count": len(events),
        }

    @router.get("/trace/{session_id}/stream", include_in_schema=False)
    async def trace_stream(session_id: str, request: Request) -> StreamingResponse:
        runner = request.app.state.runner
        queue = runner.subscribe_trace(session_id)

        async def _generate() -> AsyncGenerator[str, None]:
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event: AgentEvent | None = await asyncio.wait_for(queue.get(), timeout=25.0)
                    except TimeoutError:
                        yield 'data: {"heartbeat": true}\n\n'
                        continue
                    if event is None:
                        yield 'data: {"done": true}\n\n'
                        break
                    yield f"data: {event.model_dump_json()}\n\n"
            finally:
                runner.unsubscribe_trace(session_id, queue)

        return StreamingResponse(_generate(), media_type="text/event-stream")

    @router.post("/webhooks", response_model=WebhookResponse, status_code=201)
    async def register_webhook(body: WebhookRegisterRequest, request: Request) -> WebhookResponse:
        registry: WebhookRegistry = request.app.state.webhook_registry
        config = WebhookConfig(
            name=body.name,
            secret=body.secret if body.secret is not None else secrets.token_hex(32),
            input_template=body.input_template,
            input_field=body.input_field,
            async_mode=body.async_mode,
            callback_url=body.callback_url,
        )
        registry.register(config)
        return WebhookResponse(**config.model_dump())

    @router.get("/webhooks", response_model=WebhookListResponse)
    async def list_webhooks(request: Request) -> WebhookListResponse:
        registry: WebhookRegistry = request.app.state.webhook_registry
        webhooks = [WebhookResponse(**_mask_secret(c.model_dump())) for c in registry.list_all()]
        return WebhookListResponse(webhooks=webhooks, count=len(webhooks))

    @router.delete("/webhooks/{webhook_id}", status_code=204)
    async def delete_webhook(webhook_id: str, request: Request) -> None:
        registry: WebhookRegistry = request.app.state.webhook_registry
        found = registry.delete(webhook_id)
        if not found:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail=f"Webhook '{webhook_id}' not found")

    @router.post("/webhooks/{webhook_id}/trigger")
    async def trigger_webhook(
        webhook_id: str,
        request: Request,
    ) -> Any:
        registry: WebhookRegistry = request.app.state.webhook_registry
        config = registry.get(webhook_id)
        if config is None:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail=f"Webhook '{webhook_id}' not found")

        raw_body = await request.body()
        sig_header = request.headers.get("X-Nexus-Signature")

        if not verify_signature(raw_body, config.secret, sig_header):
            from fastapi import HTTPException

            raise HTTPException(status_code=401, detail="Invalid webhook signature")

        try:
            payload: dict[str, Any] = _json.loads(raw_body) if raw_body else {}
        except _json.JSONDecodeError:
            payload = {"raw": raw_body.decode(errors="replace")}

        agent_input = extract_input(payload, config)
        session_id = f"{config.session_prefix}-{secrets.token_hex(6)}"

        runner = request.app.state.runner
        agent_def = cast(AgentDefinition, request.app.state.agent_def)

        if config.async_mode:
            asyncio.create_task(
                _run_and_callback(runner, agent_def, agent_input, session_id, config)
            )
            return WebhookAcceptedResponse(session_id=session_id, webhook_id=webhook_id)

        result = await runner.run(agent_def, agent_input, session_id=session_id)
        return WebhookTriggerResponse(
            session_id=session_id,
            output=result.output,
            status=str(result.status),
            steps_taken=result.steps_taken,
            token_usage=result.token_usage,
            duration_seconds=result.duration_seconds,
        )

    @router.post("/job/{job_name}", include_in_schema=False)
    async def job_callback(job_name: str, request: Request) -> dict[str, Any]:
        """Dapr Jobs callback — fired when a scheduled job triggers."""
        from uuid import uuid4

        runner = request.app.state.runner
        agent_def = cast(AgentDefinition, request.app.state.agent_def)
        schedule_store = cast(
            ScheduleStore | None,
            getattr(request.app.state, "schedule_store", None),
        )

        raw = await request.body()
        try:
            outer: dict[str, Any] = _json.loads(raw) if raw else {}
            inner_str: str = outer.get("value", "{}")
            payload: dict[str, Any] = _json.loads(inner_str) if inner_str else {}
        except (ValueError, KeyError):
            payload = {}

        input_text: str = payload.get("input", f"Scheduled run: {job_name}")
        session_prefix: str = payload.get("session_prefix", "sched")
        session_id = f"{session_prefix}-{uuid4().hex[:8]}"

        _log.info("job_callback_fired", job=job_name, session_id=session_id)

        asyncio.create_task(
            _run_scheduled_job(runner, agent_def, input_text, session_id, job_name, schedule_store)
        )

        return {"accepted": True, "session_id": session_id, "job": job_name}

    # ------------------------------------------------------------------
    # A2A Protocol v1.2 endpoints
    # ------------------------------------------------------------------

    @router.get("/.well-known/agent.json")
    async def agent_card(request: Request) -> Any:
        agent_def = cast(AgentDefinition, request.app.state.agent_def)
        registry: AgentRegistry = getattr(request.app.state, "agent_registry", None) or AgentRegistry()
        base_url = str(request.base_url).rstrip("/")
        card = registry.generate_agent_card(agent_def, base_url)
        return card.model_dump()

    @router.get("/a2a/agents")
    async def list_a2a_agents(request: Request) -> dict[str, Any]:
        registry: AgentRegistry = getattr(request.app.state, "agent_registry", None) or AgentRegistry()
        return {"agents": registry.list_agents()}

    @router.post("/a2a/tasks", response_model=_A2ATaskResponse)
    async def submit_a2a_task(body: _A2ATaskRequest, request: Request) -> _A2ATaskResponse:
        runner = request.app.state.runner
        agent_def = cast(AgentDefinition, request.app.state.agent_def)
        task_id = body.id or uuid4().hex[:8]
        session_id = f"a2a-{task_id}"

        parts: list[dict[str, Any]] = body.message.get("parts", [])
        input_text = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")
        if not input_text:
            input_text = str(body.message)

        await runner.run(agent_def, input_text, session_id=session_id)
        return _A2ATaskResponse(
            id=task_id,
            status="completed",
            message=f"Task {task_id} completed.",
        )

    if has_memory:

        @router.post("/memory/recall", response_model=MemoryRecallResponse)
        async def memory_recall(
            body: MemoryRecallRequest, request: Request
        ) -> MemoryRecallResponse:
            mm = request.app.state.memory_manager
            result = await mm.recall(body.query, memory_types=body.memory_types, top_k=body.top_k)
            episodic: list[dict[str, Any]] = [
                r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in result.episodic
            ]
            semantic: list[dict[str, Any]] = [
                f.model_dump() if hasattr(f, "model_dump") else dict(f) for f in result.semantic
            ]
            return MemoryRecallResponse(query=body.query, episodic=episodic, semantic=semantic)

        @router.delete("/memory/{record_id}")
        async def memory_delete(
            record_id: str, memory_type: str, request: Request
        ) -> dict[str, Any]:
            mm = request.app.state.memory_manager
            await mm.forget(record_id, memory_type=memory_type)
            return {"deleted": True, "record_id": record_id}

    return router
