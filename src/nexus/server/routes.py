"""Route handlers for the Nexus REST API."""

from __future__ import annotations

import asyncio
import importlib.metadata
from collections.abc import AsyncGenerator
from typing import Any, cast
from uuid import uuid4

try:
    from fastapi import APIRouter, Request
    from fastapi.responses import StreamingResponse
except ImportError as _exc:  # pragma: no cover
    raise ImportError("Install server deps: pip install nexus-ai[server]") from _exc

from nexus.core.types import AgentDefinition, StreamEvent, StreamEventType
from nexus.server.models import (
    HealthResponse,
    MemoryRecallRequest,
    MemoryRecallResponse,
    RunRequest,
    RunResponse,
    StreamChunkResponse,
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
