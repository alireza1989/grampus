"""OpenAI-compatible /v1 router for the Nexus REST API."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any, cast
from uuid import uuid4

try:
    from fastapi import APIRouter, Depends, Header, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as _exc:  # pragma: no cover
    raise ImportError("Install server deps: pip install nexus-ai[server]") from _exc

from pydantic import BaseModel, ConfigDict

from nexus.core.types import AgentDefinition, StreamEventType, TokenUsage

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class OAIStreamOptions(BaseModel):
    include_usage: bool = False


class OAIMessage(BaseModel):
    role: str
    content: str


class OAIChatRequest(BaseModel):
    model: str
    messages: list[OAIMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream_options: OAIStreamOptions | None = None
    # Silently accepted OpenAI fields — the real SDK sends these
    top_p: float | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    seed: int | None = None
    user: str | None = None
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class OAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OAIMessageResponse(BaseModel):
    role: str = "assistant"
    content: str


class OAIChoice(BaseModel):
    index: int = 0
    message: OAIMessageResponse
    finish_reason: str = "stop"


class OAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OAIChoice]
    usage: OAIUsage
    system_fingerprint: str = "nexus-v0.1"


class OAIDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class OAIStreamChoice(BaseModel):
    index: int = 0
    delta: OAIDelta
    finish_reason: str | None = None


class OAIChatChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[OAIStreamChoice]
    usage: OAIUsage | None = None
    system_fingerprint: str = "nexus-v0.1"


class OAIModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "nexus"


class OAIModelList(BaseModel):
    object: str = "list"
    data: list[OAIModelObject]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _extract_user_input(messages: list[OAIMessage]) -> str:
    """Return the content of the last user message. Raises ValueError if none found."""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    raise ValueError("No user message in messages list")


def _extract_system_prompt(messages: list[OAIMessage]) -> str | None:
    """Return content of the first system message, or None."""
    for msg in messages:
        if msg.role == "system":
            return msg.content
    return None


def _build_completion_id() -> str:
    return f"chatcmpl-{uuid4().hex[:8]}"


def _nexus_usage_to_oai(token_usage: TokenUsage) -> OAIUsage:
    return OAIUsage(
        prompt_tokens=token_usage.input_tokens,
        completion_tokens=token_usage.output_tokens,
        total_tokens=token_usage.total_tokens,
    )


def _finish_reason(tool_calls_made: int) -> str:
    return "tool_calls" if tool_calls_made > 0 else "stop"


def _build_effective_def(
    agent_def: AgentDefinition,
    body: OAIChatRequest,
    system_prompt: str | None,
) -> AgentDefinition:
    data = agent_def.model_dump()
    if body.temperature is not None:
        data["temperature"] = body.temperature
    if system_prompt is not None:
        data["system_prompt"] = system_prompt
    # body.model is echoed back in the response only; the actual LLM used is always
    # the one configured at server startup via AgentDefinition.model.
    return AgentDefinition(**data)


# ---------------------------------------------------------------------------
# Auth passthrough (no-op)
# ---------------------------------------------------------------------------


def _accept_bearer(authorization: str | None = Header(default=None)) -> None:
    """Accept (but do not validate) the Authorization header.

    The OpenAI SDK always sends Authorization: Bearer <key>.
    We intentionally do not validate it — any key works.
    """
    return None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def create_openai_router() -> APIRouter:
    """Build and return the /v1 OpenAI-compatible router."""
    router = APIRouter()

    @router.get("/models", response_model=OAIModelList)
    async def list_models(request: Request) -> OAIModelList:
        agent_def = cast(AgentDefinition, request.app.state.agent_def)
        return OAIModelList(data=[OAIModelObject(id=agent_def.name, created=int(time.time()))])

    @router.post("/chat/completions")
    async def chat_completions(
        body: OAIChatRequest,
        request: Request,
        _: None = Depends(_accept_bearer),
    ) -> Any:
        try:
            user_input = _extract_user_input(body.messages)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if body.stream:
            return _streaming_response(body, user_input, request)
        return await _blocking_response(body, user_input, request)

    return router


async def _blocking_response(
    body: OAIChatRequest, user_input: str, request: Request
) -> JSONResponse:
    runner = request.app.state.runner
    agent_def = cast(AgentDefinition, request.app.state.agent_def)
    system_prompt = _extract_system_prompt(body.messages)
    effective_def = _build_effective_def(agent_def, body, system_prompt)
    session_id = f"oai-{uuid4().hex[:8]}"
    result = await runner.run(effective_def, user_input, session_id=session_id)
    response = OAIChatResponse(
        id=_build_completion_id(),
        created=int(time.time()),
        model=body.model,
        choices=[
            OAIChoice(
                message=OAIMessageResponse(content=result.output or ""),
                finish_reason=_finish_reason(result.tool_calls_made),
            )
        ],
        usage=_nexus_usage_to_oai(result.token_usage),
    )
    return JSONResponse(content=response.model_dump())


def _streaming_response(
    body: OAIChatRequest, user_input: str, request: Request
) -> StreamingResponse:
    runner = request.app.state.runner
    agent_def = cast(AgentDefinition, request.app.state.agent_def)
    system_prompt = _extract_system_prompt(body.messages)
    effective_def = _build_effective_def(agent_def, body, system_prompt)
    session_id = f"oai-{uuid4().hex[:8]}"
    include_usage = body.stream_options is not None and body.stream_options.include_usage

    async def _generate() -> AsyncGenerator[str, None]:
        completion_id = _build_completion_id()
        created = int(time.time())
        accumulated_usage: OAIUsage | None = None

        try:
            first = OAIChatChunk(
                id=completion_id,
                created=created,
                model=body.model,
                choices=[OAIStreamChoice(delta=OAIDelta(role="assistant"), finish_reason=None)],
            )
            yield f"data: {first.model_dump_json()}\n\n"

            async for event in runner.stream(effective_def, user_input, session_id=session_id):
                if event.event_type == StreamEventType.TOKEN and event.chunk and event.chunk.delta:
                    chunk = OAIChatChunk(
                        id=completion_id,
                        created=created,
                        model=body.model,
                        choices=[
                            OAIStreamChoice(
                                delta=OAIDelta(content=event.chunk.delta), finish_reason=None
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    await asyncio.sleep(0)

                elif event.event_type == StreamEventType.AGENT_END:
                    if event.chunk and event.chunk.token_usage:
                        accumulated_usage = _nexus_usage_to_oai(event.chunk.token_usage)

                    final = OAIChatChunk(
                        id=completion_id,
                        created=created,
                        model=body.model,
                        choices=[OAIStreamChoice(delta=OAIDelta(), finish_reason="stop")],
                    )
                    yield f"data: {final.model_dump_json()}\n\n"

                # TOOL_CALL_START, TOOL_CALL_END, ITERATION_START, AGENT_START are silently skipped.

            if include_usage and accumulated_usage is not None:
                usage_chunk = OAIChatChunk(
                    id=completion_id,
                    created=created,
                    model=body.model,
                    choices=[],
                    usage=accumulated_usage,
                )
                yield f"data: {usage_chunk.model_dump_json()}\n\n"

        except Exception as exc:
            err = OAIChatChunk(
                id=completion_id,
                created=created,
                model=body.model,
                choices=[
                    OAIStreamChoice(
                        delta=OAIDelta(content=f"\n[Error: {exc}]"), finish_reason="stop"
                    )
                ],
            )
            yield f"data: {err.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")
