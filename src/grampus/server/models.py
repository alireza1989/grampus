"""Pydantic request/response schemas for the Nexus REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from grampus.core.types import TokenUsage


class RunRequest(BaseModel):
    """Body for POST /run and POST /stream."""

    input: str
    session_id: str | None = None
    agent_name: str | None = None
    temperature: float | None = None
    max_iterations: int | None = None


class RunResponse(BaseModel):
    """Response from POST /run."""

    output: str | None
    session_id: str
    steps_taken: int
    tool_calls_made: int
    token_usage: TokenUsage
    duration_seconds: float
    status: str


class StreamChunkResponse(BaseModel):
    """Single SSE payload — one per stream event."""

    event_type: str
    delta: str = ""
    tool_name: str | None = None
    tool_result: str | None = None
    token_usage: TokenUsage | None = None
    message: str = ""


class MemoryRecallRequest(BaseModel):
    """Body for POST /memory/recall."""

    query: str
    top_k: int = 5
    memory_types: list[str] = Field(default_factory=lambda: ["episodic", "semantic"])


class MemoryRecallResponse(BaseModel):
    """Response from POST /memory/recall."""

    query: str
    episodic: list[dict[str, Any]] = Field(default_factory=list)
    semantic: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str
    version: str
    agent_name: str


class PendingSession(BaseModel):
    """Summary of one WAITING_FOR_HUMAN session."""

    session_id: str
    agent_id: str
    last_message: str
    waiting_since: str


class PendingSessionsResponse(BaseModel):
    """Response from GET /agents/pending."""

    sessions: list[PendingSession]
    count: int


class AgentStateResponse(BaseModel):
    """Response from GET /agents/{session_id}/state."""

    session_id: str
    agent_id: str
    status: str
    message_count: int
    messages: list[dict[str, Any]]


class ResumeRequest(BaseModel):
    """Body for POST /agents/{session_id}/resume."""

    input: str


class ResumeResponse(BaseModel):
    """Response from POST /agents/{session_id}/resume."""

    session_id: str
    output: str | None
    status: str
    steps_taken: int
    token_usage: TokenUsage | None = None
    still_waiting: bool


class WebhookRegisterRequest(BaseModel):
    """Body for POST /webhooks — register a new webhook."""

    name: str = ""
    secret: str | None = None
    input_template: str = ""
    input_field: str = ""
    async_mode: bool = False
    callback_url: str = ""


class WebhookResponse(BaseModel):
    """Webhook config as returned by the API (secret included on creation only)."""

    id: str
    name: str
    secret: str
    input_template: str
    input_field: str
    async_mode: bool
    callback_url: str


class WebhookListResponse(BaseModel):
    """Response from GET /webhooks."""

    webhooks: list[WebhookResponse]
    count: int


class WebhookTriggerResponse(BaseModel):
    """Response from POST /webhooks/{id}/trigger (sync mode)."""

    session_id: str
    output: str | None
    status: str
    steps_taken: int
    token_usage: TokenUsage | None = None
    duration_seconds: float


class WebhookAcceptedResponse(BaseModel):
    """Response from POST /webhooks/{id}/trigger (async_mode=True)."""

    accepted: bool = True
    session_id: str
    webhook_id: str
