"""Pydantic request/response schemas for the Nexus REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from nexus.core.types import TokenUsage


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
