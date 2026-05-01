"""Core Pydantic v2 data models for Nexus."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Role(StrEnum):
    """Message role in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class AgentStatus(StrEnum):
    """Current execution status of an agent."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING_FOR_HUMAN = "waiting_for_human"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolCall(BaseModel):
    """A request from the model to invoke a tool."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The outcome of executing a tool call."""

    tool_call_id: str
    output: Any | None = None
    error: str | None = None
    duration_ms: int = 0


class Message(BaseModel):
    """A single message in an agent conversation."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolParameter(BaseModel):
    """Describes a single parameter accepted by a tool."""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any | None = None
    enum: list[Any] | None = None


class ToolDefinition(BaseModel):
    """Full specification of a tool, including JSON-schema generation."""

    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)
    version: str = "1.0.0"

    def to_function_schema(self) -> dict[str, Any]:
        """Return an OpenAI/Anthropic-compatible function JSON schema."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {"type": param.type, "description": param.description}
            if param.enum is not None:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class TokenUsage(BaseModel):
    """Token consumption and cost for a model call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    model: str


class AgentDefinition(BaseModel):
    """Blueprint for an agent — its model, tools, and behaviour config."""

    name: str
    model: str
    system_prompt: str | None = None
    tools: list[str] = Field(default_factory=list)
    max_iterations: int = 10
    temperature: float = 0.0
    memory_enabled: bool = True
    cost_budget_usd: float | None = None

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {v}")
        return v


class AgentState(BaseModel):
    """Mutable runtime state of an executing agent."""

    agent_id: str
    session_id: str
    messages: list[Message] = Field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    current_step: int = 0
    total_token_usage: TokenUsage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExecutionResult(BaseModel):
    """The final result of an agent execution."""

    output: str | None
    messages: list[Message]
    tool_calls_made: int
    token_usage: TokenUsage
    duration_seconds: float
    steps_taken: int
    status: AgentStatus
