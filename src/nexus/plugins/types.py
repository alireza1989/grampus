"""Context dataclasses and type aliases for the Nexus plugin hook system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nexus.core.errors import NexusError

# ---------------------------------------------------------------------------
# Hook context objects — frozen dataclasses (plugins receive them read-only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentStartContext:
    """Context passed to on_agent_start hooks."""

    agent_id: str
    session_id: str
    user_input: str
    model: str


@dataclass(frozen=True)
class AgentEndContext:
    """Context passed to on_agent_end hooks."""

    agent_id: str
    session_id: str
    output: str
    steps_taken: int
    total_cost_usd: float
    duration_seconds: float


@dataclass(frozen=True)
class LLMCallContext:
    """Context passed to pre_llm_call / post_llm_call hooks."""

    agent_id: str
    session_id: str
    model: str
    step: int


@dataclass(frozen=True)
class ToolCallContext:
    """Context passed to pre_tool_call hooks."""

    agent_id: str
    session_id: str
    tool_name: str
    step: int


@dataclass(frozen=True)
class ToolResultContext:
    """Context passed to post_tool_call hooks."""

    agent_id: str
    session_id: str
    tool_name: str
    duration_ms: float
    ok: bool


@dataclass(frozen=True)
class MemoryWriteContext:
    """Context passed to pre_memory_write / post_memory_write hooks."""

    agent_id: str
    session_id: str
    memory_type: str  # "episodic" | "semantic" | "procedural" | "working"
    source_id: str


@dataclass(frozen=True)
class ErrorContext:
    """Context passed to on_error hooks."""

    agent_id: str
    session_id: str
    error: Exception
    step: int


# ---------------------------------------------------------------------------
# HookBlockedError — raised by pre-hooks to cancel an operation
# ---------------------------------------------------------------------------


class HookBlockedError(NexusError):
    """Raised by a pre-hook plugin to block the operation.

    Raise this inside pre_llm_call, pre_tool_call, or pre_memory_write to
    cancel the operation. AgentRunner treats this as a SafetyError with
    code="PLUGIN_BLOCKED".

    Args:
        message: Human-readable reason for blocking.
        code: Machine-readable error code (defaults to "PLUGIN_BLOCKED").
    """

    def __init__(
        self,
        message: str = "Operation blocked by plugin hook",
        *,
        code: str = "PLUGIN_BLOCKED",
        details: dict[str, object] | None = None,
        hint: str = "",
    ) -> None:
        super().__init__(message, code=code, details=details, hint=hint)


# ---------------------------------------------------------------------------
# Pre-hook return type aliases
# ---------------------------------------------------------------------------

# pre_llm_call returns None (pass-through) or a modified messages list
LLMCallModification = list[Any] | None

# pre_tool_call returns None (pass-through) or modified arguments dict
ToolCallModification = dict[str, Any] | None

# pre_memory_write returns None (pass-through) or modified content string
MemoryWriteModification = str | None
