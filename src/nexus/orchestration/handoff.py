"""Agent handoff primitives — OpenAI-style transfer tools, A2A Protocol v1.2 cards."""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nexus.core.errors import HandoffError
from nexus.core.types import (
    AgentDefinition,
    Message,
    Role,
    TokenUsage,
    ToolDefinition,
    ToolParameter,
)
from nexus.observability.events import EventLog, EventType

# ---------------------------------------------------------------------------
# Module-level callback registry keyed by tool name (transfer_to_<name>)
# ---------------------------------------------------------------------------

_HANDOFF_CALLBACKS: dict[str, Callable[..., Awaitable[None]]] = {}

# ---------------------------------------------------------------------------
# Injection-pattern sanitizer
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(previous|above|all)\s+instructions?|"
    r"system\s*:\s*you\s+are\s+now|"
    r"forget\s+(everything|all)\s+(above|previous)|"
    r"new\s+instructions?\s*:)",
    re.IGNORECASE,
)


def _sanitize_context(context: HandoffContext) -> HandoffContext:
    """Return a new HandoffContext with injection patterns removed from text fields."""

    def clean(text: str | None) -> str | None:
        if text is None:
            return None
        return _INJECTION_PATTERNS.sub("[REDACTED]", text)

    return context.model_copy(
        update={
            "task": clean(context.task) or context.task,
            "context_summary": clean(context.context_summary),
            "constraints": [clean(c) or c for c in context.constraints],
            "relevant_messages": [
                msg.model_copy(update={"content": clean(msg.content)}) if msg.content else msg
                for msg in context.relevant_messages
            ],
        }
    )


# ---------------------------------------------------------------------------
# Handoff data models
# ---------------------------------------------------------------------------


class HandoffContext(BaseModel):
    """Context passed from source agent to target agent."""

    task: str
    context_summary: str | None = None
    relevant_messages: list[Message] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)


class HandoffPolicy(BaseModel):
    """Per-agent rules governing which handoffs are permitted."""

    allowed_targets: list[str] | None = None
    max_depth: int = 5
    timeout_seconds: float = 60.0
    require_confirmation: bool = False


class HandoffRequest(BaseModel):
    """Immutable record of a handoff invocation."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_agent_id: str
    source_session_id: str
    target_agent_name: str
    context: HandoffContext
    handoff_depth: int = 0
    trace_context: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HandoffResult(BaseModel):
    """Outcome of a completed handoff."""

    request_id: str
    output: str | None
    messages: list[Message] = Field(default_factory=list)
    token_usage: TokenUsage | None = None
    status: Literal["completed", "failed", "paused"]
    error: str | None = None
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# A2A Protocol v1.2 Agent Card
# ---------------------------------------------------------------------------


class AgentSkill(BaseModel):
    """A single capability advertised in an AgentCard."""

    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    input_modes: list[str] = Field(default_factory=lambda: ["text"])
    output_modes: list[str] = Field(default_factory=lambda: ["text"])


class AgentCapabilities(BaseModel):
    """Feature flags advertised in an AgentCard."""

    streaming: bool = False
    push_notifications: bool = False
    state_transition_history: bool = True


class AgentCard(BaseModel):
    """A2A Protocol v1.2 Agent Card — exposed at /.well-known/agent.json."""

    name: str
    description: str
    url: str
    version: str = "1.0.0"
    protocol_version: str = "1.2"
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: list[AgentSkill] = Field(default_factory=list)
    default_input_modes: list[str] = Field(default_factory=lambda: ["text"])
    default_output_modes: list[str] = Field(default_factory=lambda: ["text"])
    authentication: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


def create_handoff_tool(
    target_name: str,
    description: str,
    *,
    on_handoff: Callable[[HandoffContext], Awaitable[None]] | None = None,
) -> ToolDefinition:
    """Create a ToolDefinition the LLM calls to hand off to another agent.

    The generated tool name follows the ``transfer_to_<normalized_name>``
    convention that AgentRunner uses to detect handoff calls.

    Args:
        target_name: The name of the target agent (will be normalized).
        description: Human-readable description of when to hand off.
        on_handoff: Optional async callback invoked just before execution.

    Returns:
        A ToolDefinition representing the handoff action.
    """
    normalized = target_name.lower().replace(" ", "_").replace("-", "_")
    tool_name = f"transfer_to_{normalized}"

    if on_handoff is not None:
        _HANDOFF_CALLBACKS[tool_name] = on_handoff

    return ToolDefinition(
        name=tool_name,
        description=description,
        parameters=[
            ToolParameter(
                name="task",
                type="string",
                description="Clear description of what the target agent should accomplish.",
                required=True,
            ),
            ToolParameter(
                name="context_summary",
                type="string",
                description="Brief summary of relevant context from this conversation.",
                required=False,
            ),
            ToolParameter(
                name="reason",
                type="string",
                description="Why you are handing off rather than handling this yourself.",
                required=False,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------


class AgentRegistry:
    """Registry mapping agent names to their runners and definitions.

    Supports in-process handoffs. For cross-process (A2A), use the server
    endpoints exposed via create_app().

    Args:
        default_policy: Fallback policy for agents registered without one.
    """

    def __init__(self, default_policy: HandoffPolicy | None = None) -> None:
        self._agents: dict[str, tuple[Any, AgentDefinition, HandoffPolicy]] = {}
        self._default_policy = default_policy or HandoffPolicy()

    def register(
        self,
        runner: Any,
        agent_def: AgentDefinition,
        *,
        policy: HandoffPolicy | None = None,
    ) -> None:
        """Register an agent runner and its definition.

        Args:
            runner: An AgentRunner instance (typed Any to avoid circular import).
            agent_def: The agent's definition/blueprint.
            policy: Optional handoff policy; falls back to default_policy.
        """
        self._agents[agent_def.name] = (runner, agent_def, policy or self._default_policy)

    def get(self, name: str) -> tuple[Any, AgentDefinition, HandoffPolicy]:
        """Return (runner, agent_def, policy) for the given agent name.

        Args:
            name: The registered agent name.

        Raises:
            HandoffError: code="AGENT_NOT_FOUND" when name is not registered.
        """
        if name not in self._agents:
            raise HandoffError(
                f"Agent '{name}' not found in registry.",
                code="AGENT_NOT_FOUND",
                hint=(
                    f"Register the agent first: registry.register(runner, agent_def). "
                    f"Available: {list(self._agents)}"
                ),
            )
        return self._agents[name]

    def list_agents(self) -> list[str]:
        """Return all registered agent names."""
        return list(self._agents.keys())

    def generate_agent_card(self, agent_def: AgentDefinition, base_url: str) -> AgentCard:
        """Build an A2A-compatible AgentCard for the given agent definition.

        Args:
            agent_def: The agent whose card to generate.
            base_url: Base URL of the server (used to construct the agent URL).

        Returns:
            An AgentCard ready for serialization at /.well-known/agent.json.
        """
        skills = [
            AgentSkill(id=tool_name, name=tool_name, description="")
            for tool_name in (agent_def.tools or [])
        ]
        return AgentCard(
            name=agent_def.name,
            description=agent_def.system_prompt or agent_def.name,
            url=f"{base_url}/a2a/agents/{agent_def.name}",
            skills=skills,
        )


# ---------------------------------------------------------------------------
# Handoff executor
# ---------------------------------------------------------------------------


class HandoffExecutor:
    """Executes validated handoff requests against the AgentRegistry.

    Security guarantees enforced here (allowlist, depth, sanitize).
    Observability events written here (not in AgentRunner).

    Args:
        registry: Agent registry for target lookup.
        event_log: Optional event log for audit trail.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        event_log: EventLog | None = None,
    ) -> None:
        self._registry = registry
        self._event_log = event_log

    async def execute(self, request: HandoffRequest) -> HandoffResult:
        """Execute a validated handoff request.

        Args:
            request: The immutable handoff request to process.

        Returns:
            HandoffResult describing the outcome.

        Raises:
            HandoffError: When the handoff is rejected (policy) or fails.
        """
        start = time.monotonic()

        if self._event_log:
            await self._event_log.append(
                EventType.HANDOFF_INITIATED,
                {
                    "handoff_id": request.id,
                    "source": request.source_agent_id,
                    "target": request.target_agent_name,
                    "depth": request.handoff_depth,
                    "task_len": len(request.context.task),
                    "trace_context": request.trace_context,
                },
            )

        try:
            runner, target_def, policy = self._registry.get(request.target_agent_name)

            if (
                policy.allowed_targets is not None
                and request.source_agent_id not in policy.allowed_targets
            ):
                raise HandoffError(
                    f"Agent '{request.source_agent_id}' is not permitted to hand off "
                    f"to '{request.target_agent_name}'.",
                    code="HANDOFF_NOT_PERMITTED",
                    hint=(
                        f"Update HandoffPolicy.allowed_targets for '{request.target_agent_name}'."
                    ),
                )

            if request.handoff_depth >= policy.max_depth:
                raise HandoffError(
                    f"Handoff chain depth {request.handoff_depth} exceeds max {policy.max_depth}.",
                    code="MAX_HANDOFF_DEPTH_EXCEEDED",
                    hint="Increase HandoffPolicy.max_depth or restructure the agent workflow.",
                )

            safe_context = _sanitize_context(request.context)

            tool_name = (
                f"transfer_to_"
                f"{request.target_agent_name.lower().replace(' ', '_').replace('-', '_')}"
            )
            callback = _HANDOFF_CALLBACKS.get(tool_name)
            if callback is not None:
                await callback(safe_context)

            input_text = safe_context.task
            messages_prefix: list[Message] = list(safe_context.relevant_messages)
            if safe_context.context_summary:
                messages_prefix.insert(
                    0,
                    Message(
                        role=Role.SYSTEM,
                        content=f"[Handoff context]\n{safe_context.context_summary}",
                    ),
                )
            if safe_context.constraints:
                constraint_text = "Constraints:\n" + "\n".join(
                    f"- {c}" for c in safe_context.constraints
                )
                messages_prefix.append(Message(role=Role.SYSTEM, content=constraint_text))

            child_session = f"{request.source_session_id}__handoff_{request.id[:8]}"
            result = await runner.run(
                target_def,
                input_text,
                session_id=child_session,
                _handoff_depth=request.handoff_depth + 1,
                _prefix_messages=messages_prefix if messages_prefix else None,
            )

            handoff_result = HandoffResult(
                request_id=request.id,
                output=result.output,
                messages=result.messages,
                token_usage=result.token_usage,
                status="completed",
                duration_seconds=time.monotonic() - start,
            )

        except HandoffError:
            raise
        except Exception as exc:
            handoff_result = HandoffResult(
                request_id=request.id,
                output=None,
                status="failed",
                error=str(exc),
                duration_seconds=time.monotonic() - start,
            )
            if self._event_log:
                await self._event_log.append(
                    EventType.HANDOFF_FAILED,
                    {"handoff_id": request.id, "error": str(exc)},
                )
            raise HandoffError(
                f"Handoff to '{request.target_agent_name}' failed: {exc}",
                code="HANDOFF_EXECUTION_FAILED",
            ) from exc

        if self._event_log:
            await self._event_log.append(
                EventType.HANDOFF_COMPLETED,
                {
                    "handoff_id": request.id,
                    "target": request.target_agent_name,
                    "status": handoff_result.status,
                    "duration_s": round(handoff_result.duration_seconds, 3),
                    "output_len": len(handoff_result.output or ""),
                },
            )

        return handoff_result
