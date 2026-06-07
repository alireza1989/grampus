"""Agent handoff primitives — OpenAI-style transfer tools, A2A Protocol v1.2 cards."""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from nexus.core.errors import HandoffError, OrchestrationError
from nexus.core.types import (
    AgentDefinition,
    Message,
    Role,
    TokenUsage,
    ToolDefinition,
    ToolParameter,
)
from nexus.observability.events import EventLog, EventType

if TYPE_CHECKING:
    from nexus.orchestration.a2a.client import A2AAgentClient
    from nexus.orchestration.a2a.registry import AgentEntry

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
# A2A response helpers
# ---------------------------------------------------------------------------


def _extract_output_from_a2a_response(response: Any) -> str | None:
    """Extract plain text output from an A2A send_message result dict."""
    if response is None:
        return None
    if isinstance(response, dict):
        status = response.get("status", {})
        msg = status.get("message", {})
        parts = msg.get("parts", []) if isinstance(msg, dict) else []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
        return "\n".join(texts) or None
    # Handle proto-style response objects
    try:
        task = getattr(response, "task", None) or response
        status = getattr(task, "status", None)
        message = getattr(status, "message", None)
        parts = getattr(message, "parts", [])
        texts = []
        for part in parts:
            if hasattr(part, "HasField") and part.HasField("text") or hasattr(part, "text"):
                texts.append(part.text)
        return "\n".join(t for t in texts if t) or None
    except Exception:
        return str(response)


def _extract_output_from_jsonrpc_response(data: dict[str, Any]) -> str | None:
    """Extract plain text from a JSON-RPC A2A message/send response dict."""
    result = data.get("result", {}) if isinstance(data, dict) else {}
    status = result.get("status", {}) if isinstance(result, dict) else {}
    message = status.get("message", {}) if isinstance(status, dict) else {}
    parts = message.get("parts", []) if isinstance(message, dict) else []
    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
    return "\n".join(texts) or None


# ---------------------------------------------------------------------------
# Handoff executor
# ---------------------------------------------------------------------------


class HandoffExecutor:
    """Executes validated handoff requests against the AgentRegistry.

    Supports two registry types:
    - The legacy ``AgentRegistry`` in this module (in-process runners only).
    - The A2A ``AgentRegistry`` from ``nexus.orchestration.a2a.registry`` which
      supports local runners, Dapr sibling services, and remote A2A agents.

    Security guarantees enforced here (allowlist, depth, sanitize).
    Observability events written here (not in AgentRunner).

    Args:
        registry: Agent registry for target lookup.
        event_log: Optional event log for audit trail.
        dapr_http_port: Port of the local Dapr HTTP sidecar (default: 3500).
    """

    def __init__(
        self,
        registry: Any,
        event_log: EventLog | None = None,
        dapr_http_port: int = 3500,
    ) -> None:
        self._registry = registry
        self._event_log = event_log
        self._dapr_http_port = dapr_http_port

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
            raw_entry = self._registry.get(request.target_agent_name)

            # New A2A AgentRegistry returns AgentEntry | None
            if raw_entry is None:
                raise HandoffError(
                    f"Agent '{request.target_agent_name}' not found in registry.",
                    code="AGENT_NOT_FOUND",
                    hint=f"Register the agent first. Available: {self._registry.list_agent_names()}"
                    if hasattr(self._registry, "list_agent_names")
                    else "Register the agent first.",
                )

            if isinstance(raw_entry, tuple):
                # Legacy registry returns (runner, agent_def, policy)
                runner, target_def, policy = raw_entry
                handoff_result = await self._local_handoff(
                    runner, target_def, policy, request, start
                )
            else:
                # New AgentEntry — choose one of three communication paths:
                #
                #   Path 1 — Local runner (runner is not None):
                #     Same process or same Nexus instance. Call AgentRunner directly.
                #     No network hop. Dapr state/pub-sub still available to the target.
                #
                #   Path 2 — Dapr service invocation (dapr_app_id is not None):
                #     Another Nexus service in the same cluster. Route through the Dapr
                #     sidecar (localhost:{port}) to get automatic mTLS, retries, tracing.
                #     Use this for Nexus-to-Nexus multi-service deployments.
                #
                #   Path 3 — External A2A HTTP (remote_url / client is not None):
                #     Non-Nexus agent (LangGraph, CrewAI, etc.) or Nexus in a different
                #     cluster. Plain HTTP with JSON-RPC A2A protocol. No Dapr involved.
                entry: AgentEntry = raw_entry
                if entry.runner is not None:
                    target_def = entry.agent_def or AgentDefinition(
                        name=entry.name, model="claude-3-5-haiku-20241022"
                    )
                    handoff_result = await self._local_handoff(
                        entry.runner, target_def, HandoffPolicy(), request, start
                    )
                elif entry.dapr_app_id is not None:
                    handoff_result = await self._dapr_handoff(entry, request, start)
                elif entry.client is not None:
                    handoff_result = await self._remote_handoff(entry.client, request, start)
                else:
                    raise HandoffError(
                        f"Agent '{request.target_agent_name}' has no execution path configured.",
                        code="AGENT_NOT_CONFIGURED",
                        hint=(
                            "Register with register_local(), register_dapr_service(), "
                            "or register_remote()."
                        ),
                    )

        except HandoffError:
            raise
        except Exception as exc:
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

    # ------------------------------------------------------------------
    # Internal execution paths
    # ------------------------------------------------------------------

    async def _local_handoff(
        self,
        runner: Any,
        target_def: AgentDefinition,
        policy: HandoffPolicy,
        request: HandoffRequest,
        start: float,
    ) -> HandoffResult:
        """Execute a handoff against a local in-process AgentRunner."""
        if (
            policy.allowed_targets is not None
            and request.source_agent_id not in policy.allowed_targets
        ):
            raise HandoffError(
                f"Agent '{request.source_agent_id}' is not permitted to hand off "
                f"to '{request.target_agent_name}'.",
                code="HANDOFF_NOT_PERMITTED",
                hint=f"Update HandoffPolicy.allowed_targets for '{request.target_agent_name}'.",
            )

        if request.handoff_depth >= policy.max_depth:
            raise HandoffError(
                f"Handoff chain depth {request.handoff_depth} exceeds max {policy.max_depth}.",
                code="MAX_HANDOFF_DEPTH_EXCEEDED",
                hint="Increase HandoffPolicy.max_depth or restructure the agent workflow.",
            )

        safe_context = _sanitize_context(request.context)

        tool_name = (
            f"transfer_to_{request.target_agent_name.lower().replace(' ', '_').replace('-', '_')}"
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

        return HandoffResult(
            request_id=request.id,
            output=result.output,
            messages=result.messages,
            token_usage=result.token_usage,
            status="completed",
            duration_seconds=time.monotonic() - start,
        )

    async def _remote_handoff(
        self,
        client: A2AAgentClient,
        request: HandoffRequest,
        start: float,
    ) -> HandoffResult:
        """Execute a handoff against a remote A2A agent via A2AAgentClient."""
        safe_context = _sanitize_context(request.context)
        input_text = safe_context.task
        if safe_context.context_summary:
            input_text = f"[Context]\n{safe_context.context_summary}\n\n{input_text}"

        try:
            response = await client.send_message(input_text)
        except OrchestrationError as exc:
            raise HandoffError(
                f"Remote handoff to '{request.target_agent_name}' failed: {exc}",
                code="HANDOFF_EXECUTION_FAILED",
            ) from exc

        output = _extract_output_from_a2a_response(response)
        return HandoffResult(
            request_id=request.id,
            output=output,
            status="completed",
            duration_seconds=time.monotonic() - start,
        )

    async def _dapr_handoff(
        self,
        entry: AgentEntry,
        request: HandoffRequest,
        start: float,
    ) -> HandoffResult:
        """Call a sibling Nexus service via Dapr service invocation.

        Builds a JSON-RPC message/send payload and POSTs to
        http://localhost:{dapr_http_port}/v1.0/invoke/{app_id}/method/{method}
        so traffic routes through the Dapr sidecar for mTLS, retries, and tracing.

        Args:
            entry: AgentEntry with dapr_app_id and dapr_method set.
            request: The handoff request to forward.
            start: Monotonic start time for duration tracking.
        """
        safe_context = _sanitize_context(request.context)
        input_text = safe_context.task
        if safe_context.context_summary:
            input_text = f"[Context]\n{safe_context.context_summary}\n\n{input_text}"

        url = (
            f"http://localhost:{self._dapr_http_port}"
            f"/v1.0/invoke/{entry.dapr_app_id}/method/{entry.dapr_method}"
        )
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": input_text}],
                }
            },
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data: dict[str, Any] = response.json()
        except Exception as exc:
            raise HandoffError(
                f"Dapr handoff to '{request.target_agent_name}' "
                f"(app_id={entry.dapr_app_id}) failed: {exc}",
                code="HANDOFF_DAPR_ERROR",
                hint=(f"Ensure the target service is running with --app-id {entry.dapr_app_id}"),
            ) from exc

        output = _extract_output_from_jsonrpc_response(data)
        return HandoffResult(
            request_id=request.id,
            output=output,
            status="completed",
            duration_seconds=time.monotonic() - start,
        )
