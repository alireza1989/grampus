"""AgentRunner — ReAct loop, memory integration, cost tracking, state persistence."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from nexus.core.errors import OrchestrationError
from nexus.core.logging import get_logger
from nexus.core.types import (
    AgentDefinition,
    AgentState,
    AgentStatus,
    ExecutionResult,
    Message,
    Role,
    TokenUsage,
    ToolCall,
    ToolResult,
)
from nexus.orchestration.model_router import ModelSpec, ModelTier

if TYPE_CHECKING:
    from nexus.memory.manager import MemoryManager, MemoryRecallResult
    from nexus.orchestration.cost_tracker import CostSummary, CostTracker
    from nexus.tools.executor import ToolExecutor

_log = get_logger(__name__)


class RunnerConfig(BaseModel):
    """Tuning parameters for AgentRunner."""

    max_iterations: int = 10
    memory_top_k: int = 5
    enable_memory: bool = True
    react_pattern: bool = True


class AgentRunner:
    """Main agent execution loop implementing the ReAct pattern.

    Observe → Think (LLM) → Act (tools) → repeat until done or max_iterations.

    Args:
        model_client: LLM client for completions (duck-typed as Any).
        tool_executor: Executor for tool calls.
        memory_manager: Optional memory facade. When None, memory is disabled.
        cost_tracker: Optional cost tracker. When None, cost is not tracked.
        state_store: Optional Dapr state store for persisting AgentState between
            turns. When None, state is only in-memory.
        config: Tuning parameters.
    """

    def __init__(
        self,
        model_client: Any,
        tool_executor: ToolExecutor,
        *,
        memory_manager: MemoryManager | None = None,
        cost_tracker: CostTracker | None = None,
        state_store: Any | None = None,
        config: RunnerConfig | None = None,
    ) -> None:
        self._model_client = model_client
        self._tool_executor = tool_executor
        self._memory_manager = memory_manager
        self._cost_tracker = cost_tracker
        self._state_store = state_store
        self._config = config or RunnerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        agent_def: AgentDefinition,
        user_input: str,
        *,
        session_id: str,
        agent_state: AgentState | None = None,
    ) -> ExecutionResult:
        """Execute the agent loop for one user turn.

        Args:
            agent_def: Blueprint describing model, tools, and behaviour config.
            user_input: The user's message or task.
            session_id: Unique identifier for this session.
            agent_state: Pre-existing state to restore, or None to build fresh.

        Returns:
            ExecutionResult with output, messages, costs, and timing.

        Raises:
            OrchestrationError: code="MAX_ITERATIONS_EXCEEDED" when loop limit
                is reached without a final answer.
            BudgetExceededError: Propagated from CostTracker when budget is hit.
        """
        start = time.monotonic()
        state = agent_state or self._build_state(agent_def, session_id)
        state.status = AgentStatus.RUNNING

        if self._config.enable_memory and self._memory_manager:
            await self._recall_context(user_input, state)

        state.messages.append(Message(role=Role.USER, content=user_input))

        accumulated = _zero_usage(agent_def.model)
        tool_calls_made = 0
        steps = 0
        hit_limit = True

        for i in range(self._config.max_iterations):
            steps = i + 1
            response = await self._model_client.complete(
                messages=state.messages,
                model=agent_def.model,
                temperature=agent_def.temperature,
            )

            accumulated = _add_usage(accumulated, response.token_usage)

            if self._cost_tracker:
                await self._cost_tracker.record(
                    response.token_usage,
                    step_name=f"step_{steps}",
                    model_spec=_minimal_spec(response.model),
                )

            state.messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            if response.tool_calls:
                results = await self._execute_tool_calls(response.tool_calls, state)
                tool_calls_made += len(results)
                state.messages.append(Message(role=Role.TOOL, tool_results=results))
                if state.status == AgentStatus.WAITING_FOR_HUMAN:
                    hit_limit = False
                    break
            else:
                hit_limit = False
                break

        if hit_limit:
            raise OrchestrationError(
                f"Max iterations ({self._config.max_iterations}) exceeded without final answer (MAX_ITERATIONS_EXCEEDED)",
                code="MAX_ITERATIONS_EXCEEDED",
            )

        final_output = _extract_final_output(state.messages)

        if self._config.enable_memory and self._memory_manager:
            await self._store_memory(user_input, final_output, session_id)

        if state.status == AgentStatus.RUNNING:
            state.status = AgentStatus.COMPLETED
        state.updated_at = datetime.now(UTC)
        state.metadata["agent_def"] = agent_def.model_dump()

        await self._persist_state(agent_def, session_id, state)

        duration = time.monotonic() - start
        _log.debug(
            "agent_run_complete",
            agent=agent_def.name,
            steps=steps,
            tool_calls=tool_calls_made,
            duration=duration,
        )
        return ExecutionResult(
            output=final_output,
            messages=state.messages,
            tool_calls_made=tool_calls_made,
            token_usage=accumulated,
            duration_seconds=duration,
            steps_taken=steps,
            status=state.status,
        )

    async def resume(
        self,
        agent_id: str,
        session_id: str,
        human_response: str,
    ) -> ExecutionResult:
        """Resume a paused agent (status=WAITING_FOR_HUMAN).

        Loads state from state_store, appends human_response as a user message,
        then re-enters the ReAct loop.

        Args:
            agent_id: The agent name used as key (matches agent_def.name).
            session_id: Session to restore.
            human_response: Human reply to inject as a user message.

        Raises:
            OrchestrationError: code="NO_STATE_FOUND" when state not in store.
            OrchestrationError: code="AGENT_NOT_WAITING" when not paused.
        """
        state = await self._load_state(agent_id, session_id)
        if state.status != AgentStatus.WAITING_FOR_HUMAN:
            raise OrchestrationError(
                f"Agent '{agent_id}' is not waiting for human input (status={state.status})",
                code="AGENT_NOT_WAITING",
            )
        agent_def = self._reconstruct_agent_def(agent_id, state)
        return await self.run(agent_def, human_response, session_id=session_id, agent_state=state)

    def cost_summary(self) -> CostSummary | None:
        """Return cost accumulation summary, or None when no tracker is configured."""
        if self._cost_tracker is None:
            return None
        return self._cost_tracker.summary()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_state(self, agent_def: AgentDefinition, session_id: str) -> AgentState:
        messages: list[Message] = []
        if agent_def.system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=agent_def.system_prompt))
        return AgentState(
            agent_id=agent_def.name,
            session_id=session_id,
            messages=messages,
        )

    async def _recall_context(self, query: str, state: AgentState) -> None:
        result = await self._memory_manager.recall(  # type: ignore[union-attr]
            query, top_k=self._config.memory_top_k
        )
        context = _format_recall(result)
        if context:
            state.messages.append(Message(role=Role.SYSTEM, content=context))

    async def _execute_tool_calls(
        self, tool_calls: list[ToolCall], state: AgentState | None = None
    ) -> list[ToolResult]:
        results = []
        for tc in tool_calls:
            if tc.name == "human_input":
                if state is not None:
                    state.status = AgentStatus.WAITING_FOR_HUMAN
                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        output="Paused: waiting for human input.",
                        error=None,
                        duration_ms=0,
                    )
                )
            else:
                result = await self._tool_executor.execute(tc)
                results.append(result)
        return results

    async def _store_memory(
        self, user_input: str, final_output: str | None, session_id: str
    ) -> None:
        summary = f"{user_input}\n{final_output or ''}"
        await self._memory_manager.remember(summary, session_id=session_id)  # type: ignore[union-attr]

    async def _persist_state(
        self, agent_def: AgentDefinition, session_id: str, state: AgentState
    ) -> None:
        if self._state_store is None:
            return
        entity_id = f"agent:{agent_def.name}:{session_id}"
        await self._state_store.save("runner", entity_id, state)

    async def _load_state(self, agent_id: str, session_id: str) -> AgentState:
        if self._state_store is None:
            raise OrchestrationError(
                f"No state store configured; cannot resume agent '{agent_id}'",
                code="NO_STATE_FOUND",
            )
        entity_id = f"agent:{agent_id}:{session_id}"
        state: AgentState | None
        state, _ = await self._state_store.get("runner", entity_id, AgentState)
        if state is None:
            raise OrchestrationError(
                f"No state found for agent='{agent_id}' session='{session_id}'",
                code="NO_STATE_FOUND",
            )
        return state

    def _reconstruct_agent_def(self, agent_id: str, state: AgentState) -> AgentDefinition:
        stored = state.metadata.get("agent_def")
        if isinstance(stored, dict):
            return AgentDefinition(**stored)
        return AgentDefinition(name=agent_id, model="")


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _zero_usage(model: str) -> TokenUsage:
    return TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0, cost_usd=0.0, model=model)


def _add_usage(a: TokenUsage, b: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
        cost_usd=a.cost_usd + b.cost_usd,
        model=b.model,
    )


def _minimal_spec(model_id: str) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        tier=ModelTier.BALANCED,
        provider="unknown",
        input_cost_per_1k_tokens=0.0,
        output_cost_per_1k_tokens=0.0,
        context_window=200_000,
    )


def _extract_final_output(messages: list[Message]) -> str | None:
    for msg in reversed(messages):
        if msg.role == Role.ASSISTANT and msg.content:
            return msg.content
    return None


def _format_recall(result: MemoryRecallResult) -> str:
    parts: list[str] = []
    for rec in result.episodic:
        # RetrievedRecord wraps EpisodicRecord under .record
        inner = getattr(rec, "record", rec)
        text = getattr(inner, "content", None)
        if text:
            parts.append(str(text))
    for fact in result.semantic:
        subj = getattr(fact, "subject", "")
        pred = getattr(fact, "predicate", "")
        obj = getattr(fact, "object_value", "")
        line = f"{subj} {pred} {obj}".strip()
        if line:
            parts.append(line)
    return "\n".join(parts)
