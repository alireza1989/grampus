"""Pre-built graph node handler factories for common agent steps."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from nexus.core.logging import get_logger
from nexus.core.types import AgentState, AgentStatus, Message, Role, TokenUsage

_log = get_logger(__name__)

NodeHandler = Callable[[AgentState], Coroutine[Any, Any, AgentState]]


def llm_node(
    model_client: Any,
    *,
    model: str,
    system_prompt: str = "",
    extract_tool_calls: bool = True,
) -> NodeHandler:
    """Return a handler that calls the LLM with the current message window.

    Appends the assistant response as a new Message to state.messages.
    Accumulates token usage in state.total_token_usage.
    Sets state.status = AgentStatus.RUNNING.
    """

    async def handler(state: AgentState) -> AgentState:
        messages = list(state.messages)
        if system_prompt:
            messages = [Message(role=Role.SYSTEM, content=system_prompt)] + messages

        response = await model_client.complete(
            messages=messages,
            model=model,
        )

        new_state = state.model_copy(deep=True)
        new_state.messages.append(
            Message(
                role=Role.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls if extract_tool_calls else [],
            )
        )
        new_state.total_token_usage = _accumulate_usage(
            new_state.total_token_usage, response.token_usage
        )
        new_state.status = AgentStatus.RUNNING
        _log.debug("llm_node.complete", model=model, tokens=response.token_usage.total_tokens)
        return new_state

    return handler


def tool_node(executor: Any) -> NodeHandler:
    """Return a handler that executes all pending tool calls in state.

    Reads tool calls from the last assistant message.
    Executes each, appends a TOOL message containing the ToolResult objects.
    Sets state.status = AgentStatus.RUNNING when tool calls are found.
    Passes state through unchanged when no tool calls are pending.
    """

    async def handler(state: AgentState) -> AgentState:
        pending = _last_assistant_tool_calls(state)
        if not pending:
            return state

        new_state = state.model_copy(deep=True)
        results = []
        for tc in pending:
            result = await executor.execute(tc)
            results.append(result)
            _log.debug("tool_node.executed", tool=tc.name, call_id=tc.id)

        new_state.messages.append(Message(role=Role.TOOL, content=None, tool_results=results))
        new_state.status = AgentStatus.RUNNING
        return new_state

    return handler


def human_node(prompt: str = "Waiting for human input...") -> NodeHandler:
    """Return a handler that pauses execution for human review.

    Sets state.status = AgentStatus.WAITING_FOR_HUMAN.
    Appends a system message with the prompt text.
    Returns immediately — the caller is responsible for resuming.
    """

    async def handler(state: AgentState) -> AgentState:
        new_state = state.model_copy(deep=True)
        new_state.messages.append(Message(role=Role.SYSTEM, content=prompt))
        new_state.status = AgentStatus.WAITING_FOR_HUMAN
        return new_state

    return handler


def conditional_node(
    condition_fn: Callable[[AgentState], Coroutine[Any, Any, str]],
) -> NodeHandler:
    """Return a pass-through handler marking a conditional decision point.

    The handler returns state unchanged — routing is handled via
    add_conditional_edge() on the parent Graph using the same condition_fn.
    """

    async def handler(state: AgentState) -> AgentState:
        return state

    return handler


HumanNode = human_node
"""Alias for human_node factory — class-style import convenience."""


def subgraph_node(subgraph: Any) -> NodeHandler:
    """Return a handler that executes a nested Graph as a single node.

    Passes current state into the subgraph's execute() and returns
    the subgraph's final state.
    """

    async def handler(state: AgentState) -> AgentState:
        result: AgentState = await subgraph.execute(state)
        return result

    return handler


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _accumulate_usage(
    existing: TokenUsage | None,
    incoming: TokenUsage,
) -> TokenUsage:
    """Add incoming token usage onto existing totals."""
    if existing is None:
        return incoming
    return TokenUsage(
        input_tokens=existing.input_tokens + incoming.input_tokens,
        output_tokens=existing.output_tokens + incoming.output_tokens,
        total_tokens=existing.total_tokens + incoming.total_tokens,
        cost_usd=existing.cost_usd + incoming.cost_usd,
        model=incoming.model,
    )


def _last_assistant_tool_calls(state: AgentState) -> list[Any]:
    """Return tool_calls from the last assistant message, or empty list."""
    for msg in reversed(state.messages):
        if msg.role == Role.ASSISTANT:
            return list(msg.tool_calls)
    return []


def _last_assistant_content(state: AgentState) -> str:
    """Return content of the last ASSISTANT message, or empty string."""
    for msg in reversed(state.messages):
        if msg.role == Role.ASSISTANT and msg.content is not None:
            return msg.content
    return ""


def _last_user_message(state: AgentState) -> str:
    """Return content of the last USER message in state.

    Raises:
        ValueError: If no USER message is found.
    """
    for msg in reversed(state.messages):
        if msg.role == Role.USER and msg.content is not None:
            return msg.content
    raise ValueError("No USER message found in agent state")


def uncertainty_guard_node(
    monitor: Any,
    *,
    step_type: str = "decision",
    escalate_node: str | None = None,
) -> NodeHandler:
    """Return a handler that evaluates uncertainty and optionally escalates.

    Reads the last assistant message content, passes it to
    monitor.observe_llm_response(), updates state.metadata["uncertainty"],
    and sets WAITING_FOR_HUMAN on escalation. Use as an explicit uncertainty
    checkpoint between graph nodes.

    Args:
        monitor: UncertaintyMonitor (duck-typed to avoid circular import).
        step_type: Step category passed to the monitor (default "decision").
        escalate_node: When set, writes True to metadata["uncertainty_escalate"]
            so a conditional_edge can route to a human_node.
    """

    async def handler(state: AgentState) -> AgentState:
        content = _last_assistant_content(state)
        from nexus.orchestration.uncertainty.types import UncertaintyAction

        step_unc, action = await monitor.observe_llm_response(
            response_text=content,
            step_id=f"guard_{state.current_step}",
            step_type=step_type,
        )
        _ = step_unc
        new_state = state.model_copy(deep=True)
        new_state.metadata["uncertainty"] = monitor.summary_metadata()
        if action in (UncertaintyAction.PAUSE_FOR_HUMAN, UncertaintyAction.ABORT):
            new_state.status = AgentStatus.WAITING_FOR_HUMAN
            if escalate_node:
                new_state.metadata["uncertainty_escalate"] = True
        return new_state

    return handler


def planning_node(
    planning_runner: Any,
    agent_def: Any,
    *,
    tool_names: list[str] | None = None,
    memory_context_key: str = "memory_context",
) -> NodeHandler:
    """Return a handler that runs PlanningRunner and injects PlanResult into state.

    Extracts task from the last USER message. Reads optional memory_context from
    state.metadata[memory_context_key]. Appends the final synthesized answer as
    an ASSISTANT message and stores the full PlanResult dict in state.metadata.

    Args:
        planning_runner: PlanningRunner instance (duck-typed to avoid circular import).
        agent_def: AgentDefinition for subgoal execution.
        tool_names: Optional list of available tool names passed to the planner.
        memory_context_key: Key in state.metadata to read memory context from.
    """

    async def handler(state: AgentState) -> AgentState:
        task = _last_user_message(state)
        memory_ctx = state.metadata.get(memory_context_key, "")
        result = await planning_runner.run(
            task,
            agent_def,
            tool_names=tool_names,
            memory_context=memory_ctx,
        )
        new_state = state.model_copy(deep=True)
        new_state.messages.append(
            Message(
                role=Role.ASSISTANT,
                content=result.final_output,
                metadata={
                    "plan_result": result.model_dump(),
                    "replans_triggered": result.replans_triggered,
                    "subgoals_completed": len(result.completed_subgoals),
                },
            )
        )
        if result.total_token_usage:
            new_state.total_token_usage = _accumulate_usage(
                new_state.total_token_usage, result.total_token_usage
            )
        new_state.status = AgentStatus.COMPLETED if result.success else AgentStatus.FAILED
        new_state.metadata["plan_result"] = result.model_dump()
        return new_state

    return handler


def debate_node(
    orchestrator: Any,
    *,
    question_extractor: Callable[[AgentState], str] | None = None,
    on_escalate: str | None = None,
) -> NodeHandler:
    """Return a handler that runs multi-agent debate and injects the result into state.

    Args:
        orchestrator: A DebateOrchestrator (duck-typed to avoid circular import).
        question_extractor: Extracts the debate question from AgentState.
            Defaults to the content of the last USER message.
        on_escalate: When set, writes ``True`` to ``state.metadata["debate_escalate"]``
            if the result has escalate_to_human=True, enabling conditional graph routing.
    """

    async def handler(state: AgentState) -> AgentState:
        question = question_extractor(state) if question_extractor else _last_user_message(state)
        result = await orchestrator.run(question)
        new_state = state.model_copy(deep=True)
        new_state.messages.append(
            Message(
                role=Role.ASSISTANT,
                content=result.final_answer,
                metadata={
                    "debate_result": result.model_dump(),
                    "debate_confidence": result.confidence,
                    "debate_escalate": result.escalate_to_human,
                    "debate_rounds": result.total_rounds_run,
                    "debate_routing": result.routing_decision,
                },
            )
        )
        if result.total_token_usage:
            new_state.total_token_usage = _accumulate_usage(
                new_state.total_token_usage, result.total_token_usage
            )
        if on_escalate and result.escalate_to_human:
            new_state.metadata["debate_escalate"] = True
        new_state.status = AgentStatus.RUNNING
        return new_state

    return handler
