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
