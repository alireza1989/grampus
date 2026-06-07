"""Tests for NexusA2AExecutor."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events.event_queue import EventQueueLegacy
from a2a.types.a2a_pb2 import (
    Message,
    Part,
    Role,
    SendMessageRequest,
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from google.protobuf.json_format import MessageToDict

from nexus.core.types import AgentDefinition, ExecutionResult, AgentStatus, TokenUsage


def _make_request_context(
    text: str = "hello",
    task_id: str = "task-1",
    context_id: str = "ctx-1",
    existing_task: Task | None = None,
) -> RequestContext:
    """Build a minimal RequestContext for testing."""
    call_context = ServerCallContext()

    req = SendMessageRequest()
    req.message.role = Role.ROLE_USER
    req.message.task_id = task_id
    req.message.context_id = context_id
    p = req.message.parts.add()
    p.text = text

    return RequestContext(
        call_context=call_context,
        request=req,
        task_id=task_id,
        context_id=context_id,
        task=existing_task,
    )


def _make_runner(output: str = "done", raise_exc: Exception | None = None) -> Any:
    runner = MagicMock()
    if raise_exc:
        runner.run = AsyncMock(side_effect=raise_exc)
        runner.stream = MagicMock()
    else:
        result = ExecutionResult(
            output=output,
            messages=[],
            tool_calls_made=0,
            token_usage=TokenUsage(
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                cost_usd=0.001,
                model="claude-3-5-haiku-20241022",
            ),
            duration_seconds=0.1,
            steps_taken=1,
            status=AgentStatus.COMPLETED,
        )
        runner.run = AsyncMock(return_value=result)
        runner.stream = MagicMock()
    return runner


def _make_agent_def() -> AgentDefinition:
    return AgentDefinition(name="test-agent", model="claude-3-5-haiku-20241022")


class _CollectingQueue(EventQueueLegacy):
    """EventQueue that records all enqueued events for assertion."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[Any] = []

    async def enqueue_event(self, event: Any) -> None:
        self.events.append(event)


async def test_execute_emits_working_then_completed() -> None:
    from nexus.orchestration.a2a.executor import NexusA2AExecutor

    runner = _make_runner("result text")
    agent_def = _make_agent_def()
    executor = NexusA2AExecutor(runner=runner, agent_def=agent_def)

    context = _make_request_context("do something")
    queue = _CollectingQueue()

    await executor.execute(context, queue)

    assert len(queue.events) == 2
    working_evt, done_evt = queue.events
    assert isinstance(working_evt, TaskStatusUpdateEvent)
    assert working_evt.status.state == TaskState.TASK_STATE_WORKING

    assert isinstance(done_evt, TaskStatusUpdateEvent)
    assert done_evt.status.state == TaskState.TASK_STATE_COMPLETED
    assert done_evt.task_id == "task-1"
    assert done_evt.context_id == "ctx-1"


async def test_execute_emits_failed_on_runner_exception() -> None:
    from nexus.orchestration.a2a.executor import NexusA2AExecutor

    runner = _make_runner(raise_exc=RuntimeError("boom"))
    agent_def = _make_agent_def()
    executor = NexusA2AExecutor(runner=runner, agent_def=agent_def)

    context = _make_request_context()
    queue = _CollectingQueue()

    await executor.execute(context, queue)

    states = [e.status.state for e in queue.events if isinstance(e, TaskStatusUpdateEvent)]
    assert TaskState.TASK_STATE_WORKING in states
    assert TaskState.TASK_STATE_FAILED in states


async def test_execute_extracts_text_from_message_parts() -> None:
    from nexus.orchestration.a2a.executor import NexusA2AExecutor

    runner = _make_runner()
    agent_def = _make_agent_def()
    executor = NexusA2AExecutor(runner=runner, agent_def=agent_def)

    context = _make_request_context(text="my task text")
    queue = _CollectingQueue()

    await executor.execute(context, queue)

    runner.run.assert_called_once()
    call_args = runner.run.call_args
    assert "my task text" in call_args.args or any(
        "my task text" in str(v) for v in call_args.kwargs.values()
    )


async def test_cancel_emits_canceled_state() -> None:
    from nexus.orchestration.a2a.executor import NexusA2AExecutor

    runner = _make_runner()
    agent_def = _make_agent_def()
    executor = NexusA2AExecutor(runner=runner, agent_def=agent_def)

    context = _make_request_context()
    queue = _CollectingQueue()

    await executor.cancel(context, queue)

    assert len(queue.events) == 1
    evt = queue.events[0]
    assert isinstance(evt, TaskStatusUpdateEvent)
    assert evt.status.state == TaskState.TASK_STATE_CANCELED


async def test_execute_output_text_in_completed_message() -> None:
    from nexus.orchestration.a2a.executor import NexusA2AExecutor

    runner = _make_runner("the answer")
    agent_def = _make_agent_def()
    executor = NexusA2AExecutor(runner=runner, agent_def=agent_def)

    context = _make_request_context()
    queue = _CollectingQueue()

    await executor.execute(context, queue)

    completed_evt = queue.events[-1]
    assert isinstance(completed_evt, TaskStatusUpdateEvent)
    msg = completed_evt.status.message
    text_parts = [p.text for p in msg.parts if p.HasField("text")]
    assert any("the answer" in t for t in text_parts)


async def test_execute_builds_history_from_context_task() -> None:
    from nexus.orchestration.a2a.executor import NexusA2AExecutor

    existing_task = Task()
    existing_task.id = "task-1"
    existing_task.context_id = "ctx-1"
    prior_msg = existing_task.history.add()
    prior_msg.role = Role.ROLE_USER
    p = prior_msg.parts.add()
    p.text = "prior message"

    runner = _make_runner("follow-up result")
    agent_def = _make_agent_def()
    executor = NexusA2AExecutor(runner=runner, agent_def=agent_def)

    context = _make_request_context(
        text="follow-up",
        task_id="task-1",
        context_id="ctx-1",
        existing_task=existing_task,
    )
    queue = _CollectingQueue()

    await executor.execute(context, queue)

    runner.run.assert_called_once()
    completed = [
        e
        for e in queue.events
        if isinstance(e, TaskStatusUpdateEvent) and e.status.state == TaskState.TASK_STATE_COMPLETED
    ]
    assert len(completed) == 1
