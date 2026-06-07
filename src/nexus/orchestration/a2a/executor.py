"""NexusA2AExecutor — bridges the A2A protocol into an AgentRunner execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    from a2a.types.a2a_pb2 import (
        Message,
        Part,
        Role,
        TaskState,
        TaskStatus,
        TaskStatusUpdateEvent,
    )

    _HAS_A2A = True
except ImportError:  # pragma: no cover
    _HAS_A2A = False
    AgentExecutor = object  # type: ignore[misc,assignment]
    RequestContext = object  # type: ignore[misc,assignment]
    EventQueue = object  # type: ignore[misc,assignment]

from nexus.core.logging import get_logger
from nexus.core.types import AgentDefinition

if TYPE_CHECKING:
    from nexus.orchestration.runner import AgentRunner

_log = get_logger(__name__)


def _build_text_message(text: str) -> Message:
    """Build an A2A Message with a single text Part."""
    msg = Message()
    msg.role = Role.ROLE_AGENT
    p: Part = msg.parts.add()
    p.text = text
    return msg


def _build_status_event(
    task_id: str,
    context_id: str,
    state: Any,
    output_text: str | None = None,
) -> TaskStatusUpdateEvent:
    """Build a TaskStatusUpdateEvent proto."""
    status = TaskStatus()
    status.state = state
    if output_text is not None:
        status.message.CopyFrom(_build_text_message(output_text))

    event = TaskStatusUpdateEvent()
    event.task_id = task_id
    event.context_id = context_id
    event.status.CopyFrom(status)
    return event


class NexusA2AExecutor(AgentExecutor):
    """Bridges the A2A AgentExecutor protocol into a Nexus AgentRunner.

    Args:
        runner: AgentRunner instance that processes user messages.
        agent_def: AgentDefinition blueprint for the runner.
        event_publisher: Optional DaprPubSub for observability events.
    """

    def __init__(
        self,
        runner: AgentRunner,
        agent_def: AgentDefinition,
        event_publisher: Any | None = None,
    ) -> None:
        if not _HAS_A2A:
            from nexus.core.errors import ToolError

            raise ToolError(
                "a2a-sdk is not installed. Install with: pip install 'nexus-ai[a2a]'",
                code="A2A_SDK_MISSING",
                hint="pip install 'nexus-ai[a2a]'",
            )
        self._runner = runner
        self._agent_def = agent_def
        self._event_publisher = event_publisher

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the agent for the given A2A request context.

        Emits WORKING → runs AgentRunner → emits COMPLETED or FAILED.
        """
        task_id = context.task_id or ""
        context_id = context.context_id or ""

        await event_queue.enqueue_event(
            _build_status_event(task_id, context_id, TaskState.TASK_STATE_WORKING)
        )

        user_text = context.get_user_input()
        session_id = f"a2a-{task_id}" if task_id else "a2a-unknown"

        try:
            result = await self._runner.run(
                self._agent_def,
                user_text,
                session_id=session_id,
            )
            output = result.output or ""
            await event_queue.enqueue_event(
                _build_status_event(
                    task_id,
                    context_id,
                    TaskState.TASK_STATE_COMPLETED,
                    output,
                )
            )
        except Exception as exc:
            _log.exception("a2a_executor_run_failed", error=str(exc))
            await event_queue.enqueue_event(
                _build_status_event(
                    task_id,
                    context_id,
                    TaskState.TASK_STATE_FAILED,
                    str(exc),
                )
            )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancel the current task by emitting a CANCELED status event."""
        task_id = context.task_id or ""
        context_id = context.context_id or ""
        await event_queue.enqueue_event(
            _build_status_event(task_id, context_id, TaskState.TASK_STATE_CANCELED)
        )
