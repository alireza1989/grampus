"""Append-only, replayable agent event log."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nexus.core.logging import get_logger

logger = get_logger(__name__)


class EventType(StrEnum):
    """All structured event types emitted during agent execution."""

    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    LLM_CALLED = "agent.llm_called"
    TOOL_CALLED = "agent.tool_called"
    TOOL_RESULT = "agent.tool_result"
    MEMORY_READ = "agent.memory_read"
    MEMORY_WRITTEN = "agent.memory_written"
    SAFETY_VIOLATION = "agent.safety_violation"
    HUMAN_INPUT_REQUESTED = "agent.human_input_requested"
    BUDGET_EXCEEDED = "agent.budget_exceeded"
    HANDOFF_INITIATED = "agent.handoff_initiated"
    HANDOFF_COMPLETED = "agent.handoff_completed"
    HANDOFF_FAILED = "agent.handoff_failed"


class AgentEvent(BaseModel):
    """Immutable record of a single agent action."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    agent_id: str
    session_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = Field(default_factory=dict)
    sequence_number: int = 0


class EventLog:
    """Append-only, replayable log of agent events.

    Backed by Dapr state store when configured; falls back to in-memory list
    when state_store is None (useful for testing).

    Events are immutable once written. No update or delete operations.

    Args:
        agent_id: Scopes the log to this agent.
        session_id: Current session.
        state_store: Optional Dapr state store for persistence.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        session_id: str,
        state_store: Any | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._session_id = session_id
        self._state_store = state_store
        self._events: list[AgentEvent] = []
        self._next_seq: int = 0

    async def append(
        self, event_type: EventType, payload: dict[str, Any] | None = None
    ) -> AgentEvent:
        """Create and store an AgentEvent. Returns the stored event.

        Args:
            event_type: The type of agent action being recorded.
            payload: Arbitrary metadata about the event.

        Returns:
            The persisted AgentEvent with an auto-assigned sequence number.
        """
        event = AgentEvent(
            event_type=event_type,
            agent_id=self._agent_id,
            session_id=self._session_id,
            payload=payload or {},
            sequence_number=self._next_seq,
        )
        self._next_seq += 1
        if self._state_store is None:
            self._events.append(event)
        await self._persist(event)
        logger.debug("event_appended", event_type=event_type, seq=event.sequence_number)
        return event

    async def _persist(self, event: AgentEvent) -> None:
        if self._state_store is None:
            return
        entity_id = f"{self._session_id}:{event.sequence_number}"
        await self._state_store.save("events", entity_id, event)

    async def replay(self) -> list[AgentEvent]:
        """Return all events for this agent/session in sequence order.

        Returns:
            Ordered list of AgentEvent records from sequence 0 onward.
        """
        if self._state_store is None:
            return list(self._events)
        return await self._load_from_store(0)

    async def replay_since(self, sequence_number: int) -> list[AgentEvent]:
        """Return events with sequence_number >= the given value.

        Args:
            sequence_number: Inclusive lower bound on sequence number.

        Returns:
            Filtered, ordered list of AgentEvent records.
        """
        all_events = await self.replay()
        return [e for e in all_events if e.sequence_number >= sequence_number]

    async def _load_from_store(self, start: int) -> list[AgentEvent]:
        store = self._state_store
        assert store is not None
        events: list[AgentEvent] = []
        seq = start
        while True:
            entity_id = f"{self._session_id}:{seq}"
            record, _ = await store.get("events", entity_id, AgentEvent)
            if record is None:
                break
            events.append(record)
            seq += 1
        return events

    async def _count_from_store(self) -> int:
        """Return how many events exist in the store without loading them."""
        store = self._state_store
        assert store is not None
        seq = 0
        while True:
            entity_id = f"{self._session_id}:{seq}"
            record, _ = await store.get("events", entity_id, AgentEvent)
            if record is None:
                break
            seq += 1
        return seq

    @classmethod
    async def open(
        cls,
        *,
        agent_id: str,
        session_id: str,
        state_store: Any | None = None,
    ) -> EventLog:
        """Load or create an EventLog, initializing sequence counter from the store.

        Use this instead of __init__ when the session may already have events
        (e.g., resume flow or CLI replay).

        Args:
            agent_id: Agent identifier.
            session_id: Session to open.
            state_store: Optional Dapr state store.

        Returns:
            EventLog with _next_seq set to the next available sequence number.
        """
        log = cls(agent_id=agent_id, session_id=session_id, state_store=state_store)
        if state_store is not None:
            # Count existing events without loading them into memory — they are
            # already durable in the store and fetched on demand via replay().
            log._next_seq = await log._count_from_store()
        return log

    def event_count(self) -> int:
        """Return the number of events appended in this instance's lifetime."""
        return self._next_seq
