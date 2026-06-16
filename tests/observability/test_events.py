"""Tests for EventLog append-only agent event log."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from grampus.observability.events import AgentEvent, EventLog, EventType


class TestEventLogAppend:
    async def test_append_returns_agent_event(self) -> None:
        log = EventLog(agent_id="a1", session_id="s1")
        event = await log.append(EventType.AGENT_STARTED)
        assert isinstance(event, AgentEvent)

    async def test_append_increments_sequence_number(self) -> None:
        log = EventLog(agent_id="a1", session_id="s1")
        e0 = await log.append(EventType.AGENT_STARTED)
        e1 = await log.append(EventType.LLM_CALLED)
        e2 = await log.append(EventType.TOOL_CALLED)
        assert e0.sequence_number == 0
        assert e1.sequence_number == 1
        assert e2.sequence_number == 2

    async def test_append_sets_correct_event_type(self) -> None:
        log = EventLog(agent_id="a1", session_id="s1")
        event = await log.append(EventType.SAFETY_VIOLATION, {"reason": "injection"})
        assert event.event_type == EventType.SAFETY_VIOLATION

    async def test_append_sets_agent_and_session_id(self) -> None:
        log = EventLog(agent_id="myagent", session_id="mysession")
        event = await log.append(EventType.AGENT_STARTED)
        assert event.agent_id == "myagent"
        assert event.session_id == "mysession"

    async def test_append_stores_payload(self) -> None:
        log = EventLog(agent_id="a1", session_id="s1")
        event = await log.append(EventType.LLM_CALLED, {"model": "claude", "tokens": 100})
        assert event.payload["model"] == "claude"
        assert event.payload["tokens"] == 100

    async def test_event_count_increments(self) -> None:
        log = EventLog(agent_id="a1", session_id="s1")
        assert log.event_count() == 0
        await log.append(EventType.AGENT_STARTED)
        await log.append(EventType.LLM_CALLED)
        assert log.event_count() == 2


class TestEventLogReplay:
    async def test_replay_empty_initially(self) -> None:
        log = EventLog(agent_id="a1", session_id="s1")
        events = await log.replay()
        assert events == []

    async def test_replay_returns_all_events_in_order(self) -> None:
        log = EventLog(agent_id="a1", session_id="s1")
        await log.append(EventType.AGENT_STARTED)
        await log.append(EventType.LLM_CALLED)
        await log.append(EventType.AGENT_COMPLETED)
        events = await log.replay()
        assert len(events) == 3
        assert [e.sequence_number for e in events] == [0, 1, 2]

    async def test_replay_since_filters_by_sequence(self) -> None:
        log = EventLog(agent_id="a1", session_id="s1")
        for et in [
            EventType.AGENT_STARTED,
            EventType.LLM_CALLED,
            EventType.TOOL_CALLED,
            EventType.AGENT_COMPLETED,
        ]:
            await log.append(et)
        events = await log.replay_since(2)
        assert len(events) == 2
        assert events[0].sequence_number == 2
        assert events[1].sequence_number == 3

    async def test_replay_with_state_store_persists_events(self) -> None:
        store = MagicMock()
        store.save = AsyncMock()
        log = EventLog(agent_id="a1", session_id="s1", state_store=store)
        await log.append(EventType.AGENT_STARTED)
        store.save.assert_called_once()

    async def test_replay_with_state_store_loads_persisted_events(self) -> None:
        stored_event = AgentEvent(
            event_type=EventType.AGENT_STARTED,
            agent_id="a1",
            session_id="s1",
            sequence_number=0,
        )
        store = MagicMock()
        store.save = AsyncMock()

        async def fake_get(entity_type: str, entity_id: str, cls: type) -> tuple:
            if entity_id == "s1:0":
                return stored_event, "etag"
            return None, ""

        store.get = fake_get
        log = EventLog(agent_id="a1", session_id="s1", state_store=store)
        # Manually set sequence counter so replay stops at 1
        log._next_seq = 1
        events = await log.replay()
        assert len(events) == 1
        assert events[0].event_type == EventType.AGENT_STARTED


class TestAgentEventImmutability:
    def test_agent_event_is_frozen(self) -> None:
        event = AgentEvent(
            event_type=EventType.LLM_CALLED,
            agent_id="a",
            session_id="s",
        )
        with pytest.raises((TypeError, ValidationError)):
            event.agent_id = "modified"  # type: ignore[misc]

    async def test_sequence_numbers_are_monotonic(self) -> None:
        log = EventLog(agent_id="a1", session_id="s1")
        events = []
        for et in [EventType.AGENT_STARTED, EventType.LLM_CALLED, EventType.TOOL_CALLED]:
            events.append(await log.append(et))
        seqs = [e.sequence_number for e in events]
        assert seqs == sorted(seqs)
        assert len(seqs) == len(set(seqs))
