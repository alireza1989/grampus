"""Tests for grampus replay CLI command and EventLog.open()."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from grampus.cli.main import cli
from grampus.observability.events import AgentEvent, EventLog, EventType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    event_type: EventType,
    seq: int,
    payload: dict[str, Any] | None = None,
    *,
    agent_id: str = "test-agent",
    session_id: str = "sess-123",
) -> AgentEvent:
    return AgentEvent(
        event_type=event_type,
        agent_id=agent_id,
        session_id=session_id,
        sequence_number=seq,
        payload=payload or {},
        timestamp=datetime(2025, 1, 1, 14, 3, 21, tzinfo=UTC),
    )


def _make_log_with_events(events: list[AgentEvent]) -> EventLog:
    log = EventLog(agent_id="test-agent", session_id="sess-123")
    log._events = list(events)
    log._next_seq = len(events)
    return log


# ---------------------------------------------------------------------------
# EventLog.open() unit tests
# ---------------------------------------------------------------------------


class TestEventLogOpen:
    async def test_open_empty_store_returns_zero_seq(self) -> None:
        log = await EventLog.open(agent_id="a", session_id="s", state_store=None)
        assert log._next_seq == 0
        assert log._events == []

    async def test_open_no_state_store_returns_zero_seq(self) -> None:
        log = await EventLog.open(agent_id="a", session_id="s")
        assert log._next_seq == 0

    async def test_open_initializes_seq_from_store(self) -> None:
        stored = [
            _make_event(EventType.AGENT_STARTED, 0, session_id="s"),
            _make_event(EventType.LLM_CALLED, 1, session_id="s"),
            _make_event(EventType.AGENT_COMPLETED, 2, session_id="s"),
        ]

        async def fake_get(entity_type: str, entity_id: str, cls: type) -> tuple:
            parts = entity_id.split(":")
            seq = int(parts[-1])
            if seq < len(stored):
                return stored[seq], "etag"
            return None, ""

        store = MagicMock()
        store.get = fake_get
        log = await EventLog.open(agent_id="a", session_id="s", state_store=store)
        assert log._next_seq == 3
        # _events is intentionally empty when a state_store is configured —
        # events are durable in the store and fetched via replay(), not held in RAM.
        assert len(log._events) == 0

    async def test_append_after_open_does_not_overwrite(self) -> None:
        stored = [
            _make_event(EventType.AGENT_STARTED, 0, session_id="s"),
            _make_event(EventType.LLM_CALLED, 1, session_id="s"),
        ]
        saved: list[tuple[str, str, Any]] = []

        async def fake_get(entity_type: str, entity_id: str, cls: type) -> tuple:
            parts = entity_id.split(":")
            seq = int(parts[-1])
            if seq < len(stored):
                return stored[seq], "etag"
            return None, ""

        async def fake_save(entity_type: str, entity_id: str, model: Any, **_: Any) -> None:
            saved.append((entity_type, entity_id, model))

        store = MagicMock()
        store.get = fake_get
        store.save = fake_save

        log = await EventLog.open(agent_id="a", session_id="s", state_store=store)
        new_event = await log.append(EventType.AGENT_COMPLETED)
        assert new_event.sequence_number == 2


class TestEventLogLoadFromStoreProbe:
    async def test_probe_stops_at_first_none(self) -> None:
        events = [
            _make_event(EventType.AGENT_STARTED, 0, session_id="s"),
            _make_event(EventType.LLM_CALLED, 1, session_id="s"),
            _make_event(EventType.TOOL_CALLED, 2, session_id="s"),
        ]

        async def fake_get(entity_type: str, entity_id: str, cls: type) -> tuple:
            parts = entity_id.split(":")
            seq = int(parts[-1])
            if seq < 3:
                return events[seq], "etag"
            return None, ""

        store = MagicMock()
        store.get = fake_get
        log = EventLog(agent_id="a", session_id="s", state_store=store)
        result = await log._load_from_store(0)
        assert len(result) == 3
        assert result[0].sequence_number == 0
        assert result[2].sequence_number == 2


# ---------------------------------------------------------------------------
# Fixtures for CLI tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _patch_replay(monkeypatch: pytest.MonkeyPatch, log: EventLog) -> None:
    """Patch DaprGateway, DaprStateStore, and EventLog.open to return *log*."""

    async def _fake_open(
        cls: type, *, agent_id: str, session_id: str, state_store: Any = None
    ) -> EventLog:
        return log

    # Patch EventLog.open at the class level; monkeypatch will restore it after the test.
    monkeypatch.setattr(EventLog, "open", classmethod(_fake_open))

    # DaprGateway and DaprStateStore are imported inside _replay_async,
    # so patch them at their source module so the lazy import picks up the mock.
    monkeypatch.setattr("grampus.dapr.client.DaprGateway", MagicMock())
    monkeypatch.setattr("grampus.dapr.state.DaprStateStore", MagicMock())


# ---------------------------------------------------------------------------
# CLI tests — grampus replay
# ---------------------------------------------------------------------------


class TestReplayNoEvents:
    def test_no_events_exits_1(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        log = _make_log_with_events([])
        _patch_replay(monkeypatch, log)

        # load_config must not raise — use a fake config
        monkeypatch.setattr(
            "grampus.cli.commands.replay.load_config",
            lambda path: MagicMock(dapr=None, agent=None),
        )

        result = runner.invoke(cli, ["replay", "sess-empty", "--config", "grampus.yaml"])
        assert result.exit_code == 1
        assert "No events found" in result.output


class TestReplayRendersAgentStarted:
    def test_renders_agent_started(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events = [
            _make_event(
                EventType.AGENT_STARTED,
                0,
                {"input": "hello world", "model": "claude-sonnet-4-6"},
            )
        ]
        log = _make_log_with_events(events)
        _patch_replay(monkeypatch, log)
        monkeypatch.setattr(
            "grampus.cli.commands.replay.load_config",
            lambda path: MagicMock(dapr=None, agent=None),
        )

        result = runner.invoke(cli, ["replay", "sess-123", "--config", "grampus.yaml"])
        assert result.exit_code == 0
        assert "AGENT STARTED" in result.output
        assert "hello world" in result.output
        assert "claude-sonnet-4-6" in result.output


class TestReplayRendersToolCall:
    def test_renders_tool_call_success(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events = [
            _make_event(
                EventType.TOOL_CALLED, 0, {"tool": "web_search", "args": '{"query": "AI"}'}
            ),
            _make_event(
                EventType.TOOL_RESULT,
                1,
                {"tool": "web_search", "ok": True, "output": "Top results..."},
            ),
        ]
        log = _make_log_with_events(events)
        _patch_replay(monkeypatch, log)
        monkeypatch.setattr(
            "grampus.cli.commands.replay.load_config",
            lambda path: MagicMock(dapr=None, agent=None),
        )

        result = runner.invoke(cli, ["replay", "sess-123", "--config", "grampus.yaml"])
        assert result.exit_code == 0
        assert "web_search" in result.output
        assert "✓" in result.output

    def test_renders_tool_call_failure(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events = [
            _make_event(
                EventType.TOOL_RESULT, 0, {"tool": "bad_tool", "ok": False, "output": "error msg"}
            ),
        ]
        log = _make_log_with_events(events)
        _patch_replay(monkeypatch, log)
        monkeypatch.setattr(
            "grampus.cli.commands.replay.load_config",
            lambda path: MagicMock(dapr=None, agent=None),
        )

        result = runner.invoke(cli, ["replay", "sess-123", "--config", "grampus.yaml"])
        assert result.exit_code == 0
        assert "✗" in result.output


class TestReplayRendersHumanPause:
    def test_renders_paused(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [
            _make_event(
                EventType.HUMAN_INPUT_REQUESTED,
                0,
                {"question": "What is your preferred format?"},
            )
        ]
        log = _make_log_with_events(events)
        _patch_replay(monkeypatch, log)
        monkeypatch.setattr(
            "grampus.cli.commands.replay.load_config",
            lambda path: MagicMock(dapr=None, agent=None),
        )

        result = runner.invoke(cli, ["replay", "sess-123", "--config", "grampus.yaml"])
        assert result.exit_code == 0
        assert "PAUSED" in result.output
        assert "What is your preferred format?" in result.output


class TestReplayJsonFlag:
    def test_json_output_is_valid(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [
            _make_event(EventType.AGENT_STARTED, 0, {"input": "hi", "model": "m"}),
            _make_event(
                EventType.AGENT_COMPLETED, 1, {"output": "bye", "steps": 1, "cost_usd": 0.001}
            ),
        ]
        log = _make_log_with_events(events)
        _patch_replay(monkeypatch, log)
        monkeypatch.setattr(
            "grampus.cli.commands.replay.load_config",
            lambda path: MagicMock(dapr=None, agent=None),
        )

        result = runner.invoke(cli, ["replay", "sess-123", "--config", "grampus.yaml", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["event_type"] == EventType.AGENT_STARTED
        assert "sequence_number" in parsed[0]
        assert "agent_id" in parsed[0]
        assert "session_id" in parsed[0]

    def test_json_no_events_exits_1(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = _make_log_with_events([])
        _patch_replay(monkeypatch, log)
        monkeypatch.setattr(
            "grampus.cli.commands.replay.load_config",
            lambda path: MagicMock(dapr=None, agent=None),
        )

        result = runner.invoke(cli, ["replay", "sess-123", "--config", "grampus.yaml", "--json"])
        assert result.exit_code == 1


class TestReplayFromStep:
    def test_from_step_filters_earlier_events(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        all_events = [
            _make_event(EventType.AGENT_STARTED, 0, {"input": "early", "model": "m"}),
            _make_event(
                EventType.LLM_CALLED, 1, {"step": 1, "input_tokens": 10, "output_tokens": 5}
            ),
            _make_event(
                EventType.LLM_CALLED, 2, {"step": 2, "input_tokens": 20, "output_tokens": 8}
            ),
            _make_event(EventType.TOOL_CALLED, 3, {"tool": "search", "args": "{}"}),
            _make_event(
                EventType.AGENT_COMPLETED, 4, {"output": "done", "steps": 2, "cost_usd": 0.002}
            ),
        ]
        log = _make_log_with_events(all_events)
        _patch_replay(monkeypatch, log)
        monkeypatch.setattr(
            "grampus.cli.commands.replay.load_config",
            lambda path: MagicMock(dapr=None, agent=None),
        )

        result = runner.invoke(
            cli, ["replay", "sess-123", "--config", "grampus.yaml", "--from-step", "3"]
        )
        assert result.exit_code == 0
        # Events at seq 3,4 are shown; seq 0,1,2 are NOT
        assert "search" in result.output  # seq 3 TOOL_CALLED
        assert "AGENT COMPLETED" in result.output  # seq 4
        assert "early" not in result.output  # seq 0 filtered out


class TestReplayConfigError:
    def test_missing_config_exits_1(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["replay", "sess-123", "--config", "/nonexistent/grampus.yaml"])
        assert result.exit_code == 1
        assert "Error" in result.output
