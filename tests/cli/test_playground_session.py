"""Tests for PlaygroundSession and PlaygroundTurn models."""

from __future__ import annotations

from pathlib import Path

import pytest

from nexus.cli.playground.session import PlaygroundSession, PlaygroundTurn
from nexus.core.types import TokenUsage


def _usage(cost: float = 0.01, tokens: int = 100) -> TokenUsage:
    return TokenUsage(
        input_tokens=tokens // 2,
        output_tokens=tokens // 2,
        total_tokens=tokens,
        cost_usd=cost,
        model="test-model",
    )


def _turn(
    cost: float = 0.01, tokens: int = 100, *, user: str = "Hello", assistant: str = "World"
) -> PlaygroundTurn:
    return PlaygroundTurn(
        user_input=user,
        assistant_output=assistant,
        model="test-model",
        token_usage=_usage(cost, tokens),
    )


class TestTotalCost:
    def test_session_total_cost_sums_turns(self) -> None:
        session = PlaygroundSession(
            model="test-model",
            turns=[_turn(0.01), _turn(0.02)],
        )
        assert abs(session.total_cost_usd() - 0.03) < 1e-9

    def test_session_total_cost_zero_on_empty(self) -> None:
        session = PlaygroundSession(model="test-model")
        assert session.total_cost_usd() == 0.0

    def test_session_total_cost_ignores_turns_without_usage(self) -> None:
        t_no_usage = PlaygroundTurn(user_input="q", assistant_output="a", model="m")
        session = PlaygroundSession(model="m", turns=[_turn(0.05), t_no_usage])
        assert abs(session.total_cost_usd() - 0.05) < 1e-9

    def test_session_total_tokens_sums(self) -> None:
        session = PlaygroundSession(
            model="test-model",
            turns=[_turn(tokens=100), _turn(tokens=200)],
        )
        assert session.total_tokens() == 300


class TestToEvalCase:
    def test_session_to_eval_case_last_turn(self) -> None:
        t1 = _turn(user="Q1", assistant="A1")
        t2 = PlaygroundTurn(user_input="Q2", assistant_output="A2", model="m")
        session = PlaygroundSession(model="test-model", turns=[t1, t2])
        case = session.to_eval_case()
        assert case.input == "Q2"
        assert case.metadata["expected_output"] == "A2"

    def test_session_to_eval_case_specific_index(self) -> None:
        t1 = PlaygroundTurn(user_input="Q1", assistant_output="A1", model="m")
        t2 = _turn()
        session = PlaygroundSession(model="test-model", turns=[t1, t2])
        case = session.to_eval_case(turn_index=0)
        assert case.input == "Q1"
        assert case.metadata["expected_output"] == "A1"

    def test_session_to_eval_case_empty_raises(self) -> None:
        session = PlaygroundSession(model="test-model")
        with pytest.raises(ValueError, match="no turns"):
            session.to_eval_case()

    def test_session_to_eval_case_has_name(self) -> None:
        session = PlaygroundSession(model="m", turns=[_turn()])
        case = session.to_eval_case()
        assert case.name.startswith("playground_")


class TestToMessages:
    def test_session_to_messages_system_present(self) -> None:
        session = PlaygroundSession(
            model="test-model",
            system_prompt="You are helpful.",
            turns=[_turn()],
        )
        msgs = session.to_messages()
        assert msgs[0] == {"role": "system", "content": "You are helpful."}

    def test_session_to_messages_alternating_roles(self) -> None:
        t1 = PlaygroundTurn(user_input="Hi", assistant_output="Hello", model="m")
        t2 = PlaygroundTurn(user_input="Bye", assistant_output="Goodbye", model="m")
        session = PlaygroundSession(model="test-model", turns=[t1, t2])
        msgs = session.to_messages()
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hi"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Hello"
        assert msgs[2]["role"] == "user"
        assert msgs[3]["role"] == "assistant"

    def test_session_to_messages_no_system_when_empty_prompt(self) -> None:
        session = PlaygroundSession(model="m", turns=[_turn()])
        msgs = session.to_messages()
        assert all(m["role"] != "system" for m in msgs)


class TestSaveLoad:
    def test_session_save_and_load(self, tmp_path: Path) -> None:
        session = PlaygroundSession(
            name="test-session",
            model="claude-haiku",
            system_prompt="Test system",
            turns=[_turn()],
        )
        session.save(directory=tmp_path)
        loaded = PlaygroundSession.load("test-session", directory=tmp_path)
        assert loaded.session_id == session.session_id
        assert loaded.name == session.name
        assert loaded.system_prompt == session.system_prompt
        assert len(loaded.turns) == 1

    def test_session_load_by_partial_name(self, tmp_path: Path) -> None:
        session = PlaygroundSession(name="partial-match-session", model="m")
        session.save(directory=tmp_path)
        loaded = PlaygroundSession.load("partial-match", directory=tmp_path)
        assert loaded.session_id == session.session_id

    def test_session_load_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PlaygroundSession.load("nonexistent", directory=tmp_path)

    def test_session_save_returns_path(self, tmp_path: Path) -> None:
        session = PlaygroundSession(name="my-sess", model="m")
        path = session.save(directory=tmp_path)
        assert path.exists()
        assert path.name == "my-sess.json"


class TestFilename:
    def test_session_filename_uses_name(self) -> None:
        session = PlaygroundSession(name="my-session", model="test-model")
        assert session._filename() == "my-session.json"

    def test_session_filename_falls_back_to_id(self) -> None:
        session = PlaygroundSession(model="test-model")
        assert session._filename() == f"{session.session_id[:8]}.json"
