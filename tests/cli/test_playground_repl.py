"""Tests for REPL _send_message, _handle_command, and run_repl."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import SecretStr

from nexus.cli.playground.renderer import Renderer
from nexus.cli.playground.repl import (
    _handle_command,
    _ReplState,
    _send_message,
    run_repl,
)
from nexus.cli.playground.session import PlaygroundSession, PlaygroundTurn
from nexus.core.types import Message, Role, StreamChunk, TokenUsage
from nexus.evaluation.prompt_versions import PromptVersionManager


def _usage() -> TokenUsage:
    return TokenUsage(
        input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.001, model="test"
    )


async def _mock_stream_gen(*args: object, **kwargs: object):  # type: ignore[return]
    yield StreamChunk(delta="Hello", model="test")
    yield StreamChunk(delta=" world", model="test")
    yield StreamChunk(
        delta="",
        is_final=True,
        token_usage=_usage(),
        model="test",
    )


def _mock_client(stream_fn=None) -> MagicMock:
    client = MagicMock()
    client.stream = stream_fn if stream_fn is not None else _mock_stream_gen
    return client


def _config() -> MagicMock:
    cfg = MagicMock()
    cfg.model.anthropic_api_key = SecretStr("key")
    cfg.model.ollama_host = "http://localhost:11434"
    return cfg


def _state(
    sessions_dir: Path | None = None,
    client: MagicMock | None = None,
    turns: list[PlaygroundTurn] | None = None,
    system_prompt: str = "",
) -> _ReplState:
    sess = PlaygroundSession(model="test-model", system_prompt=system_prompt)
    if turns:
        sess.turns.extend(turns)
    history: list[Message] = []
    if system_prompt:
        history.append(Message(role=Role.SYSTEM, content=system_prompt))
    return _ReplState(
        session=sess,
        active_model="test-model",
        client=client or _mock_client(),
        renderer=Renderer(use_color=False),
        history=history,
        config=_config(),
        version_manager=PromptVersionManager(agent_id="test"),
        sessions_dir=sessions_dir or Path("/tmp/nexus-test-sessions"),
    )


class TestSendMessage:
    async def test_repl_send_message_appends_turn(self) -> None:
        state = _state()
        await _send_message("Hello", state)
        assert len(state.session.turns) == 1
        assert state.session.turns[0].user_input == "Hello"
        assert state.session.turns[0].assistant_output == "Hello world"

    async def test_repl_send_message_streams_output(self, capsys) -> None:
        state = _state()
        await _send_message("Hi", state)
        captured = capsys.readouterr()
        assert "Hello" in captured.out
        assert "world" in captured.out

    async def test_repl_multi_turn_history_grows(self) -> None:
        state = _state()
        await _send_message("First", state)
        await _send_message("Second", state)
        assert len(state.session.turns) == 2
        # history: [user, assistant, user, assistant]
        user_msgs = [m for m in state.history if m.role == Role.USER]
        assert len(user_msgs) == 2

    async def test_repl_send_message_records_token_usage(self) -> None:
        state = _state()
        await _send_message("q", state)
        assert state.session.turns[0].token_usage is not None
        assert state.session.turns[0].token_usage.total_tokens == 30

    async def test_repl_send_message_on_stream_error_no_turn_added(self) -> None:
        async def _error_stream(*args, **kwargs):
            raise RuntimeError("network error")
            yield  # make it a generator

        state = _state(client=_mock_client(_error_stream))
        await _send_message("oops", state)
        assert len(state.session.turns) == 0


class TestCommandModel:
    async def test_repl_command_model_switch(self) -> None:
        state = _state()
        new_client = _mock_client()
        with patch("nexus.cli.playground.repl.make_client", return_value=new_client):
            result = await _handle_command("/model gpt-4o", state)
        assert result is True
        assert state.active_model == "gpt-4o"
        assert state.client is new_client


class TestCommandSystem:
    async def test_repl_command_system_sets_prompt(self) -> None:
        state = _state()
        result = await _handle_command("/system You are a pirate.", state)
        assert result is True
        assert state.session.system_prompt == "You are a pirate."
        sys_msgs = [m for m in state.history if m.role == Role.SYSTEM]
        assert len(sys_msgs) == 1
        assert sys_msgs[0].content == "You are a pirate."

    async def test_repl_command_system_file(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("You are a helpful assistant.")
        state = _state()
        result = await _handle_command(f"/system file:{prompt_file}", state)
        assert result is True
        assert state.session.system_prompt == "You are a helpful assistant."

    async def test_repl_command_system_file_not_found(self, capsys) -> None:
        state = _state()
        result = await _handle_command("/system file:/nonexistent/path.txt", state)
        assert result is True
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "File not found" in captured.out


class TestCommandReset:
    async def test_repl_command_reset_clears_history(self) -> None:
        state = _state()
        await _send_message("Hello", state)
        assert len(state.history) >= 2
        await _handle_command("/reset", state)
        user_msgs = [m for m in state.history if m.role == Role.USER]
        assert len(user_msgs) == 0
        assert len(state.session.turns) == 0

    async def test_repl_command_reset_preserves_system_prompt(self) -> None:
        state = _state(system_prompt="Keep this.")
        await _send_message("Hello", state)
        await _handle_command("/reset", state)
        sys_msgs = [m for m in state.history if m.role == Role.SYSTEM]
        assert len(sys_msgs) == 1
        assert sys_msgs[0].content == "Keep this."


class TestCommandCost:
    async def test_repl_command_cost_prints_summary(self, capsys) -> None:
        turn = PlaygroundTurn(
            user_input="q",
            assistant_output="a",
            model="test",
            token_usage=_usage(),
        )
        state = _state(turns=[turn])
        await _handle_command("/cost", state)
        captured = capsys.readouterr()
        assert "turn" in captured.out.lower()


class TestCommandSave:
    async def test_repl_command_save_writes_file(self, tmp_path: Path) -> None:
        state = _state(sessions_dir=tmp_path)
        result = await _handle_command("/save my-session", state)
        assert result is True
        assert state.session.name == "my-session"
        assert (tmp_path / "my-session.json").exists()

    async def test_repl_command_save_prints_path(self, tmp_path: Path, capsys) -> None:
        state = _state(sessions_dir=tmp_path)
        await _handle_command("/save abc", state)
        captured = capsys.readouterr()
        assert "abc" in captured.out


class TestCommandLoad:
    async def test_repl_command_load_restores_session(self, tmp_path: Path) -> None:
        saved = PlaygroundSession(
            name="saved-sess",
            model="gpt-4o",
            system_prompt="Sys",
            turns=[PlaygroundTurn(user_input="U1", assistant_output="A1", model="gpt-4o")],
        )
        saved.save(directory=tmp_path)

        state = _state(sessions_dir=tmp_path)
        result = await _handle_command("/load saved-sess", state)
        assert result is True
        assert len(state.session.turns) == 1
        assert state.session.system_prompt == "Sys"
        user_msgs = [m for m in state.history if m.role == Role.USER]
        assert len(user_msgs) == 1

    async def test_repl_command_load_not_found_prints_error(self, tmp_path, capsys) -> None:
        state = _state(sessions_dir=tmp_path)
        result = await _handle_command("/load no-such-session", state)
        assert result is True
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "no-such" in captured.out


class TestCommandSessions:
    async def test_repl_command_sessions_lists_files(self, tmp_path: Path, capsys) -> None:
        s1 = PlaygroundSession(name="sess-alpha", model="m")
        s2 = PlaygroundSession(name="sess-beta", model="m")
        s1.save(directory=tmp_path)
        s2.save(directory=tmp_path)

        state = _state(sessions_dir=tmp_path)
        await _handle_command("/sessions", state)
        captured = capsys.readouterr()
        assert "sess-alpha" in captured.out
        assert "sess-beta" in captured.out

    async def test_repl_command_sessions_empty(self, tmp_path: Path, capsys) -> None:
        state = _state(sessions_dir=tmp_path)
        await _handle_command("/sessions", state)
        captured = capsys.readouterr()
        assert "no" in captured.out.lower() or captured.out.strip() == ""


class TestCommandExport:
    async def test_repl_command_export_writes_eval_case(self, tmp_path: Path) -> None:
        turn = PlaygroundTurn(user_input="Q", assistant_output="A", model="test")
        state = _state(turns=[turn])
        out_path = tmp_path / "case.json"
        result = await _handle_command(f"/export {out_path}", state)
        assert result is True
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["input"] == "Q"

    async def test_repl_command_export_to_stdout(self, capsys) -> None:
        turn = PlaygroundTurn(user_input="Q2", assistant_output="A2", model="test")
        state = _state(turns=[turn])
        await _handle_command("/export", state)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["input"] == "Q2"

    async def test_repl_command_export_no_turns_error(self, capsys) -> None:
        state = _state()
        await _handle_command("/export", state)
        captured = capsys.readouterr()
        assert "no turns" in captured.out.lower()


class TestCommandExit:
    async def test_repl_command_exit_returns_false(self) -> None:
        state = _state()
        result = await _handle_command("/exit", state)
        assert result is False

    async def test_repl_command_quit_returns_false(self) -> None:
        state = _state()
        result = await _handle_command("/quit", state)
        assert result is False


class TestCommandUnknown:
    async def test_repl_command_unknown_prints_error(self, capsys) -> None:
        state = _state()
        result = await _handle_command("/bogus-command", state)
        assert result is True
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out
        assert "/bogus-command" in captured.out

    async def test_repl_command_help_returns_true(self, capsys) -> None:
        state = _state()
        result = await _handle_command("/help", state)
        assert result is True
        captured = capsys.readouterr()
        assert "/model" in captured.out


class TestRunRepl:
    async def test_repl_eof_exits_gracefully(self, tmp_path: Path) -> None:
        """REPL should exit cleanly on EOFError from input()."""
        cfg = _config()
        client = _mock_client()

        call_count = 0

        def _raising_input(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            raise EOFError

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **kw: _raising_input(*a)):
            await run_repl(
                cfg,
                model="test-model",
                system_prompt="",
                sessions_dir=tmp_path,
                _client=client,
            )

        assert call_count >= 1
