"""Tests for playground Click commands via CliRunner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner
from pydantic import SecretStr

from grampus.cli.main import cli
from grampus.cli.playground.session import PlaygroundSession, PlaygroundTurn
from grampus.core.models.base import ModelResponse
from grampus.core.types import StreamChunk, TokenUsage


def _usage(model: str = "test") -> TokenUsage:
    return TokenUsage(
        input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.001, model=model
    )


def _response(content: str = "Hello!", model: str = "test") -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=[],
        token_usage=_usage(model),
        model=model,
        stop_reason="end_turn",
    )


async def _mock_stream(*args, **kwargs):
    yield StreamChunk(delta="Hello!", model="test")
    yield StreamChunk(delta="", is_final=True, token_usage=_usage(), model="test")


def _mock_client(stream_fn=None, content: str = "Hello!") -> MagicMock:
    client = MagicMock()
    client.stream = stream_fn or _mock_stream
    client.complete = AsyncMock(return_value=_response(content))
    return client


def _config() -> MagicMock:
    cfg = MagicMock()
    cfg.model.anthropic_api_key = SecretStr("key")
    cfg.model.openai_api_key = SecretStr("key")
    cfg.model.gemini_api_key = SecretStr("key")
    cfg.model.ollama_host = "http://localhost:11434"
    return cfg


class TestPlaygroundRun:
    def test_playground_run_prints_response(self) -> None:
        runner = CliRunner()
        client = _mock_client()
        with (
            patch("grampus.cli.commands.playground._load_config", return_value=_config()),
            patch("grampus.cli.commands.playground.make_client", return_value=client),
        ):
            result = runner.invoke(
                cli, ["playground", "run", "What is 2+2?", "--model", "llama3.2"]
            )
        assert result.exit_code == 0, result.output
        assert "Hello!" in result.output

    def test_playground_run_no_stream_still_works(self) -> None:
        runner = CliRunner()
        client = _mock_client()
        with (
            patch("grampus.cli.commands.playground._load_config", return_value=_config()),
            patch("grampus.cli.commands.playground.make_client", return_value=client),
        ):
            result = runner.invoke(
                cli,
                ["playground", "run", "hi", "--model", "llama3.2", "--no-stream"],
            )
        assert result.exit_code == 0, result.output
        assert "Hello!" in result.output

    def test_playground_run_with_system_prompt(self) -> None:
        runner = CliRunner()
        client = _mock_client()
        with (
            patch("grampus.cli.commands.playground._load_config", return_value=_config()),
            patch("grampus.cli.commands.playground.make_client", return_value=client),
        ):
            result = runner.invoke(
                cli,
                ["playground", "run", "hello", "-s", "Be brief.", "--model", "llama3.2"],
            )
        assert result.exit_code == 0, result.output


class TestPlaygroundCompare:
    def test_playground_compare_multiple_models(self) -> None:
        runner = CliRunner()

        def _factory(model, _cfg):
            return _mock_client(content=f"Answer from {model}")

        with (
            patch("grampus.cli.commands.playground._load_config", return_value=_config()),
            patch("grampus.cli.playground.repl.make_client", side_effect=_factory),
        ):
            result = runner.invoke(
                cli,
                ["playground", "compare", "hello", "--models", "llama3.2,mistral"],
            )

        assert result.exit_code == 0, result.output
        assert "llama3.2" in result.output
        assert "mistral" in result.output

    def test_playground_compare_shows_cost_table(self) -> None:
        runner = CliRunner()

        def _factory(model, _cfg):
            return _mock_client()

        with (
            patch("grampus.cli.commands.playground._load_config", return_value=_config()),
            patch("grampus.cli.playground.repl.make_client", side_effect=_factory),
        ):
            result = runner.invoke(
                cli,
                ["playground", "compare", "q", "--models", "llama3.2,mistral"],
            )

        assert result.exit_code == 0, result.output
        assert "cost" in result.output.lower() or "$" in result.output


class TestPlaygroundSessions:
    def test_playground_sessions_empty(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with patch("grampus.cli.commands.playground._SESSIONS_DIR", tmp_path):
            result = runner.invoke(cli, ["playground", "sessions"])
        assert result.exit_code == 0, result.output
        assert "no" in result.output.lower() or result.output.strip() == ""

    def test_playground_sessions_lists_saved(self, tmp_path: Path) -> None:
        s = PlaygroundSession(name="my-test-sess", model="m")
        s.save(directory=tmp_path)
        runner = CliRunner()
        with patch("grampus.cli.commands.playground._SESSIONS_DIR", tmp_path):
            result = runner.invoke(cli, ["playground", "sessions"])
        assert result.exit_code == 0, result.output
        assert "my-test-sess" in result.output


class TestPlaygroundShow:
    def _save_session(self, tmp_path: Path) -> PlaygroundSession:
        turn = PlaygroundTurn(
            user_input="What is Python?",
            assistant_output="Python is a language.",
            model="test",
        )
        session = PlaygroundSession(
            name="show-test",
            model="test",
            system_prompt="Be helpful.",
            turns=[turn],
        )
        session.save(directory=tmp_path)
        return session

    def test_playground_show_transcript(self, tmp_path: Path) -> None:
        session = self._save_session(tmp_path)
        runner = CliRunner()
        with patch("grampus.cli.commands.playground.PlaygroundSession") as mock_cls:
            mock_cls.load.return_value = session
            result = runner.invoke(cli, ["playground", "show", "show-test"])
        assert result.exit_code == 0, result.output
        assert "What is Python?" in result.output
        assert "Python is a language." in result.output

    def test_playground_show_json(self, tmp_path: Path) -> None:
        session = self._save_session(tmp_path)
        runner = CliRunner()
        with patch("grampus.cli.commands.playground.PlaygroundSession") as mock_cls:
            mock_cls.load.return_value = session
            result = runner.invoke(cli, ["playground", "show", "show-test", "--format", "json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["name"] == "show-test"
        assert len(data["turns"]) == 1

    def test_playground_show_not_found_exits_nonzero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with patch("grampus.cli.commands.playground.PlaygroundSession") as mock_cls:
            mock_cls.load.side_effect = FileNotFoundError("not found")
            result = runner.invoke(cli, ["playground", "show", "ghost"])
        assert result.exit_code != 0
