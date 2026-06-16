"""Tests for run_compare() concurrent multi-model calls."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from grampus.cli.playground.repl import run_compare
from grampus.core.errors import ModelError
from grampus.core.models.base import ModelResponse
from grampus.core.types import TokenUsage


def _usage(model: str = "test", cost: float = 0.001) -> TokenUsage:
    return TokenUsage(
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        cost_usd=cost,
        model=model,
    )


def _response(content: str = "answer", model: str = "test") -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=[],
        token_usage=_usage(model),
        model=model,
        stop_reason="end_turn",
    )


def _mock_client(content: str = "answer", model: str = "test") -> MagicMock:
    client = MagicMock()
    client.complete = AsyncMock(return_value=_response(content, model))
    return client


def _config() -> MagicMock:
    cfg = MagicMock()
    cfg.model.anthropic_api_key = SecretStr("key")
    cfg.model.openai_api_key = SecretStr("key")
    cfg.model.gemini_api_key = SecretStr("key")
    cfg.model.ollama_host = "http://localhost:11434"
    return cfg


class TestRunCompare:
    async def test_compare_returns_correct_outputs(self) -> None:
        clients = {
            "llama3.2": _mock_client("response-A", "llama3.2"),
            "mistral": _mock_client("response-B", "mistral"),
        }

        def _factory(model: str, _cfg: object) -> MagicMock:
            return clients[model]

        with patch("grampus.cli.playground.repl.make_client", side_effect=_factory):
            results = await run_compare(
                user_message="Hello",
                models=["llama3.2", "mistral"],
                system_prompt="",
                config=_config(),
            )

        assert len(results) == 2
        by_model = {r.model: r for r in results}
        assert by_model["llama3.2"].output == "response-A"
        assert by_model["mistral"].output == "response-B"

    async def test_compare_calls_all_models_concurrently(self) -> None:
        models = ["llama3.2", "mistral", "codellama"]
        called: list[str] = []

        def _factory(model: str, _cfg: object) -> MagicMock:
            client = MagicMock()

            async def _complete(**kwargs: object) -> ModelResponse:
                called.append(model)
                return _response("ok", model)

            client.complete = _complete
            return client

        with patch("grampus.cli.playground.repl.make_client", side_effect=_factory):
            results = await run_compare(
                user_message="hi",
                models=models,
                system_prompt="",
                config=_config(),
            )

        assert set(called) == set(models)
        assert len(results) == 3

    async def test_compare_handles_one_model_error(self) -> None:
        def _factory(model: str, _cfg: object) -> MagicMock:
            client = MagicMock()
            if model == "bad-model":
                client.complete = AsyncMock(side_effect=RuntimeError("boom"))
            else:
                client.complete = AsyncMock(return_value=_response("ok", model))
            return client

        with patch("grampus.cli.playground.repl.make_client", side_effect=_factory):
            results = await run_compare(
                user_message="test",
                models=["llama3.2", "bad-model"],
                system_prompt="",
                config=_config(),
            )

        by_model = {r.model: r for r in results}
        assert by_model["llama3.2"].error is None
        assert by_model["llama3.2"].output == "ok"
        assert by_model["bad-model"].error == "boom"

    async def test_compare_result_model_error_field_set(self) -> None:
        def _factory(model: str, _cfg: object) -> MagicMock:
            client = MagicMock()
            client.complete = AsyncMock(side_effect=ModelError("api fail", code="MODEL_API_ERROR"))
            return client

        with patch("grampus.cli.playground.repl.make_client", side_effect=_factory):
            results = await run_compare(
                user_message="x",
                models=["llama3.2"],
                system_prompt="",
                config=_config(),
            )

        assert len(results) == 1
        assert results[0].error is not None
        assert results[0].output == ""
        assert results[0].token_usage is None

    async def test_compare_empty_model_list_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            await run_compare(
                user_message="x",
                models=[],
                system_prompt="",
                config=_config(),
            )

    async def test_compare_uses_provided_messages(self) -> None:
        from grampus.core.types import Message, Role

        received_messages: list[list[Message]] = []

        def _factory(model: str, _cfg: object) -> MagicMock:
            client = MagicMock()

            async def _complete(messages: list[Message], **kwargs: object) -> ModelResponse:
                received_messages.append(list(messages))
                return _response("ok", model)

            client.complete = _complete
            return client

        msgs = [
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="user msg"),
        ]
        with patch("grampus.cli.playground.repl.make_client", side_effect=_factory):
            await run_compare(
                user_message="ignored",
                models=["llama3.2"],
                system_prompt="",
                config=_config(),
                _messages=msgs,
            )

        assert len(received_messages[0]) == 2
        assert received_messages[0][0].role == Role.SYSTEM
