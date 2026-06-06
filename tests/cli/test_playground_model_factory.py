"""Tests for make_client() routing and error handling."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from nexus.cli.playground.model_factory import make_client
from nexus.core.errors import ModelError
from nexus.core.models.anthropic import AnthropicClient
from nexus.core.models.gemini import GeminiClient
from nexus.core.models.ollama import OllamaClient
from nexus.core.models.openai import OpenAIClient

# SDK stubs for providers not installed in the test environment
_FAKE_OLLAMA = MagicMock()
_FAKE_OLLAMA.AsyncClient.return_value = MagicMock()

_FAKE_GOOGLE = MagicMock()
_FAKE_GENAI = MagicMock()
_FAKE_GENAI.Client.return_value = MagicMock()

_SDK_PATCHES: dict[str, MagicMock] = {
    "ollama": _FAKE_OLLAMA,
    "google": _FAKE_GOOGLE,
    "google.genai": _FAKE_GENAI,
}


def _config(
    anthropic_key: str | None = "ant-key",
    openai_key: str | None = "oai-key",
    gemini_key: str | None = "gem-key",
    ollama_host: str = "http://localhost:11434",
) -> MagicMock:
    """Build a minimal config mock with the given key values."""
    cfg = MagicMock()
    cfg.model.anthropic_api_key = SecretStr(anthropic_key) if anthropic_key else None
    cfg.model.openai_api_key = SecretStr(openai_key) if openai_key else None
    cfg.model.gemini_api_key = SecretStr(gemini_key) if gemini_key else None
    cfg.model.ollama_host = ollama_host
    return cfg


class TestRouting:
    def test_make_client_claude_returns_anthropic(self) -> None:
        client = make_client("claude-haiku-4-5", _config())
        assert isinstance(client, AnthropicClient)

    def test_make_client_claude_3_returns_anthropic(self) -> None:
        client = make_client("claude-3-5-sonnet-20241022", _config())
        assert isinstance(client, AnthropicClient)

    def test_make_client_gpt_returns_openai(self) -> None:
        client = make_client("gpt-4o", _config())
        assert isinstance(client, OpenAIClient)

    def test_make_client_gpt_mini_returns_openai(self) -> None:
        client = make_client("gpt-4o-mini", _config())
        assert isinstance(client, OpenAIClient)

    def test_make_client_o1_returns_openai(self) -> None:
        client = make_client("o1", _config())
        assert isinstance(client, OpenAIClient)

    def test_make_client_o3_returns_openai(self) -> None:
        client = make_client("o3-mini", _config())
        assert isinstance(client, OpenAIClient)

    def test_make_client_gemini_returns_gemini(self) -> None:
        with patch.dict(sys.modules, _SDK_PATCHES):
            client = make_client("gemini-2.5-flash", _config())
        assert isinstance(client, GeminiClient)

    def test_make_client_gemini_pro_returns_gemini(self) -> None:
        with patch.dict(sys.modules, _SDK_PATCHES):
            client = make_client("gemini-1.5-pro", _config())
        assert isinstance(client, GeminiClient)

    def test_make_client_llama_returns_ollama(self) -> None:
        with patch.dict(sys.modules, _SDK_PATCHES):
            client = make_client("llama3.2", _config())
        assert isinstance(client, OllamaClient)

    def test_make_client_unknown_returns_ollama(self) -> None:
        with patch.dict(sys.modules, _SDK_PATCHES):
            client = make_client("my-custom-local-model", _config())
        assert isinstance(client, OllamaClient)

    def test_make_client_mistral_returns_ollama(self) -> None:
        with patch.dict(sys.modules, _SDK_PATCHES):
            client = make_client("mistral", _config())
        assert isinstance(client, OllamaClient)

    def test_make_client_passes_ollama_host(self) -> None:
        with patch.dict(sys.modules, _SDK_PATCHES):
            client = make_client("llama3.2", _config(ollama_host="http://myhost:11434"))
        assert isinstance(client, OllamaClient)


class TestMissingKeys:
    def test_make_client_missing_anthropic_key_raises_model_error(self) -> None:
        cfg = _config(anthropic_key=None)
        with pytest.raises(ModelError) as exc_info:
            make_client("claude-haiku-4-5", cfg)
        err = exc_info.value
        assert err.code == "MISSING_API_KEY"
        assert "ANTHROPIC_API_KEY" in err.hint

    def test_make_client_missing_openai_key_raises_model_error(self) -> None:
        cfg = _config(openai_key=None)
        with pytest.raises(ModelError) as exc_info:
            make_client("gpt-4o", cfg)
        err = exc_info.value
        assert err.code == "MISSING_API_KEY"
        assert "OPENAI_API_KEY" in err.hint

    def test_make_client_missing_gemini_key_raises_model_error(self) -> None:
        cfg = _config(gemini_key=None)
        with pytest.raises(ModelError) as exc_info:
            make_client("gemini-2.5-flash", cfg)
        err = exc_info.value
        assert err.code == "MISSING_API_KEY"
        assert "GEMINI_API_KEY" in err.hint

    def test_make_client_ollama_no_key_required(self) -> None:
        # Ollama doesn't need any API key — should not raise
        cfg = _config(anthropic_key=None, openai_key=None, gemini_key=None)
        with patch.dict(sys.modules, _SDK_PATCHES):
            client = make_client("llama3.2", cfg)
        assert isinstance(client, OllamaClient)
