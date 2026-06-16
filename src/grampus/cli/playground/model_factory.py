"""Factory that instantiates the correct ModelClient from a model name."""

from __future__ import annotations

from grampus.core.config import GrampusConfig
from grampus.core.errors import ModelError
from grampus.core.models.base import ModelClient


def make_client(model: str, config: GrampusConfig) -> ModelClient:
    """Instantiate the right ModelClient for the given model name.

    Routing rules (prefix-based):
      claude-*           → AnthropicClient
      gpt-* / o1* / o3* → OpenAIClient
      gemini-*           → GeminiClient
      anything else      → OllamaClient

    Raises:
        ModelError: When a required API key is absent, with an actionable hint.
    """
    m = model.lower()

    if m.startswith("claude"):
        from grampus.core.models.anthropic import AnthropicClient

        key = config.model.anthropic_api_key
        if not key:
            raise ModelError(
                "Anthropic API key is required for Claude models",
                code="MISSING_API_KEY",
                details={"model": model},
                hint=(
                    "Set ANTHROPIC_API_KEY in your environment or "
                    "under model.anthropic_api_key in grampus.yaml."
                ),
            )
        return AnthropicClient(api_key=key.get_secret_value())

    if m.startswith(("gpt-", "o1", "o3")):
        from grampus.core.models.openai import OpenAIClient

        key = config.model.openai_api_key
        if not key:
            raise ModelError(
                "OpenAI API key is required for GPT/o1/o3 models",
                code="MISSING_API_KEY",
                details={"model": model},
                hint=(
                    "Set OPENAI_API_KEY in your environment or "
                    "under model.openai_api_key in grampus.yaml."
                ),
            )
        return OpenAIClient(api_key=key.get_secret_value())

    if m.startswith("gemini"):
        from grampus.core.models.gemini import GeminiClient

        key = config.model.gemini_api_key
        if not key:
            raise ModelError(
                "Gemini API key is required for Gemini models",
                code="MISSING_API_KEY",
                details={"model": model},
                hint=(
                    "Set GEMINI_API_KEY in your environment or "
                    "under model.gemini_api_key in grampus.yaml."
                ),
            )
        return GeminiClient(api_key=key.get_secret_value())

    # Default: Ollama (local, no API key needed)
    from grampus.core.models.ollama import OllamaClient

    return OllamaClient(host=config.model.ollama_host)
