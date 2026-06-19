"""Tests verifying error-handling for real API error shapes.

Uses mock SDK clients — no actual API calls are made. Still gated by
RUN_REAL_LLM_TESTS because the conftest module-level guard applies to all
tests in this directory, including mock-based ones.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from grampus.core.errors import ModelError
from grampus.core.models.anthropic import AnthropicClient
from grampus.core.types import Message, Role

pytestmark = [pytest.mark.real_llm, pytest.mark.integration]

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


@pytest.mark.asyncio
async def test_anthropic_429_raises_model_error() -> None:
    """A 429 / rate-limit exception from the SDK is wrapped as ModelError(code='MODEL_API_ERROR')."""
    mock_sdk = MagicMock()
    mock_sdk.messages.create = AsyncMock(side_effect=Exception("Rate limit exceeded: 429"))
    client = AnthropicClient(api_key="fake", _client=mock_sdk)

    messages = [Message(role=Role.USER, content="Hello")]
    with pytest.raises(ModelError) as exc_info:
        await client.complete(messages=messages, model=ANTHROPIC_MODEL, max_tokens=16)

    assert exc_info.value.code == "MODEL_API_ERROR"
    # No retry logic in the current client — SDK called exactly once
    assert mock_sdk.messages.create.call_count == 1


@pytest.mark.asyncio
async def test_token_usage_from_api_not_tiktoken() -> None:
    """AnthropicClient reads token counts from the API response, not from tiktoken.

    Build a mock that returns usage.input_tokens=50 / output_tokens=10 and
    verify the ModelResponse reflects those values exactly.
    """
    # Build the mock content block (text type, no tool_use)
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = "Hello, world!"

    # Build the mock usage object
    mock_usage = MagicMock()
    mock_usage.input_tokens = 50
    mock_usage.output_tokens = 10

    # Build the full mock response
    mock_response = MagicMock()
    mock_response.usage = mock_usage
    mock_response.content = [mock_block]
    mock_response.model = ANTHROPIC_MODEL
    mock_response.stop_reason = "end_turn"

    mock_sdk = MagicMock()
    mock_sdk.messages.create = AsyncMock(return_value=mock_response)

    client = AnthropicClient(api_key="fake", _client=mock_sdk)
    messages = [Message(role=Role.USER, content="Hello")]
    result = await client.complete(messages=messages, model=ANTHROPIC_MODEL, max_tokens=64)

    # Values come from the API response, not from tiktoken estimation
    assert result.token_usage.input_tokens == 50
    assert result.token_usage.output_tokens == 10
    assert result.token_usage.total_tokens == 60
    assert result.content == "Hello, world!"
