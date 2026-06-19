"""Streaming edge case tests — requires RUN_REAL_LLM_TESTS=true.

Uses the Anthropic client (temperature=0 for determinism).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from grampus.core.models.anthropic import AnthropicClient
from grampus.core.types import Message, Role

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


@pytest.mark.asyncio
async def test_streaming_with_system_prompt(
    anthropic_client: AnthropicClient,
    record_cost: Callable[[float], None],
) -> None:
    """System prompt + user message — both should participate in streaming."""
    messages = [
        Message(role=Role.SYSTEM, content="You are a concise assistant."),
        Message(role=Role.USER, content="What is the capital of France?"),
    ]
    chunks = []
    async for chunk in anthropic_client.stream(
        messages=messages, model=ANTHROPIC_MODEL, temperature=0, max_tokens=32
    ):
        chunks.append(chunk)

    text_chunks = [c for c in chunks if c.delta]
    final = next((c for c in chunks if c.is_final), None)

    assert len(text_chunks) >= 1, "Should yield at least one text chunk"
    full_text = "".join(c.delta for c in chunks)
    assert len(full_text) > 0, "Combined text must be non-empty"
    assert final is not None
    record_cost(final.token_usage.cost_usd if final.token_usage else 0.0)


@pytest.mark.asyncio
async def test_stream_interrupt_cleanup(
    anthropic_client: AnthropicClient,
) -> None:
    """Breaking out of a stream loop must not leave the event loop in a broken state."""
    messages = [Message(role=Role.USER, content="Count from 1 to 100.")]
    got_first = False
    try:
        async for chunk in anthropic_client.stream(
            messages=messages, model=ANTHROPIC_MODEL, max_tokens=256
        ):
            if chunk.delta:
                got_first = True
                break  # intentional early exit
    except Exception as exc:
        pytest.fail(f"Unexpected exception when breaking out of stream: {exc}")

    assert got_first, "Should have received at least one text chunk before breaking"
    # Verify the event loop is still alive after the interrupted stream
    assert asyncio.get_event_loop().is_running()


@pytest.mark.asyncio
async def test_streaming_handles_empty_content(
    anthropic_client: AnthropicClient,
    record_cost: Callable[[float], None],
) -> None:
    """StreamChunk with delta='' (intermediate empty deltas) must not raise AttributeError."""
    messages = [Message(role=Role.USER, content="Say yes.")]
    chunks = []
    try:
        async for chunk in anthropic_client.stream(
            messages=messages, model=ANTHROPIC_MODEL, temperature=0, max_tokens=16
        ):
            # Access .delta even if it's empty — should never raise AttributeError
            _ = chunk.delta
            chunks.append(chunk)
    except AttributeError as exc:
        pytest.fail(f"AttributeError accessing chunk.delta: {exc}")

    total_text = "".join(c.delta for c in chunks)
    assert len(total_text) > 0, "Accumulated text must be non-empty"

    final = next((c for c in chunks if c.is_final), None)
    assert final is not None
    record_cost(final.token_usage.cost_usd if final.token_usage else 0.0)
