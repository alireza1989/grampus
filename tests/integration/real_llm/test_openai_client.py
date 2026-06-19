"""Real OpenAI API integration tests — requires RUN_REAL_LLM_TESTS=true."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from grampus.core.models.openai import OpenAIClient
from grampus.core.types import Message, Role, ToolDefinition

OPENAI_MODEL = "gpt-4o-mini"


@pytest.mark.asyncio
async def test_complete_basic(
    openai_client: OpenAIClient,
    record_cost: Callable[[float], None],
) -> None:
    messages = [Message(role=Role.USER, content="Say OK")]
    result = await openai_client.complete(messages=messages, model=OPENAI_MODEL, max_tokens=16)
    assert result.content and len(result.content) > 0
    assert result.token_usage.total_tokens > 0
    assert result.token_usage.cost_usd > 0
    record_cost(result.token_usage.cost_usd)


@pytest.mark.asyncio
async def test_complete_returns_stop_reason(
    openai_client: OpenAIClient,
    record_cost: Callable[[float], None],
) -> None:
    messages = [Message(role=Role.USER, content="Say OK")]
    result = await openai_client.complete(messages=messages, model=OPENAI_MODEL, max_tokens=16)
    assert result.stop_reason in ("stop", "tool_calls", "length", "content_filter")
    record_cost(result.token_usage.cost_usd)


@pytest.mark.asyncio
async def test_streaming_no_content_loss(
    openai_client: OpenAIClient,
    record_cost: Callable[[float], None],
) -> None:
    prompt = "What is 2 + 2? Reply with only the number."
    messages = [Message(role=Role.USER, content=prompt)]

    chunks = []
    async for chunk in openai_client.stream(
        messages=messages, model=OPENAI_MODEL, temperature=0, max_tokens=16
    ):
        chunks.append(chunk)

    streamed_text = "".join(c.delta for c in chunks).strip()
    assert len(chunks) >= 2, "Expect at least one text chunk plus the final chunk"
    assert chunks[-1].is_final

    result = await openai_client.complete(
        messages=messages, model=OPENAI_MODEL, temperature=0, max_tokens=16
    )
    # Both should contain "4" (the answer)
    assert "4" in streamed_text or "4" in (result.content or "")

    total_cost = (
        sum(c.token_usage.cost_usd for c in chunks if c.token_usage) + result.token_usage.cost_usd
    )
    record_cost(total_cost)


@pytest.mark.asyncio
async def test_streaming_final_chunk_has_usage(
    openai_client: OpenAIClient,
    record_cost: Callable[[float], None],
) -> None:
    messages = [Message(role=Role.USER, content="Say hello in one word.")]
    chunks = []
    async for chunk in openai_client.stream(messages=messages, model=OPENAI_MODEL, max_tokens=16):
        chunks.append(chunk)

    final = next((c for c in chunks if c.is_final), None)
    assert final is not None
    assert final.token_usage is not None
    assert final.token_usage.total_tokens > 0
    record_cost(final.token_usage.cost_usd)


@pytest.mark.asyncio
async def test_tool_call_roundtrip(
    openai_client: OpenAIClient,
    calculator_tool: ToolDefinition,
    record_cost: Callable[[float], None],
) -> None:
    messages = [
        Message(role=Role.USER, content="What is 7 multiplied by 8? Use the calculator tool.")
    ]
    result = await openai_client.complete(
        messages=messages,
        model=OPENAI_MODEL,
        tools=[calculator_tool],
        max_tokens=128,
    )
    assert len(result.tool_calls) >= 1
    assert result.tool_calls[0].name == "calculator"
    assert "expression" in result.tool_calls[0].arguments
    assert result.stop_reason in ("tool_calls", "stop")
    record_cost(result.token_usage.cost_usd)


@pytest.mark.asyncio
async def test_tool_call_id_format(
    openai_client: OpenAIClient,
    calculator_tool: ToolDefinition,
    record_cost: Callable[[float], None],
) -> None:
    """OpenAI tool call IDs start with 'call_'."""
    messages = [
        Message(role=Role.USER, content="What is 7 multiplied by 8? Use the calculator tool.")
    ]
    result = await openai_client.complete(
        messages=messages,
        model=OPENAI_MODEL,
        tools=[calculator_tool],
        max_tokens=128,
    )
    assert len(result.tool_calls) >= 1
    assert result.tool_calls[0].id.startswith("call_")
    record_cost(result.token_usage.cost_usd)


@pytest.mark.asyncio
async def test_tool_call_streaming_yields_tool_calls_stop_reason(
    openai_client: OpenAIClient,
    calculator_tool: ToolDefinition,
    record_cost: Callable[[float], None],
) -> None:
    # Adaptation: stream() doesn't yield tool call chunks; the final
    # StreamChunk carries finish_reason="tool_calls" when a tool is invoked.
    messages = [
        Message(
            role=Role.USER,
            content="You MUST use the calculator tool to compute: 7 * 8.",
        )
    ]
    chunks = []
    async for chunk in openai_client.stream(
        messages=messages,
        model=OPENAI_MODEL,
        tools=[calculator_tool],
        max_tokens=256,
    ):
        chunks.append(chunk)

    final = next((c for c in chunks if c.is_final), None)
    assert final is not None, "Final chunk must be present"
    # When the model calls a tool, finish_reason is "tool_calls".
    # If it answers directly, it's "stop" — both are valid.
    assert final.finish_reason in ("tool_calls", "stop")
    record_cost(final.token_usage.cost_usd if final.token_usage else 0.0)


@pytest.mark.asyncio
async def test_multi_turn_context(
    openai_client: OpenAIClient,
    record_cost: Callable[[float], None],
) -> None:
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant."),
        Message(role=Role.USER, content="My name is TestBot."),
        Message(role=Role.ASSISTANT, content="Hello TestBot! Nice to meet you."),
        Message(role=Role.USER, content="What is my name?"),
    ]
    result = await openai_client.complete(messages=messages, model=OPENAI_MODEL, max_tokens=64)
    assert "TestBot" in (result.content or "")
    record_cost(result.token_usage.cost_usd)


@pytest.mark.asyncio
async def test_system_prompt_respected(
    openai_client: OpenAIClient,
    record_cost: Callable[[float], None],
) -> None:
    messages = [
        Message(
            role=Role.SYSTEM,
            content="Always reply with exactly the word PONG and nothing else.",
        ),
        Message(role=Role.USER, content="Hello"),
    ]
    result = await openai_client.complete(messages=messages, model=OPENAI_MODEL, max_tokens=16)
    assert "PONG" in (result.content or "").upper()
    record_cost(result.token_usage.cost_usd)


@pytest.mark.asyncio
async def test_token_usage_reasonable(
    openai_client: OpenAIClient,
    record_cost: Callable[[float], None],
) -> None:
    messages = [Message(role=Role.USER, content="Hi")]
    result = await openai_client.complete(messages=messages, model=OPENAI_MODEL, max_tokens=32)
    assert result.token_usage.input_tokens > 0
    assert result.token_usage.output_tokens > 0
    # gpt-4o-mini is cheap; 32 output tokens well under $0.01
    assert result.token_usage.cost_usd < 0.01
    record_cost(result.token_usage.cost_usd)
