"""Tests for nexus.memory.summarizer — conversation compression strategies."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nexus.core.models.base import ModelResponse
from nexus.core.types import Message, Role, TokenUsage
from nexus.memory.summarizer import SummarizationStrategy, Summarizer
from nexus.memory.token_counter import TokenCounter


def make_model_response(content: str) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=[],
        token_usage=TokenUsage(
            input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.0, model="test"
        ),
        model="test",
        stop_reason="end_turn",
    )


def make_messages(n: int, role: Role = Role.USER, prefix: str = "msg") -> list[Message]:
    return [Message(role=role, content=f"{prefix} {i}") for i in range(n)]


@pytest.fixture()
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.complete = AsyncMock(return_value=make_model_response("Summary of the conversation."))
    return client


@pytest.fixture()
def counter() -> TokenCounter:
    return TokenCounter(model="gpt-4")


class TestSummarizationStrategyEnum:
    def test_has_truncate(self) -> None:
        assert SummarizationStrategy.TRUNCATE == "truncate"

    def test_has_summarize(self) -> None:
        assert SummarizationStrategy.SUMMARIZE == "summarize"

    def test_has_hybrid(self) -> None:
        assert SummarizationStrategy.HYBRID == "hybrid"


class TestTruncateStrategy:
    async def test_empty_list_returns_empty(
        self, mock_client: AsyncMock, counter: TokenCounter
    ) -> None:
        s = Summarizer(mock_client, "gpt-4", SummarizationStrategy.TRUNCATE, counter)
        result = await s.summarize([], max_tokens=100)
        assert result == []
        mock_client.complete.assert_not_called()

    async def test_below_limit_returns_unchanged(
        self, mock_client: AsyncMock, counter: TokenCounter
    ) -> None:
        msgs = make_messages(2)
        s = Summarizer(mock_client, "gpt-4", SummarizationStrategy.TRUNCATE, counter)
        result = await s.summarize(msgs, max_tokens=10_000)
        assert result == msgs
        mock_client.complete.assert_not_called()

    async def test_drops_oldest_first(self, mock_client: AsyncMock, counter: TokenCounter) -> None:
        # 20 messages; tiny token budget forces dropping old ones
        msgs = make_messages(20)
        s = Summarizer(mock_client, "gpt-4", SummarizationStrategy.TRUNCATE, counter)
        result = await s.summarize(msgs, max_tokens=10)
        # Last messages should survive
        last_content = msgs[-1].content
        assert any(m.content == last_content for m in result)
        # First message should be dropped
        first_content = msgs[0].content
        assert all(m.content != first_content for m in result)

    async def test_system_messages_always_preserved(
        self, mock_client: AsyncMock, counter: TokenCounter
    ) -> None:
        system = Message(role=Role.SYSTEM, content="You are a helpful assistant.")
        user_msgs = make_messages(20)
        all_msgs = [system] + user_msgs
        s = Summarizer(mock_client, "gpt-4", SummarizationStrategy.TRUNCATE, counter)
        result = await s.summarize(all_msgs, max_tokens=5)
        system_msgs = [m for m in result if m.role == Role.SYSTEM]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "You are a helpful assistant."

    async def test_no_model_call_for_truncate(
        self, mock_client: AsyncMock, counter: TokenCounter
    ) -> None:
        msgs = make_messages(20)
        s = Summarizer(mock_client, "gpt-4", SummarizationStrategy.TRUNCATE, counter)
        await s.summarize(msgs, max_tokens=5)
        mock_client.complete.assert_not_called()


class TestSummarizeStrategy:
    async def test_calls_model_client_once(
        self, mock_client: AsyncMock, counter: TokenCounter
    ) -> None:
        msgs = make_messages(5)
        s = Summarizer(mock_client, "gpt-4", SummarizationStrategy.SUMMARIZE, counter)
        await s.summarize(msgs, max_tokens=100)
        mock_client.complete.assert_called_once()

    async def test_returns_one_summary_message(
        self, mock_client: AsyncMock, counter: TokenCounter
    ) -> None:
        msgs = make_messages(5)
        s = Summarizer(mock_client, "gpt-4", SummarizationStrategy.SUMMARIZE, counter)
        result = await s.summarize(msgs, max_tokens=100)
        non_system = [m for m in result if m.role != Role.SYSTEM]
        assert len(non_system) == 1

    async def test_summary_message_has_is_summary_metadata(
        self, mock_client: AsyncMock, counter: TokenCounter
    ) -> None:
        msgs = make_messages(5)
        s = Summarizer(mock_client, "gpt-4", SummarizationStrategy.SUMMARIZE, counter)
        result = await s.summarize(msgs, max_tokens=100)
        summary = next(m for m in result if m.role != Role.SYSTEM)
        assert summary.metadata.get("is_summary") is True

    async def test_summary_metadata_has_original_count(
        self, mock_client: AsyncMock, counter: TokenCounter
    ) -> None:
        msgs = make_messages(5)
        s = Summarizer(mock_client, "gpt-4", SummarizationStrategy.SUMMARIZE, counter)
        result = await s.summarize(msgs, max_tokens=100)
        summary = next(m for m in result if m.role != Role.SYSTEM)
        assert summary.metadata.get("original_count") == 5

    async def test_preserves_system_message(
        self, mock_client: AsyncMock, counter: TokenCounter
    ) -> None:
        system = Message(role=Role.SYSTEM, content="Be concise.")
        msgs = [system] + make_messages(3)
        s = Summarizer(mock_client, "gpt-4", SummarizationStrategy.SUMMARIZE, counter)
        result = await s.summarize(msgs, max_tokens=100)
        system_msgs = [m for m in result if m.role == Role.SYSTEM]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "Be concise."

    async def test_empty_list_returns_empty_no_call(
        self, mock_client: AsyncMock, counter: TokenCounter
    ) -> None:
        s = Summarizer(mock_client, "gpt-4", SummarizationStrategy.SUMMARIZE, counter)
        result = await s.summarize([], max_tokens=100)
        assert result == []
        mock_client.complete.assert_not_called()

    async def test_summary_content_comes_from_model(
        self, mock_client: AsyncMock, counter: TokenCounter
    ) -> None:
        mock_client.complete.return_value = make_model_response("Key facts preserved.")
        msgs = make_messages(3)
        s = Summarizer(mock_client, "gpt-4", SummarizationStrategy.SUMMARIZE, counter)
        result = await s.summarize(msgs, max_tokens=100)
        summary = next(m for m in result if m.role != Role.SYSTEM)
        assert summary.content == "Key facts preserved."

    async def test_model_receives_all_messages_in_prompt(
        self, mock_client: AsyncMock, counter: TokenCounter
    ) -> None:
        msgs = [
            Message(role=Role.USER, content="My name is Alice."),
            Message(role=Role.ASSISTANT, content="Hello Alice!"),
        ]
        s = Summarizer(mock_client, "gpt-4", SummarizationStrategy.SUMMARIZE, counter)
        await s.summarize(msgs, max_tokens=100)
        call_kwargs = mock_client.complete.call_args
        messages_sent = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
        combined = " ".join(m.content or "" for m in messages_sent)
        assert "Alice" in combined


class TestHybridStrategy:
    async def test_keeps_last_n_messages_verbatim(
        self, mock_client: AsyncMock, counter: TokenCounter
    ) -> None:
        msgs = make_messages(10)
        s = Summarizer(mock_client, "gpt-4", SummarizationStrategy.HYBRID, counter, keep_last_n=3)
        result = await s.summarize(msgs, max_tokens=100)
        recent = [m for m in result if not m.metadata.get("is_summary")]
        # Should have exactly the last 3 non-system messages
        recent_non_sys = [m for m in recent if m.role != Role.SYSTEM]
        assert len(recent_non_sys) == 3
        assert recent_non_sys[-1].content == msgs[-1].content
        assert recent_non_sys[0].content == msgs[-3].content

    async def test_summarizes_older_messages(
        self, mock_client: AsyncMock, counter: TokenCounter
    ) -> None:
        msgs = make_messages(10)
        s = Summarizer(mock_client, "gpt-4", SummarizationStrategy.HYBRID, counter, keep_last_n=3)
        await s.summarize(msgs, max_tokens=100)
        mock_client.complete.assert_called_once()

    async def test_no_summarize_call_when_all_recent(
        self, mock_client: AsyncMock, counter: TokenCounter
    ) -> None:
        msgs = make_messages(3)
        s = Summarizer(mock_client, "gpt-4", SummarizationStrategy.HYBRID, counter, keep_last_n=5)
        result = await s.summarize(msgs, max_tokens=10_000)
        mock_client.complete.assert_not_called()
        assert result == msgs

    async def test_system_message_preserved_in_hybrid(
        self, mock_client: AsyncMock, counter: TokenCounter
    ) -> None:
        system = Message(role=Role.SYSTEM, content="System prompt.")
        msgs = [system] + make_messages(10)
        s = Summarizer(mock_client, "gpt-4", SummarizationStrategy.HYBRID, counter, keep_last_n=3)
        result = await s.summarize(msgs, max_tokens=100)
        system_msgs = [m for m in result if m.role == Role.SYSTEM]
        assert len(system_msgs) == 1

    async def test_summary_metadata_has_is_summary(
        self, mock_client: AsyncMock, counter: TokenCounter
    ) -> None:
        msgs = make_messages(10)
        s = Summarizer(mock_client, "gpt-4", SummarizationStrategy.HYBRID, counter, keep_last_n=3)
        result = await s.summarize(msgs, max_tokens=100)
        summary_msgs = [m for m in result if m.metadata.get("is_summary")]
        assert len(summary_msgs) == 1
