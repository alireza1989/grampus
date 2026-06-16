"""Tests for grampus.memory.token_counter — model-aware token counting."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from grampus.core.types import Message, Role, ToolCall, ToolResult
from grampus.memory.token_counter import TokenCounter


@pytest.fixture()
def counter() -> TokenCounter:
    return TokenCounter(model="gpt-4")


class TestModelToEncoding:
    def test_gpt4_uses_cl100k(self, counter: TokenCounter) -> None:
        assert counter.encoding_name == "cl100k_base"

    def test_gpt4o_uses_o200k(self) -> None:
        c = TokenCounter(model="gpt-4o")
        assert c.encoding_name == "o200k_base"

    def test_claude_uses_cl100k(self) -> None:
        c = TokenCounter(model="claude-3-5-haiku-20241022")
        assert c.encoding_name == "cl100k_base"

    def test_claude_sonnet_uses_cl100k(self) -> None:
        c = TokenCounter(model="claude-sonnet-4-6")
        assert c.encoding_name == "cl100k_base"

    def test_unknown_model_uses_fallback(self) -> None:
        c = TokenCounter(model="some-unknown-model-xyz")
        assert c.encoding_name == "fallback"

    def test_gpt35_turbo_uses_cl100k(self) -> None:
        c = TokenCounter(model="gpt-3.5-turbo")
        assert c.encoding_name == "cl100k_base"


class TestCountText:
    def test_hello_world_is_two_tokens(self, counter: TokenCounter) -> None:
        assert counter.count_text("hello world") == 2

    def test_empty_string_is_zero(self, counter: TokenCounter) -> None:
        assert counter.count_text("") == 0

    def test_single_word_is_one(self, counter: TokenCounter) -> None:
        assert counter.count_text("hello") == 1

    def test_longer_text_more_tokens(self, counter: TokenCounter) -> None:
        short = counter.count_text("hi")
        long = counter.count_text("hi " * 100)
        assert long > short

    def test_fallback_counter_approximates(self) -> None:
        c = TokenCounter(model="unknown-model")
        # Fallback: 4 chars per token
        tokens = c.count_text("abcd")  # 4 chars = 1 token
        assert tokens == 1

    def test_fallback_rounds_up(self) -> None:
        c = TokenCounter(model="unknown-model")
        # 5 chars → ceil(5/4) = 2 tokens
        tokens = c.count_text("abcde")
        assert tokens == 2

    @given(st.text(min_size=0, max_size=500))
    def test_count_always_non_negative(self, text: str) -> None:
        c = TokenCounter(model="gpt-4")
        assert c.count_text(text) >= 0


class TestCountMessages:
    def test_empty_list_is_zero(self, counter: TokenCounter) -> None:
        assert counter.count_messages([]) == 0

    def test_single_message_positive(self, counter: TokenCounter) -> None:
        msg = Message(role=Role.USER, content="Hello there")
        assert counter.count_messages([msg]) > 0

    def test_more_messages_more_tokens(self, counter: TokenCounter) -> None:
        msgs_1 = [Message(role=Role.USER, content="Hi")]
        msgs_5 = [Message(role=Role.USER, content="Hi")] * 5
        assert counter.count_messages(msgs_5) > counter.count_messages(msgs_1)

    def test_system_message_counted(self, counter: TokenCounter) -> None:
        msg = Message(role=Role.SYSTEM, content="You are a helpful assistant.")
        assert counter.count_messages([msg]) > 0

    def test_message_with_tool_call_counts_more(self, counter: TokenCounter) -> None:
        plain = Message(role=Role.ASSISTANT, content="Ok")
        with_tool = Message(
            role=Role.ASSISTANT,
            content="Ok",
            tool_calls=[ToolCall(id="t1", name="search", arguments={"q": "python"})],
        )
        assert counter.count_messages([with_tool]) > counter.count_messages([plain])

    def test_message_with_tool_result_counted(self, counter: TokenCounter) -> None:
        msg = Message(
            role=Role.TOOL,
            tool_results=[ToolResult(tool_call_id="t1", output="result text here")],
        )
        assert counter.count_messages([msg]) > 0

    def test_none_content_does_not_crash(self, counter: TokenCounter) -> None:
        msg = Message(role=Role.ASSISTANT, content=None)
        assert counter.count_messages([msg]) >= 0

    def test_accumulated_count_equals_sum_of_individual(self, counter: TokenCounter) -> None:
        msgs = [
            Message(role=Role.USER, content="Question one"),
            Message(role=Role.ASSISTANT, content="Answer one"),
        ]
        total = counter.count_messages(msgs)
        individual = sum(counter.count_messages([m]) for m in msgs)
        assert total == individual

    @given(
        st.lists(
            st.builds(
                Message,
                role=st.sampled_from(list(Role)),
                content=st.one_of(st.none(), st.text(max_size=200)),
            ),
            max_size=20,
        )
    )
    def test_count_messages_always_non_negative(self, messages: list[Message]) -> None:
        c = TokenCounter(model="gpt-4")
        assert c.count_messages(messages) >= 0
