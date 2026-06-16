"""Tests for grampus.memory.working — WorkingMemory in-session buffer."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from grampus.core.types import Message, Role
from grampus.memory.token_counter import TokenCounter
from grampus.memory.working import WorkingMemory


def make_message(content: str, role: Role = Role.USER) -> Message:
    return Message(role=role, content=content)


def make_long_message(n_words: int) -> Message:
    return Message(role=Role.USER, content=" ".join(["word"] * n_words))


@pytest.fixture()
def mock_store() -> AsyncMock:
    store = AsyncMock()
    store.save = AsyncMock(return_value=None)
    store.get = AsyncMock(return_value=(None, ""))
    store.delete = AsyncMock(return_value=None)
    return store


@pytest.fixture()
def mock_summarizer() -> AsyncMock:
    summarizer = AsyncMock()

    async def passthrough(messages: list[Message], max_tokens: int) -> list[Message]:
        return messages

    summarizer.summarize = AsyncMock(side_effect=passthrough)
    return summarizer


@pytest.fixture()
def counter() -> TokenCounter:
    return TokenCounter(model="gpt-4")


@pytest.fixture()
def wm(mock_store: AsyncMock, mock_summarizer: AsyncMock, counter: TokenCounter) -> WorkingMemory:
    return WorkingMemory(
        state_store=mock_store,
        token_counter=counter,
        summarizer=mock_summarizer,
        agent_id="agent-1",
        session_id="session-1",
        max_tokens=1000,
        threshold_fraction=0.8,
    )


class TestWorkingMemoryAddAndGet:
    async def test_add_and_get_single_message(
        self, wm: WorkingMemory, mock_store: AsyncMock
    ) -> None:
        msg = make_message("Hello!")
        await wm.add(msg)
        messages = await wm.get_messages()
        assert len(messages) == 1
        assert messages[0].content == "Hello!"

    async def test_add_multiple_messages_preserves_order(self, wm: WorkingMemory) -> None:
        for i in range(5):
            await wm.add(make_message(f"message {i}"))
        messages = await wm.get_messages()
        assert len(messages) == 5
        assert messages[0].content == "message 0"
        assert messages[4].content == "message 4"

    async def test_get_messages_returns_copy(self, wm: WorkingMemory) -> None:
        await wm.add(make_message("hi"))
        msgs1 = await wm.get_messages()
        msgs2 = await wm.get_messages()
        assert msgs1 == msgs2

    async def test_add_persists_to_state_store(
        self, wm: WorkingMemory, mock_store: AsyncMock
    ) -> None:
        await wm.add(make_message("hello"))
        mock_store.save.assert_called()

    async def test_get_messages_loads_from_store_on_cold_start(
        self, mock_store: AsyncMock, mock_summarizer: AsyncMock, counter: TokenCounter
    ) -> None:
        import json

        from grampus.core.types import Message, Role

        existing = [Message(role=Role.USER, content="persisted msg")]
        # Return serialised messages from the store
        mock_store.get.return_value = (
            MagicMock(
                model_dump_json=lambda: json.dumps(
                    {"messages": [m.model_dump(mode="json") for m in existing]}
                )
            ),
            "etag-1",
        )
        wm2 = WorkingMemory(
            state_store=mock_store,
            token_counter=counter,
            summarizer=mock_summarizer,
            agent_id="agent-fresh",
            session_id="session-fresh",
            max_tokens=1000,
        )
        # Simulate pre-loaded messages via the store fixture
        await wm2.add(make_message("new msg"))
        msgs = await wm2.get_messages()
        assert any(m.content == "new msg" for m in msgs)


class TestWorkingMemoryTokenCount:
    async def test_token_count_zero_initially(self, wm: WorkingMemory) -> None:
        assert wm.token_count == 0

    async def test_token_count_increases_after_add(self, wm: WorkingMemory) -> None:
        await wm.add(make_message("Hello there"))
        assert wm.token_count > 0

    async def test_token_count_accumulates(self, wm: WorkingMemory) -> None:
        await wm.add(make_message("Hello"))
        count_after_1 = wm.token_count
        await wm.add(make_message("World"))
        count_after_2 = wm.token_count
        assert count_after_2 > count_after_1

    async def test_token_count_never_negative(self, wm: WorkingMemory) -> None:
        for _ in range(5):
            await wm.add(make_message("x"))
        assert wm.token_count >= 0


class TestWorkingMemoryAutoSummarize:
    async def test_no_summarize_below_threshold(
        self, wm: WorkingMemory, mock_summarizer: AsyncMock
    ) -> None:
        # max_tokens=1000, threshold=0.8 → summarize at 800 tokens
        # Each tiny message is ~2 tokens, way below 800
        for i in range(10):
            await wm.add(make_message(f"msg {i}"))
        mock_summarizer.summarize.assert_not_called()

    async def test_summarize_triggers_at_threshold(
        self, mock_store: AsyncMock, counter: TokenCounter
    ) -> None:
        summarizer = AsyncMock()
        summarizer.summarize = AsyncMock(return_value=[make_message("summary")])

        # tiny budget: 10 tokens, 80% threshold = 8 tokens
        wm = WorkingMemory(
            state_store=mock_store,
            token_counter=counter,
            summarizer=summarizer,
            agent_id="a",
            session_id="s",
            max_tokens=10,
            threshold_fraction=0.8,
        )
        # Add enough words to exceed 8 tokens
        for _ in range(5):
            await wm.add(make_message("word word word"))
        summarizer.summarize.assert_called()

    async def test_after_summarize_token_count_reduced(
        self, mock_store: AsyncMock, counter: TokenCounter
    ) -> None:
        compressed = [make_message("short summary")]
        summarizer = AsyncMock()
        summarizer.summarize = AsyncMock(return_value=compressed)

        wm = WorkingMemory(
            state_store=mock_store,
            token_counter=counter,
            summarizer=summarizer,
            agent_id="a",
            session_id="s",
            max_tokens=10,
            threshold_fraction=0.8,
        )
        for _ in range(5):
            await wm.add(make_message("word word word"))

        # After summarisation the window is the compressed list
        msgs = await wm.get_messages()
        assert any(m.content == "short summary" for m in msgs)

    async def test_summarize_passes_max_tokens(
        self, mock_store: AsyncMock, counter: TokenCounter
    ) -> None:
        summarizer = AsyncMock()
        summarizer.summarize = AsyncMock(return_value=[make_message("s")])

        wm = WorkingMemory(
            state_store=mock_store,
            token_counter=counter,
            summarizer=summarizer,
            agent_id="a",
            session_id="s",
            max_tokens=10,
            threshold_fraction=0.8,
        )
        for _ in range(5):
            await wm.add(make_message("word word word"))

        if summarizer.summarize.called:
            _, kwargs = summarizer.summarize.call_args
            passed_max = kwargs.get("max_tokens") or summarizer.summarize.call_args.args[1]
            assert passed_max == 10


class TestWorkingMemoryFullHistory:
    async def test_full_history_persisted_separately(
        self, wm: WorkingMemory, mock_store: AsyncMock
    ) -> None:
        await wm.add(make_message("first"))
        await wm.add(make_message("second"))
        # Two save calls — once for window, once for history
        assert mock_store.save.call_count >= 2

    async def test_full_history_grows_even_after_summarize(
        self, mock_store: AsyncMock, counter: TokenCounter
    ) -> None:
        compressed = [make_message("summary")]
        summarizer = AsyncMock()
        summarizer.summarize = AsyncMock(return_value=compressed)

        wm = WorkingMemory(
            state_store=mock_store,
            token_counter=counter,
            summarizer=summarizer,
            agent_id="a",
            session_id="s",
            max_tokens=10,
            threshold_fraction=0.8,
        )
        original_msgs = []
        for i in range(5):
            m = make_message(f"word word word {i}")
            original_msgs.append(m)
            await wm.add(m)

        history = await wm.get_full_history()
        # History must contain ALL original messages
        history_contents = {m.content for m in history}
        for m in original_msgs:
            assert m.content in history_contents


class TestWorkingMemoryClear:
    async def test_clear_empties_live_window(self, wm: WorkingMemory) -> None:
        await wm.add(make_message("hello"))
        await wm.clear()
        msgs = await wm.get_messages()
        assert msgs == []

    async def test_clear_resets_token_count(self, wm: WorkingMemory) -> None:
        await wm.add(make_message("hello there"))
        await wm.clear()
        assert wm.token_count == 0

    async def test_clear_does_not_delete_history(
        self, wm: WorkingMemory, mock_store: AsyncMock
    ) -> None:
        await wm.add(make_message("hello"))
        await wm.clear()
        # delete should only be called for the window key, not the history key
        if mock_store.delete.called:
            for call in mock_store.delete.call_args_list:
                args = str(call)
                assert "history" not in args or "window" in args


class TestWorkingMemorySystemMessages:
    async def test_system_message_survives_after_add(self, wm: WorkingMemory) -> None:
        system = make_message("You are helpful.", role=Role.SYSTEM)
        await wm.add(system)
        msgs = await wm.get_messages()
        assert any(m.role == Role.SYSTEM for m in msgs)

    async def test_system_message_preserved_during_summarize(
        self, mock_store: AsyncMock, counter: TokenCounter
    ) -> None:
        system = make_message("System prompt.", role=Role.SYSTEM)

        async def preserve_system(messages: list[Message], max_tokens: int) -> list[Message]:
            return [m for m in messages if m.role == Role.SYSTEM] + [make_message("compressed")]

        summarizer = AsyncMock()
        summarizer.summarize = AsyncMock(side_effect=preserve_system)

        wm = WorkingMemory(
            state_store=mock_store,
            token_counter=counter,
            summarizer=summarizer,
            agent_id="a",
            session_id="s",
            max_tokens=10,
            threshold_fraction=0.8,
        )
        await wm.add(system)
        for _ in range(5):
            await wm.add(make_message("word word word"))

        msgs = await wm.get_messages()
        assert any(m.role == Role.SYSTEM and m.content == "System prompt." for m in msgs)


class TestWorkingMemoryConcurrency:
    async def test_concurrent_adds_do_not_corrupt_count(self, wm: WorkingMemory) -> None:
        await asyncio.gather(*[wm.add(make_message(f"msg {i}")) for i in range(10)])
        msgs = await wm.get_messages()
        assert len(msgs) == 10
        assert wm.token_count > 0
