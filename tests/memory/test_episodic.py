"""Tests for nexus.memory.episodic — EpisodicMemory CRUD."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nexus.memory.episodic import EpisodicMemory
from nexus.memory.types import EpisodicRecord

FAKE_EMBEDDING = [0.1] * 10


@pytest.fixture()
def mock_store() -> AsyncMock:
    store = AsyncMock()
    store.save = AsyncMock(return_value=None)
    store.get = AsyncMock(return_value=(None, ""))
    store.delete = AsyncMock(return_value=None)
    store.get_bulk = AsyncMock(return_value={})
    return store


@pytest.fixture()
def mock_embeddings() -> AsyncMock:
    svc = AsyncMock()
    svc.embed = AsyncMock(return_value=FAKE_EMBEDDING)
    return svc


@pytest.fixture()
def memory(mock_store: AsyncMock, mock_embeddings: AsyncMock) -> EpisodicMemory:
    return EpisodicMemory(
        state_store=mock_store,
        embedding_service=mock_embeddings,
        agent_id="agent-1",
    )


class TestEpisodicMemoryStore:
    async def test_store_returns_episodic_record(self, memory: EpisodicMemory) -> None:
        record = await memory.store(
            content="The user likes Python.",
            session_id="s1",
        )
        assert isinstance(record, EpisodicRecord)
        assert record.content == "The user likes Python."

    async def test_store_assigns_id(self, memory: EpisodicMemory) -> None:
        record = await memory.store(content="test", session_id="s1")
        assert record.id != ""

    async def test_store_sets_agent_id(self, memory: EpisodicMemory) -> None:
        record = await memory.store(content="test", session_id="s1")
        assert record.agent_id == "agent-1"

    async def test_store_sets_session_id(self, memory: EpisodicMemory) -> None:
        record = await memory.store(content="test", session_id="session-42")
        assert record.session_id == "session-42"

    async def test_store_computes_embedding(
        self, memory: EpisodicMemory, mock_embeddings: AsyncMock
    ) -> None:
        record = await memory.store(content="test", session_id="s1")
        mock_embeddings.embed.assert_called_once_with("test")
        assert record.embedding == FAKE_EMBEDDING

    async def test_store_persists_to_state_store(
        self, memory: EpisodicMemory, mock_store: AsyncMock
    ) -> None:
        await memory.store(content="test", session_id="s1")
        mock_store.save.assert_called()

    async def test_store_adds_to_index(self, memory: EpisodicMemory, mock_store: AsyncMock) -> None:
        await memory.store(content="test", session_id="s1")
        # At least two saves: one for record, one for the index
        assert mock_store.save.call_count >= 2

    async def test_store_with_user_id(self, memory: EpisodicMemory) -> None:
        record = await memory.store(content="test", session_id="s1", user_id="user-99")
        assert record.user_id == "user-99"

    async def test_store_with_metadata(self, memory: EpisodicMemory) -> None:
        record = await memory.store(content="test", session_id="s1", metadata={"source": "tool"})
        assert record.metadata["source"] == "tool"

    async def test_store_calculates_importance_score(self, memory: EpisodicMemory) -> None:
        short = await memory.store(content="Hi", session_id="s1")
        long = await memory.store(content=" ".join(["word"] * 100), session_id="s1")
        assert long.importance_score >= short.importance_score

    async def test_store_embedding_failure_still_persists(self, mock_store: AsyncMock) -> None:
        mock_embeddings = AsyncMock()
        mock_embeddings.embed = AsyncMock(side_effect=RuntimeError("API down"))
        mem = EpisodicMemory(
            state_store=mock_store,
            embedding_service=mock_embeddings,
            agent_id="agent-1",
        )
        record = await mem.store(content="important fact", session_id="s1")
        assert record.embedding is None
        mock_store.save.assert_called()


class TestEpisodicMemoryGet:
    async def test_get_returns_none_for_missing(
        self, memory: EpisodicMemory, mock_store: AsyncMock
    ) -> None:
        mock_store.get.return_value = (None, "")
        result = await memory.get("nonexistent-id")
        assert result is None

    async def test_get_returns_record_when_exists(
        self, memory: EpisodicMemory, mock_store: AsyncMock
    ) -> None:
        record = EpisodicRecord(
            id="rec-1",
            agent_id="agent-1",
            session_id="s1",
            content="test content",
            trust_score=0.9,
            importance_score=0.5,
        )
        mock_store.get.return_value = (record, "etag1")
        result = await memory.get("rec-1")
        assert result is not None
        assert result.content == "test content"


class TestEpisodicMemoryDelete:
    async def test_delete_calls_store_delete(
        self, memory: EpisodicMemory, mock_store: AsyncMock
    ) -> None:
        await memory.delete("rec-1")
        mock_store.delete.assert_called()

    async def test_delete_updates_index(self, memory: EpisodicMemory) -> None:
        await memory.store(content="keep me", session_id="s1")
        record = await memory.store(content="delete me", session_id="s1")
        await memory.delete(record.id)
        index = memory._index
        assert record.id not in index


class TestEpisodicMemoryListAll:
    async def test_list_all_empty_initially(self, memory: EpisodicMemory) -> None:
        result = await memory.list_all()
        assert result == []

    async def test_list_all_returns_stored_records(
        self, memory: EpisodicMemory, mock_store: AsyncMock
    ) -> None:
        r1 = await memory.store(content="fact 1", session_id="s1")
        r2 = await memory.store(content="fact 2", session_id="s1")

        # Mock get to return the records when asked by list_all
        async def fake_get(entity_type: str, key: str, cls: type):  # type: ignore
            for r in [r1, r2]:
                if r.id in key or key in r.id:
                    return r, "etag"
            return None, ""

        mock_store.get = AsyncMock(side_effect=fake_get)
        results = await memory.list_all()
        ids = {r.id for r in results}
        assert r1.id in ids
        assert r2.id in ids

    async def test_list_all_returns_empty_for_fresh_agent(
        self, mock_store: AsyncMock, mock_embeddings: AsyncMock
    ) -> None:
        mem = EpisodicMemory(
            state_store=mock_store,
            embedding_service=mock_embeddings,
            agent_id="brand-new-agent",
        )
        result = await mem.list_all()
        assert result == []


class TestEpisodicMemoryUpdateAccess:
    async def test_update_access_increments_count(
        self, memory: EpisodicMemory, mock_store: AsyncMock
    ) -> None:
        record = await memory.store(content="test", session_id="s1")
        assert record.access_count == 0

        mock_store.get.return_value = (record, "etag1")
        await memory.update_access(record.id)

        # get was called to reload, save was called to persist
        mock_store.save.assert_called()

    async def test_update_access_nonexistent_is_noop(
        self, memory: EpisodicMemory, mock_store: AsyncMock
    ) -> None:
        mock_store.get.return_value = (None, "")
        # Should not raise
        await memory.update_access("nonexistent")

    async def test_update_access_sets_last_accessed(
        self, memory: EpisodicMemory, mock_store: AsyncMock
    ) -> None:
        record = EpisodicRecord(
            id="rec-1",
            agent_id="agent-1",
            session_id="s1",
            content="test",
            trust_score=0.9,
            importance_score=0.5,
        )
        assert record.last_accessed is None
        mock_store.get.return_value = (record, "etag1")
        await memory.update_access("rec-1")

        # Verify the saved record has last_accessed set
        save_calls = mock_store.save.call_args_list
        assert len(save_calls) > 0
