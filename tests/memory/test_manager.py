"""Tests for grampus.memory.manager — MemoryManager unified interface."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from grampus.core.types import Message, Role
from grampus.memory.consolidation import ConsolidationResult
from grampus.memory.manager import MemoryManager, MemoryRecallResult
from grampus.memory.types import EpisodicRecord, RetrievedRecord, SemanticFact

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_episodic_record(content: str = "test content") -> EpisodicRecord:
    return EpisodicRecord(
        id=str(uuid.uuid4()),
        agent_id="agent-1",
        session_id="sess-1",
        content=content,
    )


def make_retrieved_record(content: str = "test content") -> RetrievedRecord:
    return RetrievedRecord(
        record=make_episodic_record(content),
        score=0.8,
        recency_score=0.9,
        similarity_score=0.7,
        importance_score=0.6,
    )


def make_semantic_fact(subject: str = "user") -> SemanticFact:
    return SemanticFact(
        id=str(uuid.uuid4()),
        subject=subject,
        predicate="prefers",
        object_value="dark mode",
        confidence=0.9,
    )


def make_message(content: str = "hello") -> Message:
    return Message(role=Role.USER, content=content)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_working() -> AsyncMock:
    wm = AsyncMock()
    wm.add = AsyncMock(return_value=None)
    wm.get_messages = AsyncMock(return_value=[])
    return wm


@pytest.fixture()
def mock_episodic() -> AsyncMock:
    em = AsyncMock()
    em.store = AsyncMock(return_value=make_episodic_record())
    em.delete = AsyncMock(return_value=None)
    return em


@pytest.fixture()
def mock_semantic() -> AsyncMock:
    sm = AsyncMock()
    stored_fact = make_semantic_fact()
    sm.store = AsyncMock(return_value=stored_fact)
    sm.delete = AsyncMock(return_value=None)
    return sm


@pytest.fixture()
def mock_procedural() -> AsyncMock:
    return AsyncMock()


@pytest.fixture()
def mock_ep_retriever() -> AsyncMock:
    r = AsyncMock()
    r.retrieve = AsyncMock(return_value=[make_retrieved_record()])
    return r


@pytest.fixture()
def mock_sem_retriever() -> AsyncMock:
    r = AsyncMock()
    r.retrieve_similar = AsyncMock(return_value=[])
    return r


@pytest.fixture()
def mock_consolidation() -> AsyncMock:
    c = AsyncMock()
    c.run = AsyncMock(
        return_value=ConsolidationResult(facts_extracted=3, facts_merged=1, episodes_processed=2)
    )
    return c


@pytest.fixture()
def manager(
    mock_working: AsyncMock,
    mock_episodic: AsyncMock,
    mock_semantic: AsyncMock,
    mock_procedural: AsyncMock,
    mock_ep_retriever: AsyncMock,
    mock_sem_retriever: AsyncMock,
    mock_consolidation: AsyncMock,
) -> MemoryManager:
    return MemoryManager(
        working_memory=mock_working,
        episodic_memory=mock_episodic,
        semantic_memory=mock_semantic,
        procedural_memory=mock_procedural,
        episodic_retriever=mock_ep_retriever,
        semantic_retriever=mock_sem_retriever,
        consolidation_pipeline=mock_consolidation,
        agent_id="agent-1",
    )


# ---------------------------------------------------------------------------
# remember()
# ---------------------------------------------------------------------------


class TestRemember:
    async def test_remember_episodic_calls_episodic_store(
        self, manager: MemoryManager, mock_episodic: AsyncMock
    ) -> None:
        await manager.remember("I like Python", session_id="s1")
        mock_episodic.store.assert_awaited_once()

    async def test_remember_episodic_passes_content_and_session(
        self, manager: MemoryManager, mock_episodic: AsyncMock
    ) -> None:
        await manager.remember("I prefer dark mode", session_id="sess-42")
        call_kwargs = mock_episodic.store.call_args
        assert call_kwargs.kwargs["session_id"] == "sess-42"
        assert call_kwargs.args[0] == "I prefer dark mode"

    async def test_remember_semantic_calls_semantic_store(
        self, manager: MemoryManager, mock_semantic: AsyncMock
    ) -> None:
        await manager.remember("user prefers dark mode", session_id="s1", memory_types=["semantic"])
        mock_semantic.store.assert_awaited_once()

    async def test_remember_semantic_does_not_call_episodic(
        self,
        manager: MemoryManager,
        mock_episodic: AsyncMock,
        mock_semantic: AsyncMock,
    ) -> None:
        await manager.remember("fact content", session_id="s1", memory_types=["semantic"])
        mock_episodic.store.assert_not_awaited()

    async def test_remember_both_types_calls_both_stores(
        self,
        manager: MemoryManager,
        mock_episodic: AsyncMock,
        mock_semantic: AsyncMock,
    ) -> None:
        await manager.remember("content", session_id="s1", memory_types=["episodic", "semantic"])
        mock_episodic.store.assert_awaited_once()
        mock_semantic.store.assert_awaited_once()

    async def test_remember_defaults_to_episodic(
        self, manager: MemoryManager, mock_episodic: AsyncMock
    ) -> None:
        await manager.remember("default type content", session_id="s1")
        mock_episodic.store.assert_awaited_once()

    async def test_remember_unknown_type_is_silently_ignored(
        self,
        manager: MemoryManager,
        mock_episodic: AsyncMock,
        mock_semantic: AsyncMock,
    ) -> None:
        # Should not raise even with an unknown type
        await manager.remember("content", session_id="s1", memory_types=["unknown_type"])
        mock_episodic.store.assert_not_awaited()
        mock_semantic.store.assert_not_awaited()

    async def test_remember_semantic_creates_fact_with_content(
        self, manager: MemoryManager, mock_semantic: AsyncMock
    ) -> None:
        await manager.remember("Alice prefers Python", session_id="s1", memory_types=["semantic"])
        stored_fact = mock_semantic.store.call_args.args[0]
        assert isinstance(stored_fact, SemanticFact)
        assert stored_fact.object_value == "Alice prefers Python"


# ---------------------------------------------------------------------------
# recall()
# ---------------------------------------------------------------------------


class TestRecall:
    async def test_recall_returns_memory_recall_result(self, manager: MemoryManager) -> None:
        result = await manager.recall("What does user prefer?")
        assert isinstance(result, MemoryRecallResult)

    async def test_recall_includes_query_in_result(self, manager: MemoryManager) -> None:
        result = await manager.recall("my query string")
        assert result.query == "my query string"

    async def test_recall_episodic_queries_retriever(
        self, manager: MemoryManager, mock_ep_retriever: AsyncMock
    ) -> None:
        await manager.recall("test query", memory_types=["episodic"])
        mock_ep_retriever.retrieve.assert_awaited_once()

    async def test_recall_semantic_queries_retriever(
        self, manager: MemoryManager, mock_sem_retriever: AsyncMock
    ) -> None:
        await manager.recall("test query", memory_types=["semantic"])
        mock_sem_retriever.retrieve_similar.assert_awaited_once()

    async def test_recall_defaults_episodic_and_semantic(
        self,
        manager: MemoryManager,
        mock_ep_retriever: AsyncMock,
        mock_sem_retriever: AsyncMock,
    ) -> None:
        await manager.recall("query")
        mock_ep_retriever.retrieve.assert_awaited_once()
        mock_sem_retriever.retrieve_similar.assert_awaited_once()

    async def test_recall_respects_top_k(
        self, manager: MemoryManager, mock_ep_retriever: AsyncMock
    ) -> None:
        await manager.recall("query", memory_types=["episodic"], top_k=3)
        call_kwargs = mock_ep_retriever.retrieve.call_args
        assert call_kwargs.kwargs["top_k"] == 3

    async def test_recall_episodic_only_skips_semantic(
        self,
        manager: MemoryManager,
        mock_sem_retriever: AsyncMock,
    ) -> None:
        await manager.recall("query", memory_types=["episodic"])
        mock_sem_retriever.retrieve_similar.assert_not_awaited()

    async def test_recall_semantic_only_skips_episodic(
        self,
        manager: MemoryManager,
        mock_ep_retriever: AsyncMock,
    ) -> None:
        await manager.recall("query", memory_types=["semantic"])
        mock_ep_retriever.retrieve.assert_not_awaited()

    async def test_recall_unknown_type_silently_ignored(
        self,
        manager: MemoryManager,
        mock_ep_retriever: AsyncMock,
        mock_sem_retriever: AsyncMock,
    ) -> None:
        result = await manager.recall("query", memory_types=["unknown"])
        mock_ep_retriever.retrieve.assert_not_awaited()
        mock_sem_retriever.retrieve_similar.assert_not_awaited()
        assert result.episodic == []
        assert result.semantic == []

    async def test_recall_episodic_results_in_result(
        self,
        manager: MemoryManager,
        mock_ep_retriever: AsyncMock,
    ) -> None:
        records = [make_retrieved_record("content A"), make_retrieved_record("content B")]
        mock_ep_retriever.retrieve.return_value = records
        result = await manager.recall("query", memory_types=["episodic"])
        assert result.episodic == records

    async def test_recall_semantic_results_in_result(
        self,
        manager: MemoryManager,
        mock_sem_retriever: AsyncMock,
    ) -> None:
        from grampus.memory.semantic_retriever import ScoredFact

        fact = make_semantic_fact()
        scored = ScoredFact(
            id=fact.id,
            subject=fact.subject,
            predicate=fact.predicate,
            object_value=fact.object_value,
            score=0.9,
            fact=fact,
        )
        mock_sem_retriever.retrieve_similar.return_value = [scored]
        result = await manager.recall("query", memory_types=["semantic"])
        assert len(result.semantic) == 1
        assert result.semantic[0].id == fact.id

    async def test_recall_passes_query_to_retrievers(
        self,
        manager: MemoryManager,
        mock_ep_retriever: AsyncMock,
        mock_sem_retriever: AsyncMock,
    ) -> None:
        query = "specific user preference"
        await manager.recall(query)
        assert mock_ep_retriever.retrieve.call_args.args[0] == query
        assert mock_sem_retriever.retrieve_similar.call_args.args[0] == query


# ---------------------------------------------------------------------------
# forget()
# ---------------------------------------------------------------------------


class TestForget:
    async def test_forget_episodic_calls_episodic_delete(
        self, manager: MemoryManager, mock_episodic: AsyncMock
    ) -> None:
        await manager.forget("rec-1", memory_type="episodic")
        mock_episodic.delete.assert_awaited_once_with("rec-1")

    async def test_forget_semantic_calls_semantic_delete(
        self, manager: MemoryManager, mock_semantic: AsyncMock
    ) -> None:
        await manager.forget("fact-1", memory_type="semantic")
        mock_semantic.delete.assert_awaited_once_with("fact-1")

    async def test_forget_unknown_type_raises_value_error(self, manager: MemoryManager) -> None:
        with pytest.raises(ValueError, match="unknown"):
            await manager.forget("id-1", memory_type="unknown")

    async def test_forget_episodic_does_not_touch_semantic(
        self,
        manager: MemoryManager,
        mock_semantic: AsyncMock,
    ) -> None:
        await manager.forget("rec-1", memory_type="episodic")
        mock_semantic.delete.assert_not_awaited()

    async def test_forget_semantic_does_not_touch_episodic(
        self,
        manager: MemoryManager,
        mock_episodic: AsyncMock,
    ) -> None:
        await manager.forget("fact-1", memory_type="semantic")
        mock_episodic.delete.assert_not_awaited()


# ---------------------------------------------------------------------------
# consolidate()
# ---------------------------------------------------------------------------


class TestConsolidate:
    async def test_consolidate_calls_pipeline_run(
        self, manager: MemoryManager, mock_consolidation: AsyncMock
    ) -> None:
        await manager.consolidate()
        mock_consolidation.run.assert_awaited_once()

    async def test_consolidate_returns_consolidation_result(self, manager: MemoryManager) -> None:
        result = await manager.consolidate()
        assert isinstance(result, ConsolidationResult)
        assert result.facts_extracted == 3
        assert result.facts_merged == 1
        assert result.episodes_processed == 2


# ---------------------------------------------------------------------------
# add_message() / get_messages()
# ---------------------------------------------------------------------------


class TestMessageDelegation:
    async def test_add_message_delegates_to_working_memory(
        self, manager: MemoryManager, mock_working: AsyncMock
    ) -> None:
        msg = make_message("hello")
        await manager.add_message(msg)
        mock_working.add.assert_awaited_once_with(msg)

    async def test_get_messages_delegates_to_working_memory(
        self, manager: MemoryManager, mock_working: AsyncMock
    ) -> None:
        expected = [make_message("a"), make_message("b")]
        mock_working.get_messages.return_value = expected
        result = await manager.get_messages()
        assert result == expected
        mock_working.get_messages.assert_awaited_once()

    async def test_get_messages_returns_empty_list_by_default(self, manager: MemoryManager) -> None:
        result = await manager.get_messages()
        assert result == []


# ---------------------------------------------------------------------------
# MemoryRecallResult model
# ---------------------------------------------------------------------------


class TestMemoryRecallResult:
    def test_default_empty_lists(self) -> None:
        result = MemoryRecallResult(query="q")
        assert result.episodic == []
        assert result.semantic == []

    def test_stores_query(self) -> None:
        result = MemoryRecallResult(query="my query")
        assert result.query == "my query"

    def test_json_round_trip(self) -> None:
        record = make_retrieved_record()
        fact = make_semantic_fact()
        result = MemoryRecallResult(query="q", episodic=[record], semantic=[fact])
        restored = MemoryRecallResult.model_validate_json(result.model_dump_json())
        assert restored.query == "q"
        assert len(restored.episodic) == 1
        assert len(restored.semantic) == 1
