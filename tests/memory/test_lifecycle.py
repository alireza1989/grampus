"""Tests for grampus.memory.lifecycle — LifecycleTierManager and AdaptiveRetriever."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from grampus.memory.lifecycle.adaptive_router import AdaptiveRetriever
from grampus.memory.lifecycle.tier_manager import LifecycleTierManager
from grampus.memory.lifecycle.types import (
    MemoryTier,
    MemoryType,
    QueryClassification,
    TierRecord,
)
from grampus.memory.manager import MemoryRecallResult
from grampus.memory.types import EpisodicRecord, RetrievedRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_store(existing: object | None = None) -> AsyncMock:
    store = AsyncMock()
    store.get = AsyncMock(return_value=(existing, "etag"))
    store.save = AsyncMock(return_value=None)
    return store


def _unit_vec(*vals: float) -> list[float]:
    mag = math.sqrt(sum(v * v for v in vals))
    return [v / mag for v in vals]


def _make_ep_record(record_id: str = "r1") -> EpisodicRecord:
    return EpisodicRecord(
        id=record_id,
        agent_id="agent1",
        session_id="s1",
        content="some content",
        timestamp=datetime.now(UTC),
    )


def _make_retrieved(record_id: str = "r1") -> RetrievedRecord:
    return RetrievedRecord(
        record=_make_ep_record(record_id),
        score=0.8,
        recency_score=0.9,
        similarity_score=0.7,
        importance_score=0.5,
    )


# ===========================================================================
# LifecycleTierManager tests
# ===========================================================================


@pytest.mark.asyncio
async def test_get_tier_returns_cold_for_unknown() -> None:
    store = _mock_store(existing=None)
    mgr = LifecycleTierManager(state_store=store, agent_id="agent1")
    tier = await mgr.get_tier("unknown-record")
    assert tier == MemoryTier.COLD


@pytest.mark.asyncio
async def test_record_access_creates_tier_record() -> None:
    store = _mock_store(existing=None)
    mgr = LifecycleTierManager(state_store=store, agent_id="agent1")
    tier = await mgr.record_access("r1", MemoryType.EPISODIC)
    # First access: still COLD (need >= 1 in 7d to go WARM)
    # _COLD_TO_WARM_THRESHOLD_7D = 1, so 1 access should promote
    assert tier == MemoryTier.WARM
    store.save.assert_called()


@pytest.mark.asyncio
async def test_record_access_promotes_cold_to_warm_at_threshold() -> None:
    store = _mock_store(existing=None)
    mgr = LifecycleTierManager(state_store=store, agent_id="agent1")
    # First access from COLD → should promote to WARM (threshold=1)
    tier = await mgr.record_access("r1", MemoryType.EPISODIC)
    assert tier == MemoryTier.WARM


@pytest.mark.asyncio
async def test_record_access_promotes_warm_to_hot_at_threshold() -> None:
    now = datetime.now(UTC)
    warm_record = TierRecord(
        record_id="r1",
        memory_type=MemoryType.EPISODIC,
        agent_id="agent1",
        current_tier=MemoryTier.WARM,
        access_count_7d=2,
        last_accessed=now - timedelta(minutes=5),
    )

    saved_records: dict = {}

    async def mock_get(entity: str, key: str, model: object) -> tuple:
        if key in saved_records:
            return saved_records[key], "etag"
        if "r1" in key and "hot_index" not in key:
            return warm_record, "etag"
        return None, ""

    async def mock_save(entity: str, key: str, value: object, **_: object) -> None:
        saved_records[key] = value

    store = AsyncMock()
    store.get = AsyncMock(side_effect=mock_get)
    store.save = AsyncMock(side_effect=mock_save)

    mgr = LifecycleTierManager(state_store=store, agent_id="agent1")
    # 3rd access in 7d → should promote to HOT
    tier = await mgr.record_access("r1", MemoryType.EPISODIC)
    assert tier == MemoryTier.HOT


@pytest.mark.asyncio
async def test_sweep_demotes_stale_hot_record() -> None:
    old_time = datetime.now(UTC) - timedelta(hours=2)
    hot_record = TierRecord(
        record_id="r1",
        memory_type=MemoryType.EPISODIC,
        agent_id="agent1",
        current_tier=MemoryTier.HOT,
        last_accessed=old_time,
    )

    async def mock_get(entity: str, key: str, model: object) -> tuple:
        if "hot_index" in key:
            return ["r1"], "etag"
        if "r1" in key:
            return hot_record, "etag"
        return None, ""

    store = AsyncMock()
    store.get = AsyncMock(side_effect=mock_get)
    store.save = AsyncMock(return_value=None)

    mgr = LifecycleTierManager(state_store=store, agent_id="agent1")
    stats = await mgr.sweep()
    assert stats.total_demotions >= 1


@pytest.mark.asyncio
async def test_sweep_demotes_stale_warm_record() -> None:
    old_time = datetime.now(UTC) - timedelta(days=10)
    warm_record = TierRecord(
        record_id="r1",
        memory_type=MemoryType.EPISODIC,
        agent_id="agent1",
        current_tier=MemoryTier.WARM,
        access_count_7d=0,
        last_accessed=old_time,
    )

    async def mock_get(entity: str, key: str, model: object) -> tuple:
        if "hot_index" in key:
            return ["r1"], "etag"
        if "r1" in key:
            return warm_record, "etag"
        return None, ""

    store = AsyncMock()
    store.get = AsyncMock(side_effect=mock_get)
    store.save = AsyncMock(return_value=None)

    mgr = LifecycleTierManager(state_store=store, agent_id="agent1")
    stats = await mgr.sweep()
    assert stats.total_demotions >= 1


@pytest.mark.asyncio
async def test_record_access_never_raises() -> None:
    store = AsyncMock()
    store.get = AsyncMock(side_effect=RuntimeError("store fail"))
    store.save = AsyncMock(side_effect=RuntimeError("store fail"))
    mgr = LifecycleTierManager(state_store=store, agent_id="agent1")
    # Must not raise
    tier = await mgr.record_access("r1", MemoryType.EPISODIC)
    assert tier == MemoryTier.COLD


@pytest.mark.asyncio
async def test_get_hot_record_ids_returns_hot_only() -> None:
    async def mock_get(entity: str, key: str, model: object) -> tuple:
        if "hot_index" in key:
            return ["r1", "r2"], "etag"
        return None, ""

    store = AsyncMock()
    store.get = AsyncMock(side_effect=mock_get)
    mgr = LifecycleTierManager(state_store=store, agent_id="agent1")
    ids = await mgr.get_hot_record_ids()
    assert "r1" in ids
    assert "r2" in ids


# ===========================================================================
# AdaptiveRetriever — classify() tests
# ===========================================================================


def _make_adaptive_router(
    *,
    with_graph: bool = True,
) -> AdaptiveRetriever:
    ep_ret = AsyncMock()
    ep_ret.retrieve = AsyncMock(return_value=[])
    sem_ret = AsyncMock()
    sem_ret.retrieve_similar = AsyncMock(return_value=[])

    if with_graph:
        from grampus.memory.graph.types import GraphQueryResult

        graph_ret = AsyncMock()
        graph_ret.query = AsyncMock(
            return_value=GraphQueryResult(nodes=[], query="", traversal_depth=0)
        )
    else:
        graph_ret = None

    ep_mem = AsyncMock()
    ep_mem.list_all = AsyncMock(return_value=[])

    return AdaptiveRetriever(
        episodic_retriever=ep_ret,
        semantic_retriever=sem_ret,
        graph_retriever=graph_ret,
        episodic_memory=ep_mem,
    )


def test_classify_sequential_query() -> None:
    router = _make_adaptive_router()
    assert router.classify("what did we discuss last time?") == QueryClassification.SEQUENTIAL
    assert router.classify("previously we talked about this") == QueryClassification.SEQUENTIAL
    assert router.classify("earlier you mentioned X") == QueryClassification.SEQUENTIAL


def test_classify_graph_query() -> None:
    router = _make_adaptive_router(with_graph=True)
    # Long query > 80 chars
    long_q = "Can you explain how the authentication system relates to the authorization layer and why it was designed this way?"
    assert router.classify(long_q) == QueryClassification.GRAPH
    # Short query with graph keywords
    assert router.classify("how does X work?") == QueryClassification.GRAPH
    assert router.classify("why did it fail?") == QueryClassification.GRAPH


def test_classify_flat_query() -> None:
    router = _make_adaptive_router()
    assert router.classify("what is the API endpoint?") == QueryClassification.FLAT
    assert router.classify("list all users") == QueryClassification.FLAT


def test_classify_flat_when_no_graph_retriever() -> None:
    router = _make_adaptive_router(with_graph=False)
    long_q = (
        "how does the system relate to everything and why does the effect cause all the problems?"
    )
    # Even with graph keywords, no retriever → FLAT
    assert router.classify(long_q) == QueryClassification.FLAT


# ===========================================================================
# AdaptiveRetriever — retrieve() tests
# ===========================================================================


@pytest.mark.asyncio
async def test_retrieve_routes_to_flat_by_default() -> None:
    ep_rec = _make_retrieved()
    ep_ret = AsyncMock()
    ep_ret.retrieve = AsyncMock(return_value=[ep_rec])
    sem_ret = AsyncMock()
    sem_ret.retrieve_similar = AsyncMock(return_value=[])

    router = AdaptiveRetriever(
        episodic_retriever=ep_ret,
        semantic_retriever=sem_ret,
        graph_retriever=None,
        episodic_memory=None,
    )
    result = await router.retrieve("agent1", "simple query", top_k=5)
    assert isinstance(result, MemoryRecallResult)
    assert len(result.episodic) == 1


@pytest.mark.asyncio
async def test_retrieve_routes_to_sequential_on_keywords() -> None:
    ep_rec = _make_ep_record("r1")
    ep_mem = AsyncMock()
    ep_mem.list_all = AsyncMock(return_value=[ep_rec])
    ep_ret = AsyncMock()
    ep_ret.retrieve = AsyncMock(return_value=[])
    sem_ret = AsyncMock()
    sem_ret.retrieve_similar = AsyncMock(return_value=[])

    router = AdaptiveRetriever(
        episodic_retriever=ep_ret,
        semantic_retriever=sem_ret,
        graph_retriever=None,
        episodic_memory=ep_mem,
    )
    result = await router.retrieve("agent1", "what did we discuss recently?", top_k=5)
    assert isinstance(result, MemoryRecallResult)
    assert len(result.episodic) == 1


@pytest.mark.asyncio
async def test_retrieve_never_raises() -> None:
    ep_ret = AsyncMock()
    ep_ret.retrieve = AsyncMock(side_effect=RuntimeError("fail"))
    sem_ret = AsyncMock()
    sem_ret.retrieve_similar = AsyncMock(side_effect=RuntimeError("fail"))

    router = AdaptiveRetriever(
        episodic_retriever=ep_ret,
        semantic_retriever=sem_ret,
    )
    result = await router.retrieve("agent1", "query", top_k=5)
    assert isinstance(result, MemoryRecallResult)
    assert result.episodic == []
    assert result.semantic == []


@pytest.mark.asyncio
async def test_retrieve_graph_falls_back_to_flat_on_empty_graph() -> None:
    from grampus.memory.graph.types import GraphQueryResult

    ep_rec = _make_retrieved()
    ep_ret = AsyncMock()
    ep_ret.retrieve = AsyncMock(return_value=[ep_rec])
    sem_ret = AsyncMock()
    sem_ret.retrieve_similar = AsyncMock(return_value=[])

    graph_ret = AsyncMock()
    graph_ret.query = AsyncMock(
        return_value=GraphQueryResult(nodes=[], query="", traversal_depth=0)
    )

    router = AdaptiveRetriever(
        episodic_retriever=ep_ret,
        semantic_retriever=sem_ret,
        graph_retriever=graph_ret,
    )
    # Long query → classified as GRAPH, but returns empty → falls back to flat
    long_q = "explain how everything relates to each other and why things work the way they do"
    result = await router.retrieve("agent1", long_q, top_k=5)
    assert isinstance(result, MemoryRecallResult)
    assert len(result.episodic) == 1  # came from flat fallback


# ===========================================================================
# MemoryManager integration tests (F3 paths)
# ===========================================================================


def _build_memory_manager(
    *,
    adaptive_router: AdaptiveRetriever | None = None,
) -> object:
    from grampus.memory.consolidation import ConsolidationPipeline
    from grampus.memory.episodic import EpisodicMemory
    from grampus.memory.manager import MemoryManager
    from grampus.memory.procedural import ProceduralMemory
    from grampus.memory.retriever import EpisodicRetriever
    from grampus.memory.semantic import SemanticMemory
    from grampus.memory.semantic_retriever import SemanticRetriever
    from grampus.memory.summarizer import Summarizer
    from grampus.memory.token_counter import TokenCounter
    from grampus.memory.working import WorkingMemory

    store = AsyncMock()
    store.get = AsyncMock(return_value=(None, ""))
    store.save = AsyncMock(return_value=None)
    store.delete = AsyncMock(return_value=None)

    ep_svc = AsyncMock()
    ep_svc.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

    token_counter = MagicMock(spec=TokenCounter)
    token_counter.count = MagicMock(return_value=10)
    summarizer = MagicMock(spec=Summarizer)

    working = WorkingMemory(
        state_store=store,
        token_counter=token_counter,
        summarizer=summarizer,
        agent_id="a1",
        session_id="s1",
    )
    episodic = EpisodicMemory(state_store=store, embedding_service=ep_svc, agent_id="a1")
    semantic = SemanticMemory(state_store=store, agent_id="a1")
    procedural = ProceduralMemory(state_store=store, agent_id="a1")
    ep_ret = EpisodicRetriever(episodic_memory=episodic, embedding_service=ep_svc)
    sem_ret = SemanticRetriever(semantic_memory=semantic, embedding_service=ep_svc)
    consolidation = MagicMock(spec=ConsolidationPipeline)

    return MemoryManager(
        working_memory=working,
        episodic_memory=episodic,
        semantic_memory=semantic,
        procedural_memory=procedural,
        episodic_retriever=ep_ret,
        semantic_retriever=sem_ret,
        consolidation_pipeline=consolidation,
        agent_id="a1",
        adaptive_router=adaptive_router,
    )


@pytest.mark.asyncio
async def test_recall_routes_through_adaptive_router_when_set() -> None:
    expected = MemoryRecallResult(episodic=[_make_retrieved()], semantic=[], query="test")
    router = AsyncMock(spec=AdaptiveRetriever)
    router.retrieve = AsyncMock(return_value=expected)
    router.classify = MagicMock(return_value=QueryClassification.FLAT)

    mgr = _build_memory_manager(adaptive_router=router)
    result = await mgr.recall("test query")
    router.retrieve.assert_called_once()
    assert result.episodic == expected.episodic


@pytest.mark.asyncio
async def test_recall_falls_back_to_existing_path_when_no_router() -> None:
    mgr = _build_memory_manager(adaptive_router=None)
    # Should not raise and should return an empty MemoryRecallResult
    result = await mgr.recall("test query")
    assert isinstance(result, MemoryRecallResult)


@pytest.mark.asyncio
async def test_recall_suppresses_router_error_and_falls_back() -> None:
    router = AsyncMock(spec=AdaptiveRetriever)
    router.retrieve = AsyncMock(side_effect=RuntimeError("router boom"))
    router.classify = MagicMock(return_value=QueryClassification.FLAT)

    mgr = _build_memory_manager(adaptive_router=router)
    # Should fall back to existing path without raising
    result = await mgr.recall("test query")
    assert isinstance(result, MemoryRecallResult)


# ===========================================================================
# AgentRunner integration tests (F3 hooks)
# ===========================================================================


def _build_runner(*, graph_builder: object | None = None) -> object:
    from grampus.core.models.base import ModelResponse
    from grampus.core.types import TokenUsage
    from grampus.orchestration.runner import AgentRunner
    from grampus.tools.executor import ToolExecutor

    response = ModelResponse(
        content="done",
        tool_calls=[],
        token_usage=TokenUsage(
            input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.001, model="test"
        ),
        model="test",
        stop_reason="end_turn",
    )
    model_client = AsyncMock()
    model_client.complete = AsyncMock(return_value=response)

    tool_executor = MagicMock(spec=ToolExecutor)

    return AgentRunner(
        model_client=model_client,
        tool_executor=tool_executor,
        graph_builder=graph_builder,
    )


@pytest.mark.asyncio
async def test_runner_initializes_session_graph_on_run() -> None:
    from grampus.core.types import AgentDefinition

    graph_builder = MagicMock()
    graph_builder.init_session = MagicMock(return_value=MagicMock())
    graph_builder.append_event = AsyncMock(return_value=None)
    graph_builder.end_session = MagicMock(return_value=None)

    runner = _build_runner(graph_builder=graph_builder)
    agent_def = AgentDefinition(name="test-agent", model="test")
    await runner.run(agent_def, "hello", session_id="sess1")

    graph_builder.init_session.assert_called_once_with("sess1", "test-agent")


@pytest.mark.asyncio
async def test_runner_consolidates_at_session_end() -> None:
    from grampus.core.types import AgentDefinition
    from grampus.memory.graph.types import EventGraph

    event_graph = EventGraph(session_id="sess1", agent_id="test-agent")
    graph_builder = MagicMock()
    graph_builder.init_session = MagicMock(return_value=event_graph)
    graph_builder.append_event = AsyncMock(return_value=None)
    graph_builder.end_session = MagicMock(return_value=event_graph)

    consolidator = AsyncMock()
    from grampus.memory.graph.types import MemoryGraph

    consolidator.consolidate = AsyncMock(return_value=MemoryGraph(graph_id="test-agent"))

    # Build manager with graph_consolidator
    from grampus.memory.consolidation import ConsolidationPipeline
    from grampus.memory.episodic import EpisodicMemory
    from grampus.memory.manager import MemoryManager
    from grampus.memory.procedural import ProceduralMemory
    from grampus.memory.retriever import EpisodicRetriever
    from grampus.memory.semantic import SemanticMemory
    from grampus.memory.semantic_retriever import SemanticRetriever
    from grampus.memory.summarizer import Summarizer
    from grampus.memory.token_counter import TokenCounter
    from grampus.memory.working import WorkingMemory

    store = AsyncMock()
    store.get = AsyncMock(return_value=(None, ""))
    store.save = AsyncMock(return_value=None)
    ep_svc = AsyncMock()
    ep_svc.embed = AsyncMock(return_value=[0.1])

    token_counter = MagicMock(spec=TokenCounter)
    token_counter.count = MagicMock(return_value=10)
    summarizer = MagicMock(spec=Summarizer)

    working = WorkingMemory(
        state_store=store,
        token_counter=token_counter,
        summarizer=summarizer,
        agent_id="test-agent",
        session_id="sess1",
    )
    episodic = EpisodicMemory(state_store=store, embedding_service=ep_svc, agent_id="test-agent")
    semantic = SemanticMemory(state_store=store, agent_id="test-agent")
    procedural = ProceduralMemory(state_store=store, agent_id="test-agent")
    ep_ret = EpisodicRetriever(episodic_memory=episodic, embedding_service=ep_svc)
    sem_ret = SemanticRetriever(semantic_memory=semantic, embedding_service=ep_svc)
    cp = MagicMock(spec=ConsolidationPipeline)

    mgr = MemoryManager(
        working_memory=working,
        episodic_memory=episodic,
        semantic_memory=semantic,
        procedural_memory=procedural,
        episodic_retriever=ep_ret,
        semantic_retriever=sem_ret,
        consolidation_pipeline=cp,
        agent_id="test-agent",
        graph_consolidator=consolidator,
    )

    from grampus.core.models.base import ModelResponse
    from grampus.core.types import TokenUsage
    from grampus.orchestration.runner import AgentRunner
    from grampus.tools.executor import ToolExecutor

    response = ModelResponse(
        content="done",
        tool_calls=[],
        token_usage=TokenUsage(
            input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.001, model="test"
        ),
        model="test",
        stop_reason="end_turn",
    )
    model_client = AsyncMock()
    model_client.complete = AsyncMock(return_value=response)
    tool_executor = MagicMock(spec=ToolExecutor)

    runner = AgentRunner(
        model_client=model_client,
        tool_executor=tool_executor,
        memory_manager=mgr,
        graph_builder=graph_builder,
    )
    agent_def = AgentDefinition(name="test-agent", model="test")
    await runner.run(agent_def, "hello", session_id="sess1")

    graph_builder.end_session.assert_called_once_with("sess1")
    consolidator.consolidate.assert_called_once()


@pytest.mark.asyncio
async def test_runner_without_graph_builder_unchanged() -> None:
    from grampus.core.types import AgentDefinition

    runner = _build_runner(graph_builder=None)
    agent_def = AgentDefinition(name="test-agent", model="test")
    # Should complete normally with no graph builder
    result = await runner.run(agent_def, "hello", session_id="sess1")
    assert result.output == "done"
