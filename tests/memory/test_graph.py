"""Tests for nexus.memory.graph — GraphBuilder, SemanticConsolidator, GraphRetriever."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nexus.memory.graph.builder import GraphBuilder
from nexus.memory.graph.consolidator import SemanticConsolidator
from nexus.memory.graph.retriever import GraphRetriever
from nexus.memory.graph.types import (
    ConceptNode,
    EventGraph,
    EventNode,
    GraphQueryResult,
    MemoryGraph,
    RelationshipEdge,
    SemanticShiftEvent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(*vals: float) -> list[float]:
    mag = math.sqrt(sum(v * v for v in vals))
    return [v / mag for v in vals]


def _mock_embedding_service(embedding: list[float] | None = None) -> AsyncMock:
    svc = AsyncMock()
    svc.embed = AsyncMock(return_value=embedding or _unit_vec(1, 0, 0))
    return svc


def _mock_state_store(existing: MemoryGraph | None = None) -> AsyncMock:
    store = AsyncMock()
    store.get = AsyncMock(return_value=(existing, "etag"))
    store.save = AsyncMock(return_value=None)
    return store


def _mock_model_client(concepts: list[dict] | None = None) -> AsyncMock:
    response = MagicMock()
    concepts_data = concepts or []
    import json

    response.content = json.dumps({"concepts": concepts_data})
    client = AsyncMock()
    client.complete = AsyncMock(return_value=response)
    return client


# ===========================================================================
# Graph types tests
# ===========================================================================


def test_concept_node_confidence_range() -> None:
    with pytest.raises(ValidationError):
        ConceptNode(node_id="x", label="x", description="x", confidence=1.5)


def test_memory_graph_default_empty() -> None:
    g = MemoryGraph(graph_id="agent1")
    assert g.nodes == {}
    assert g.edges == []
    assert g.version == 0
    assert g.last_consolidated is None


def test_event_graph_initializes_empty() -> None:
    eg = EventGraph(session_id="s1", agent_id="a1")
    assert eg.events == []
    assert eg.last_embedding is None


# ===========================================================================
# GraphBuilder tests
# ===========================================================================


@pytest.mark.asyncio
async def test_init_session_creates_event_graph() -> None:
    builder = GraphBuilder(embedding_service=_mock_embedding_service())
    graph = builder.init_session("sess1", "agent1")
    assert isinstance(graph, EventGraph)
    assert graph.session_id == "sess1"
    assert graph.agent_id == "agent1"
    assert builder.get_graph("sess1") is graph


@pytest.mark.asyncio
async def test_append_event_adds_to_graph() -> None:
    svc = _mock_embedding_service(_unit_vec(1, 0, 0))
    builder = GraphBuilder(embedding_service=svc)
    builder.init_session("sess1", "agent1")
    result = await builder.append_event("sess1", "tool_call", "some content", "agent1")
    graph = builder.get_graph("sess1")
    assert graph is not None
    assert len(graph.events) == 1
    assert graph.events[0].event_type == "tool_call"
    assert result is None  # below min events threshold


@pytest.mark.asyncio
async def test_append_event_no_shift_below_threshold() -> None:
    # Same embedding every time → distance 0 → no shift
    emb = _unit_vec(1, 0, 0)
    svc = _mock_embedding_service(emb)
    builder = GraphBuilder(embedding_service=svc, shift_threshold=0.30)
    builder.init_session("sess1", "agent1")

    for i in range(5):
        result = await builder.append_event("sess1", "tool_call", f"content {i}", "agent1")

    assert result is None


@pytest.mark.asyncio
async def test_append_event_returns_shift_event_above_threshold() -> None:
    call_count = 0
    embeddings = [_unit_vec(1, 0, 0)] * 3 + [_unit_vec(0, 1, 0)]

    async def embed_fn(text: str) -> list[float]:
        nonlocal call_count
        idx = min(call_count, len(embeddings) - 1)
        call_count += 1
        return embeddings[idx]

    svc = AsyncMock()
    svc.embed = embed_fn

    builder = GraphBuilder(embedding_service=svc, shift_threshold=0.30)
    builder.init_session("sess1", "agent1")

    # First 3 events — same direction, no shift
    for i in range(3):
        r = await builder.append_event("sess1", "tool_call", f"content {i}", "agent1")
        assert r is None

    # 4th event — orthogonal → big shift
    result = await builder.append_event("sess1", "tool_call", "shifted content", "agent1")
    assert isinstance(result, SemanticShiftEvent)
    assert result.shift_distance > 0.30


@pytest.mark.asyncio
async def test_append_event_no_check_below_min_events() -> None:
    svc = _mock_embedding_service(_unit_vec(0, 1, 0))
    builder = GraphBuilder(embedding_service=svc, shift_threshold=0.01)
    builder.init_session("sess1", "agent1")

    # Only 2 events — below min (3)
    for i in range(2):
        r = await builder.append_event("sess1", "tool_call", f"content {i}", "agent1")
        assert r is None


@pytest.mark.asyncio
async def test_append_event_never_raises() -> None:
    svc = AsyncMock()
    svc.embed = AsyncMock(side_effect=RuntimeError("boom"))
    builder = GraphBuilder(embedding_service=svc)
    builder.init_session("sess1", "agent1")

    # Should not raise
    result = await builder.append_event("sess1", "tool_call", "data", "agent1")
    assert result is None


@pytest.mark.asyncio
async def test_end_session_removes_graph() -> None:
    builder = GraphBuilder(embedding_service=_mock_embedding_service())
    builder.init_session("sess1", "agent1")
    removed = builder.end_session("sess1")
    assert removed is not None
    assert builder.get_graph("sess1") is None


def test_get_graph_returns_none_for_unknown_session() -> None:
    builder = GraphBuilder(embedding_service=_mock_embedding_service())
    assert builder.get_graph("nonexistent") is None


# ===========================================================================
# SemanticConsolidator tests
# ===========================================================================


@pytest.mark.asyncio
async def test_consolidate_creates_new_graph_on_first_run() -> None:
    store = _mock_state_store(existing=None)
    svc = _mock_embedding_service(_unit_vec(1, 0, 0))
    concepts = [{"label": "Python", "description": "Programming language", "relations": []}]
    client = _mock_model_client(concepts)

    consolidator = SemanticConsolidator(
        state_store=store, embedding_service=svc, model_client=client
    )
    event_graph = EventGraph(session_id="s1", agent_id="agent1")
    event_graph.events.append(
        EventNode(
            event_id="e1",
            event_type="tool_call",
            content_summary="Python code execution",
            session_id="s1",
            agent_id="agent1",
        )
    )

    graph = await consolidator.consolidate(event_graph, "agent1")
    assert isinstance(graph, MemoryGraph)
    assert graph.version == 1
    assert len(graph.nodes) == 1


@pytest.mark.asyncio
async def test_consolidate_merges_similar_existing_node() -> None:
    existing_emb = _unit_vec(1, 0, 0)
    existing_node = ConceptNode(
        node_id="n1",
        label="Python",
        description="old description",
        embedding=existing_emb,
        frequency=1,
    )
    existing_graph = MemoryGraph(graph_id="agent1", nodes={"n1": existing_node})
    store = _mock_state_store(existing=existing_graph)
    # New concept has same embedding → similarity > 0.85 → merge
    svc = _mock_embedding_service(existing_emb)
    concepts = [{"label": "Python", "description": "Updated description", "relations": []}]
    client = _mock_model_client(concepts)

    consolidator = SemanticConsolidator(
        state_store=store, embedding_service=svc, model_client=client
    )
    event_graph = EventGraph(session_id="s1", agent_id="agent1")
    event_graph.events.append(
        EventNode(
            event_id="e1",
            event_type="tool_call",
            content_summary="Python",
            session_id="s1",
            agent_id="agent1",
        )
    )

    graph = await consolidator.consolidate(event_graph, "agent1")
    # Should still have 1 node (merged)
    assert len(graph.nodes) == 1
    merged = graph.nodes["n1"]
    assert merged.frequency == 2


@pytest.mark.asyncio
async def test_consolidate_adds_new_node_when_dissimilar() -> None:
    existing_emb = _unit_vec(1, 0, 0)
    new_emb = _unit_vec(0, 1, 0)
    existing_node = ConceptNode(
        node_id="n1", label="Python", description="lang", embedding=existing_emb
    )
    existing_graph = MemoryGraph(graph_id="agent1", nodes={"n1": existing_node})
    store = _mock_state_store(existing=existing_graph)
    svc = _mock_embedding_service(new_emb)
    concepts = [{"label": "Database", "description": "Storage system", "relations": []}]
    client = _mock_model_client(concepts)

    consolidator = SemanticConsolidator(
        state_store=store, embedding_service=svc, model_client=client
    )
    event_graph = EventGraph(session_id="s1", agent_id="agent1")
    event_graph.events.append(
        EventNode(
            event_id="e1",
            event_type="tool_call",
            content_summary="db stuff",
            session_id="s1",
            agent_id="agent1",
        )
    )

    graph = await consolidator.consolidate(event_graph, "agent1")
    assert len(graph.nodes) == 2


@pytest.mark.asyncio
async def test_consolidate_increments_graph_version() -> None:
    store = _mock_state_store(existing=None)
    svc = _mock_embedding_service()
    concepts = [{"label": "X", "description": "X desc", "relations": []}]
    client = _mock_model_client(concepts)

    consolidator = SemanticConsolidator(
        state_store=store, embedding_service=svc, model_client=client
    )
    event_graph = EventGraph(session_id="s1", agent_id="agent1")
    event_graph.events.append(
        EventNode(
            event_id="e1",
            event_type="tool_call",
            content_summary="content",
            session_id="s1",
            agent_id="agent1",
        )
    )

    graph = await consolidator.consolidate(event_graph, "agent1")
    assert graph.version == 1


@pytest.mark.asyncio
async def test_consolidate_never_raises() -> None:
    store = _mock_state_store(existing=None)
    svc = AsyncMock()
    svc.embed = AsyncMock(side_effect=RuntimeError("embed fail"))
    client = AsyncMock()
    client.complete = AsyncMock(side_effect=RuntimeError("llm fail"))

    consolidator = SemanticConsolidator(
        state_store=store, embedding_service=svc, model_client=client
    )
    event_graph = EventGraph(session_id="s1", agent_id="agent1")
    event_graph.events.append(
        EventNode(
            event_id="e1",
            event_type="tool_call",
            content_summary="x",
            session_id="s1",
            agent_id="agent1",
        )
    )

    # Must not raise
    graph = await consolidator.consolidate(event_graph, "agent1")
    assert isinstance(graph, MemoryGraph)


@pytest.mark.asyncio
async def test_load_graph_returns_empty_for_new_agent() -> None:
    store = _mock_state_store(existing=None)
    svc = _mock_embedding_service()
    client = _mock_model_client()

    consolidator = SemanticConsolidator(
        state_store=store, embedding_service=svc, model_client=client
    )
    graph = await consolidator.load_graph("new-agent")
    assert isinstance(graph, MemoryGraph)
    assert graph.graph_id == "new-agent"
    assert len(graph.nodes) == 0


@pytest.mark.asyncio
async def test_save_and_load_graph_round_trip() -> None:
    saved: dict = {}

    async def mock_save(entity: str, key: str, value: object, **_: object) -> None:
        saved[key] = value

    async def mock_get(entity: str, key: str, model: object) -> tuple:
        if key in saved:
            v = saved[key]
            if isinstance(v, MemoryGraph):
                return v, "etag"
        return None, ""

    store = AsyncMock()
    store.save = AsyncMock(side_effect=mock_save)
    store.get = AsyncMock(side_effect=mock_get)

    svc = _mock_embedding_service()
    client = _mock_model_client()
    consolidator = SemanticConsolidator(
        state_store=store, embedding_service=svc, model_client=client
    )

    original = MemoryGraph(graph_id="agent1", version=3)
    original.nodes["n1"] = ConceptNode(node_id="n1", label="X", description="Desc")
    await consolidator.save_graph(original)
    loaded = await consolidator.load_graph("agent1")
    assert loaded.version == 3
    assert "n1" in loaded.nodes


# ===========================================================================
# GraphRetriever tests
# ===========================================================================


@pytest.mark.asyncio
async def test_query_returns_empty_on_empty_graph() -> None:
    store = _mock_state_store(existing=MemoryGraph(graph_id="agent1"))
    svc = _mock_embedding_service()
    client = _mock_model_client()
    consolidator = SemanticConsolidator(store, svc, client)
    retriever = GraphRetriever(consolidator=consolidator, embedding_service=svc)

    result = await retriever.query("agent1", "some query")
    assert result.nodes == []


@pytest.mark.asyncio
async def test_query_returns_seed_nodes_by_similarity() -> None:
    emb = _unit_vec(1, 0, 0)
    node = ConceptNode(node_id="n1", label="Python", description="lang", embedding=emb)
    graph = MemoryGraph(graph_id="agent1", nodes={"n1": node})
    store = _mock_state_store(existing=graph)
    svc = _mock_embedding_service(emb)
    client = _mock_model_client()
    consolidator = SemanticConsolidator(store, svc, client)
    retriever = GraphRetriever(consolidator=consolidator, embedding_service=svc)

    result = await retriever.query("agent1", "Python code", top_k=5)
    assert len(result.nodes) == 1
    assert result.nodes[0].node_id == "n1"


@pytest.mark.asyncio
async def test_query_traverses_connected_nodes() -> None:
    emb_a = _unit_vec(1, 0, 0)
    emb_b = _unit_vec(0, 1, 0)
    node_a = ConceptNode(node_id="n_a", label="A", description="Node A", embedding=emb_a)
    node_b = ConceptNode(node_id="n_b", label="B", description="Node B", embedding=emb_b)
    edge = RelationshipEdge(
        edge_id="e1", source_node_id="n_a", target_node_id="n_b", relation_type="related_to"
    )
    graph = MemoryGraph(graph_id="agent1", nodes={"n_a": node_a, "n_b": node_b}, edges=[edge])

    store = _mock_state_store(existing=graph)
    # Query points toward A
    svc = _mock_embedding_service(emb_a)
    client = _mock_model_client()
    consolidator = SemanticConsolidator(store, svc, client)
    retriever = GraphRetriever(consolidator=consolidator, embedding_service=svc)

    result = await retriever.query("agent1", "query", top_k=5, traversal_depth=2)
    node_ids = {n.node_id for n in result.nodes}
    # B should be reachable via traversal from A
    assert "n_a" in node_ids
    assert "n_b" in node_ids


@pytest.mark.asyncio
async def test_query_respects_top_k() -> None:
    emb = _unit_vec(1, 0, 0)
    nodes = {
        f"n{i}": ConceptNode(
            node_id=f"n{i}", label=f"Concept {i}", description="desc", embedding=emb
        )
        for i in range(10)
    }
    graph = MemoryGraph(graph_id="agent1", nodes=nodes)
    store = _mock_state_store(existing=graph)
    svc = _mock_embedding_service(emb)
    client = _mock_model_client()
    consolidator = SemanticConsolidator(store, svc, client)
    retriever = GraphRetriever(consolidator=consolidator, embedding_service=svc)

    result = await retriever.query("agent1", "query", top_k=3)
    assert len(result.nodes) <= 3


def test_format_as_context_empty_returns_empty_string() -> None:
    svc = _mock_embedding_service()
    store = _mock_state_store()
    client = _mock_model_client()
    consolidator = SemanticConsolidator(store, svc, client)
    retriever = GraphRetriever(consolidator=consolidator, embedding_service=svc)

    result = GraphQueryResult(nodes=[], query="x", traversal_depth=0)
    assert retriever.format_as_context(result) == ""


def test_format_as_context_formats_nodes() -> None:
    svc = _mock_embedding_service()
    store = _mock_state_store()
    client = _mock_model_client()
    consolidator = SemanticConsolidator(store, svc, client)
    retriever = GraphRetriever(consolidator=consolidator, embedding_service=svc)

    nodes = [ConceptNode(node_id="n1", label="Python", description="A programming language")]
    result = GraphQueryResult(nodes=nodes, query="x", traversal_depth=0)
    ctx = retriever.format_as_context(result)
    assert "Python" in ctx
    assert "programming language" in ctx
    assert ctx.startswith("Knowledge graph context:")
