"""Tests for CausalWorldModel — persistent SCM + LLM labeling integration."""

from __future__ import annotations

import pytest

from nexus.causal.types import (
    CausalDiagnosis,
    CausalGraph,
    CausalRelation,
    InterventionQuery,
    RootCauseCandidate,
    WorldModelGraph,
)
from nexus.causal.world_model import CausalWorldModel, _slugify

# -----------------------------------------------------------------------
# Fakes
# -----------------------------------------------------------------------


class _FakeExtractor:
    """Returns a fixed list of relations."""

    def __init__(self, relations: list[CausalRelation]) -> None:
        self._relations = relations

    async def extract(self, text: str, *, session_id: str, agent_id: str) -> list[CausalRelation]:
        if not text or not text.strip():
            return []
        return list(self._relations)


class _ErrorExtractor:
    async def extract(self, text: str, *, session_id: str, agent_id: str) -> list[CausalRelation]:
        raise RuntimeError("extractor exploded")


class _FakeStore:
    def __init__(self) -> None:
        self._data: dict[str, WorldModelGraph] = {}

    async def get(
        self, entity: str, key: str, model_cls: type
    ) -> tuple[WorldModelGraph | None, str | None]:
        return self._data.get(key), None

    async def save(self, entity: str, key: str, value: WorldModelGraph) -> None:
        self._data[key] = value


class _ErrorStore:
    async def get(self, entity: str, key: str, model_cls: type) -> tuple[None, None]:
        raise RuntimeError("store exploded")

    async def save(self, entity: str, key: str, value: object) -> None:
        raise RuntimeError("store exploded")


def _make_relation(cause: str, effect: str) -> CausalRelation:
    import uuid

    return CausalRelation(
        relation_id=str(uuid.uuid4()),
        cause_description=cause,
        effect_description=effect,
        relation_type="caused",
        confidence=0.8,
        evidence_text="test evidence",
        session_id="sess-1",
        agent_id="agent-1",
    )


def _make_diagnosis(
    session_id: str = "sess-1", root_causes: list[RootCauseCandidate] | None = None
) -> CausalDiagnosis:
    graph = CausalGraph(
        graph_id=session_id,
        agent_id="agent-1",
        nodes={
            "e1": {
                "event_type": "llm_call",
                "description": "llm output",
                "failed": False,
                "step_index": 0,
            },
            "e2": {
                "event_type": "tool_call",
                "description": "tool failure",
                "failed": True,
                "step_index": 1,
            },
        },
    )
    if root_causes is None:
        root_causes = [
            RootCauseCandidate(
                event_id="e1",
                event_type="llm_call",
                description="llm output",
                structural_score=0.8,
                positional_score=0.9,
                composite_score=0.85,
                causal_chain=["e1", "e2"],
            )
        ]
    return CausalDiagnosis(
        session_id=session_id,
        agent_id="agent-1",
        failure_event_id="e2",
        root_causes=root_causes,
        causal_graph=graph,
    )


# -----------------------------------------------------------------------
# observe() tests
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_observe_adds_variables_to_graph():
    rel = _make_relation("tool web search", "query answered")
    store = _FakeStore()
    wm = CausalWorldModel(store, _FakeExtractor([rel]), agent_id="agent-1")
    await wm.observe("some text", session_id="sess-1")
    graph = await wm.load()
    cause_id = _slugify("tool web search")
    assert cause_id in graph.variables


@pytest.mark.asyncio
async def test_observe_adds_adjacency_edge():
    rel = _make_relation("tool web search", "query answered")
    store = _FakeStore()
    wm = CausalWorldModel(store, _FakeExtractor([rel]), agent_id="agent-1")
    await wm.observe("some text", session_id="sess-1")
    graph = await wm.load()
    cause_id = _slugify("tool web search")
    effect_id = _slugify("query answered")
    assert effect_id in graph.adjacency.get(cause_id, [])


@pytest.mark.asyncio
async def test_observe_never_raises():
    store = _FakeStore()
    wm = CausalWorldModel(store, _ErrorExtractor(), agent_id="agent-1")
    result = await wm.observe("some text", session_id="sess-1")
    assert result == []


@pytest.mark.asyncio
async def test_observe_empty_text_returns_empty():
    rel = _make_relation("a", "b")
    store = _FakeStore()
    wm = CausalWorldModel(store, _FakeExtractor([rel]), agent_id="agent-1")
    result = await wm.observe("", session_id="sess-1")
    assert result == []


# -----------------------------------------------------------------------
# query_intervention() tests
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_intervention_on_populated_model():
    rel = _make_relation("tool_a", "outcome_b")
    store = _FakeStore()
    wm = CausalWorldModel(store, _FakeExtractor([rel]), agent_id="agent-1")
    await wm.observe("text", session_id="sess-1")

    q = InterventionQuery(
        natural_language="What if tool_a?",
        target_variable=_slugify("tool_a"),
        outcome_variable=_slugify("outcome_b"),
        intervention_value="active",
    )
    result = await wm.query_intervention(q)
    assert result is not None
    assert result.is_identifiable is True


@pytest.mark.asyncio
async def test_query_intervention_never_raises():
    wm = CausalWorldModel(_ErrorStore(), _FakeExtractor([]), agent_id="agent-1")
    q = InterventionQuery(
        natural_language="?",
        target_variable="x",
        outcome_variable="y",
        intervention_value="z",
    )
    result = await wm.query_intervention(q)
    assert result is not None
    assert result.confidence == 0.0


# -----------------------------------------------------------------------
# absorb_diagnosis() tests
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_absorb_diagnosis_adds_causal_chain_edges():
    store = _FakeStore()
    wm = CausalWorldModel(store, _FakeExtractor([]), agent_id="agent-1")
    diagnosis = _make_diagnosis()
    await wm.absorb_diagnosis(diagnosis)
    graph = await wm.load()
    # e1 description → e2 description edge should exist
    cause_id = _slugify("llm output")
    effect_id = _slugify("tool failure")
    assert cause_id in graph.variables
    assert effect_id in graph.adjacency.get(cause_id, [])


@pytest.mark.asyncio
async def test_absorb_diagnosis_skips_empty_candidates():
    store = _FakeStore()
    wm = CausalWorldModel(store, _FakeExtractor([]), agent_id="agent-1")
    diagnosis = _make_diagnosis(root_causes=[])
    await wm.absorb_diagnosis(diagnosis)
    graph = await wm.load()
    assert graph.variables == {}


@pytest.mark.asyncio
async def test_absorb_diagnosis_never_raises():
    wm = CausalWorldModel(_ErrorStore(), _FakeExtractor([]), agent_id="agent-1")
    diagnosis = _make_diagnosis()
    # Should not raise
    await wm.absorb_diagnosis(diagnosis)


# -----------------------------------------------------------------------
# load() / save() tests
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_returns_empty_for_new_agent():
    store = _FakeStore()
    wm = CausalWorldModel(store, _FakeExtractor([]), agent_id="agent-new")
    graph = await wm.load()
    assert isinstance(graph, WorldModelGraph)
    assert graph.variables == {}
    assert graph.adjacency == {}


@pytest.mark.asyncio
async def test_save_and_load_round_trip():
    store = _FakeStore()
    wm = CausalWorldModel(store, _FakeExtractor([]), agent_id="agent-1")
    await wm.observe("text with causal claim", session_id="sess-1")
    # Manually save a graph and reload it
    graph = WorldModelGraph(
        agent_id="agent-1",
        variables={"v1": "Variable One", "v2": "Variable Two"},
        adjacency={"v1": ["v2"]},
        version=3,
    )
    await wm.save(graph)
    loaded = await wm.load()
    assert loaded.variables == graph.variables
    assert loaded.adjacency == graph.adjacency
    assert loaded.version == 3


# -----------------------------------------------------------------------
# _slugify utility
# -----------------------------------------------------------------------


def test_slugify_stable_across_calls():
    slug1 = _slugify("Tool Web Search")
    slug2 = _slugify("Tool Web Search")
    assert slug1 == slug2
    assert slug1 == "tool_web_search"
