"""Tests for CausalTracer — event log → causal graph → root cause diagnosis."""

from __future__ import annotations

import pytest

from grampus.causal.tracer import _DATA_DEP_MIN_OVERLAP_CHARS, _MAX_BACKWARD_DEPTH, CausalTracer
from grampus.causal.types import CausalDiagnosis, CausalGraph, EdgeType


def _make_event(
    event_id: str,
    step_index: int,
    *,
    failed: bool = False,
    error_message: str | None = None,
    input_snippet: str | None = None,
    output_snippet: str | None = None,
    event_type: str = "llm_call",
) -> dict:
    return {
        "event_id": event_id,
        "event_type": event_type,
        "agent_id": "agent-1",
        "session_id": "sess-1",
        "step_index": step_index,
        "failed": failed,
        "error_message": error_message,
        "input_snippet": input_snippet,
        "output_snippet": output_snippet,
    }


class _FakeStore:
    def __init__(self, events: list[dict]) -> None:
        self._events = events

    async def get_events_for_session(self, session_id: str, agent_id: str) -> list[dict]:
        return self._events


class _ErrorStore:
    async def get_events_for_session(self, session_id: str, agent_id: str) -> list[dict]:
        raise RuntimeError("store exploded")


# -----------------------------------------------------------------------
# trace_session tests
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_session_empty_events_returns_empty_graph():
    tracer = CausalTracer(_FakeStore([]))
    graph = await tracer.trace_session("sess-1", "agent-1")
    assert isinstance(graph, CausalGraph)
    assert graph.nodes == {}
    assert graph.edges == []


@pytest.mark.asyncio
async def test_build_sequential_edges_two_events():
    events = [
        _make_event("e1", 0),
        _make_event("e2", 1),
    ]
    tracer = CausalTracer(_FakeStore(events))
    graph = await tracer.trace_session("sess-1", "agent-1")
    seq_edges = [e for e in graph.edges if e.edge_type == EdgeType.SEQUENTIAL]
    assert len(seq_edges) == 1
    assert seq_edges[0].source_event_id == "e1"
    assert seq_edges[0].target_event_id == "e2"


@pytest.mark.asyncio
async def test_build_sequential_edges_preserves_step_order():
    # Events provided out of order — must sort by step_index.
    events = [
        _make_event("e3", 2),
        _make_event("e1", 0),
        _make_event("e2", 1),
    ]
    tracer = CausalTracer(_FakeStore(events))
    graph = await tracer.trace_session("sess-1", "agent-1")
    seq_edges = sorted(
        [e for e in graph.edges if e.edge_type == EdgeType.SEQUENTIAL],
        key=lambda e: e.source_event_id,
    )
    assert seq_edges[0].source_event_id == "e1"
    assert seq_edges[0].target_event_id == "e2"
    assert seq_edges[1].source_event_id == "e2"
    assert seq_edges[1].target_event_id == "e3"


@pytest.mark.asyncio
async def test_build_data_dependency_edge_on_shared_content():
    shared = "x" * _DATA_DEP_MIN_OVERLAP_CHARS
    events = [
        _make_event("e1", 0, output_snippet=shared + " extra"),
        _make_event("e2", 1, input_snippet=shared + " other stuff"),
    ]
    tracer = CausalTracer(_FakeStore(events))
    graph = await tracer.trace_session("sess-1", "agent-1")
    dep_edges = [e for e in graph.edges if e.edge_type == EdgeType.DATA_DEPENDENCY]
    assert len(dep_edges) == 1
    assert dep_edges[0].source_event_id == "e1"
    assert dep_edges[0].target_event_id == "e2"


@pytest.mark.asyncio
async def test_build_data_dependency_no_edge_when_short_overlap():
    short = "x" * (_DATA_DEP_MIN_OVERLAP_CHARS - 1)
    events = [
        _make_event("e1", 0, output_snippet=short),
        _make_event("e2", 1, input_snippet=short),
    ]
    tracer = CausalTracer(_FakeStore(events))
    graph = await tracer.trace_session("sess-1", "agent-1")
    dep_edges = [e for e in graph.edges if e.edge_type == EdgeType.DATA_DEPENDENCY]
    assert len(dep_edges) == 0


@pytest.mark.asyncio
async def test_build_failure_cascade_edge_within_three_steps():
    events = [
        _make_event("e1", 0, error_message="timeout"),
        _make_event("e2", 1, failed=False),
        _make_event("e3", 2, failed=True),
    ]
    tracer = CausalTracer(_FakeStore(events))
    graph = await tracer.trace_session("sess-1", "agent-1")
    fail_edges = [e for e in graph.edges if e.edge_type == EdgeType.FAILURE_CASCADE]
    target_ids = {e.target_event_id for e in fail_edges}
    assert "e3" in target_ids


@pytest.mark.asyncio
async def test_build_failure_cascade_no_edge_beyond_three_steps():
    events = [
        _make_event("e1", 0, error_message="timeout"),
        _make_event("e2", 1),
        _make_event("e3", 2),
        _make_event("e4", 3),
        _make_event("e5", 4, failed=True),  # 4 steps away — outside window
    ]
    tracer = CausalTracer(_FakeStore(events))
    graph = await tracer.trace_session("sess-1", "agent-1")
    fail_edges = [e for e in graph.edges if e.edge_type == EdgeType.FAILURE_CASCADE]
    target_ids = {e.target_event_id for e in fail_edges}
    assert "e5" not in target_ids


# -----------------------------------------------------------------------
# backward BFS tests
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backward_bfs_finds_ancestors():
    events = [
        _make_event("e1", 0),
        _make_event("e2", 1),
        _make_event("e3", 2, failed=True),
    ]
    tracer = CausalTracer(_FakeStore(events))
    graph = await tracer.trace_session("sess-1", "agent-1")
    ancestors = tracer._backward_bfs(graph, "e3")
    assert "e1" in ancestors
    assert "e2" in ancestors
    assert "e3" not in ancestors


@pytest.mark.asyncio
async def test_backward_bfs_stops_at_max_depth():
    # Build a chain of _MAX_BACKWARD_DEPTH + 3 events
    n = _MAX_BACKWARD_DEPTH + 3
    events = [_make_event(f"e{i}", i) for i in range(n)]
    tracer = CausalTracer(_FakeStore(events))
    graph = await tracer.trace_session("sess-1", "agent-1")
    failure_id = f"e{n - 1}"
    ancestors = tracer._backward_bfs(graph, failure_id)
    # Should not exceed _MAX_BACKWARD_DEPTH ancestors
    assert len(ancestors) <= _MAX_BACKWARD_DEPTH


# -----------------------------------------------------------------------
# ranking tests
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rank_candidates_earlier_step_higher_positional_score():
    events = [
        _make_event("e1", 0),
        _make_event("e2", 5),
        _make_event("e3", 9, failed=True),
    ]
    tracer = CausalTracer(_FakeStore(events))
    graph = await tracer.trace_session("sess-1", "agent-1")
    ancestors = tracer._backward_bfs(graph, "e3")
    candidates = tracer._rank_candidates(graph, "e3", ancestors)
    scores = {c.event_id: c.positional_score for c in candidates}
    assert scores["e1"] > scores["e2"]


@pytest.mark.asyncio
async def test_rank_candidates_fewer_downstream_higher_structural_score():
    # e1 → e2 → e3(fail). e1 has 1 downstream; e2 also has 1.
    # But a candidate with 0 downstream should rank highest structurally.
    events = [
        _make_event("e1", 0),
        _make_event("e2", 1),
        _make_event("e3", 2, failed=True),
    ]
    tracer = CausalTracer(_FakeStore(events))
    graph = await tracer.trace_session("sess-1", "agent-1")
    ancestors = tracer._backward_bfs(graph, "e3")
    candidates = tracer._rank_candidates(graph, "e3", ancestors)
    # All structural scores should be between 0 and 1
    for c in candidates:
        assert 0.0 <= c.structural_score <= 1.0
        assert 0.0 <= c.composite_score <= 1.0


# -----------------------------------------------------------------------
# diagnose tests
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diagnose_returns_sorted_candidates():
    events = [
        _make_event("e1", 0, error_message="bad"),
        _make_event("e2", 1),
        _make_event("e3", 2, failed=True),
    ]
    tracer = CausalTracer(_FakeStore(events))
    diagnosis = await tracer.diagnose("sess-1", "agent-1", failure_event_id="e3")
    assert isinstance(diagnosis, CausalDiagnosis)
    scores = [c.composite_score for c in diagnosis.root_causes]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_diagnose_empty_when_failure_event_not_in_graph():
    events = [_make_event("e1", 0)]
    tracer = CausalTracer(_FakeStore(events))
    diagnosis = await tracer.diagnose("sess-1", "agent-1", failure_event_id="nonexistent")
    # BFS from unknown node yields no ancestors
    assert diagnosis.root_causes == []


@pytest.mark.asyncio
async def test_diagnose_never_raises():
    tracer = CausalTracer(_ErrorStore())
    diagnosis = await tracer.diagnose("sess-1", "agent-1", failure_event_id="e1")
    assert isinstance(diagnosis, CausalDiagnosis)
    assert diagnosis.root_causes == []


# -----------------------------------------------------------------------
# find_path tests
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_path_direct_connection():
    events = [_make_event("e1", 0), _make_event("e2", 1)]
    tracer = CausalTracer(_FakeStore(events))
    graph = await tracer.trace_session("sess-1", "agent-1")
    path = tracer._find_path(graph, "e1", "e2")
    assert path == ["e1", "e2"]


@pytest.mark.asyncio
async def test_find_path_returns_empty_when_no_path():
    events = [_make_event("e1", 0), _make_event("e2", 1), _make_event("e3", 2)]
    tracer = CausalTracer(_FakeStore(events))
    graph = await tracer.trace_session("sess-1", "agent-1")
    # No path from e3 to e1 (edges go forward only)
    path = tracer._find_path(graph, "e3", "e1")
    assert path == []
