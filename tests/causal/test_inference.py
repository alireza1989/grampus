"""Tests for SimpleCausalInference — backdoor adjustment over simple DAGs."""

from __future__ import annotations

from nexus.causal.inference import SimpleCausalInference
from nexus.causal.types import InterventionQuery, WorldModelGraph


def _make_graph(
    variables: dict[str, str],
    adjacency: dict[str, list[str]],
) -> WorldModelGraph:
    return WorldModelGraph(agent_id="agent-1", variables=variables, adjacency=adjacency)


# -----------------------------------------------------------------------
# DAG validation
# -----------------------------------------------------------------------


def test_is_dag_true_on_simple_chain():
    g = _make_graph({"a": "A", "b": "B", "c": "C"}, {"a": ["b"], "b": ["c"]})
    assert SimpleCausalInference(g).is_dag() is True


def test_is_dag_false_on_cycle():
    g = _make_graph({"a": "A", "b": "B"}, {"a": ["b"], "b": ["a"]})
    assert SimpleCausalInference(g).is_dag() is False


def test_topological_sort_simple_chain():
    g = _make_graph({"a": "A", "b": "B", "c": "C"}, {"a": ["b"], "b": ["c"]})
    order = SimpleCausalInference(g).topological_sort()
    assert order.index("a") < order.index("b") < order.index("c")


def test_topological_sort_returns_empty_on_cycle():
    g = _make_graph({"a": "A", "b": "B"}, {"a": ["b"], "b": ["a"]})
    assert SimpleCausalInference(g).topological_sort() == []


# -----------------------------------------------------------------------
# Path finding
# -----------------------------------------------------------------------


def test_find_causal_paths_direct():
    g = _make_graph({"a": "A", "b": "B"}, {"a": ["b"]})
    paths = SimpleCausalInference(g).find_causal_paths("a", "b")
    assert ["a", "b"] in paths


def test_find_causal_paths_multi_hop():
    g = _make_graph(
        {"a": "A", "b": "B", "c": "C"},
        {"a": ["b"], "b": ["c"]},
    )
    paths = SimpleCausalInference(g).find_causal_paths("a", "c")
    assert any(p == ["a", "b", "c"] for p in paths)


def test_find_causal_paths_empty_when_no_path():
    g = _make_graph({"a": "A", "b": "B"}, {"b": ["a"]})
    paths = SimpleCausalInference(g).find_causal_paths("a", "b")
    assert paths == []


# -----------------------------------------------------------------------
# Backdoor adjustment set
# -----------------------------------------------------------------------


def test_find_backdoor_adjustment_set_no_confounders():
    # a → b: no parents of a, so no backdoor paths
    g = _make_graph({"a": "A", "b": "B"}, {"a": ["b"]})
    result = SimpleCausalInference(g).find_backdoor_adjustment_set("a", "b")
    assert result == []


def test_find_backdoor_adjustment_set_with_parent_confounder():
    # c → a, c → b: c is a confounder; adjustment set should include c
    g = _make_graph(
        {"a": "A", "b": "B", "c": "C"},
        {"c": ["a", "b"], "a": ["b"]},
    )
    result = SimpleCausalInference(g).find_backdoor_adjustment_set("a", "b")
    # c is a parent of a (backdoor path a ← c → b) and is not a descendant of a
    assert result is not None
    assert "c" in result


# -----------------------------------------------------------------------
# Intervention queries
# -----------------------------------------------------------------------


def test_intervene_direct_path_returns_result():
    g = _make_graph({"a": "A", "b": "B"}, {"a": ["b"]})
    q = InterventionQuery(
        natural_language="What if a?",
        target_variable="a",
        outcome_variable="b",
        intervention_value="enabled",
    )
    result = SimpleCausalInference(g).intervene(q)
    assert result.is_identifiable is True
    assert result.causal_path == ["a", "b"]
    assert result.confidence > 0.0


def test_intervene_no_path_returns_not_identifiable():
    g = _make_graph({"a": "A", "b": "B"}, {})
    q = InterventionQuery(
        natural_language="What if a?",
        target_variable="a",
        outcome_variable="b",
        intervention_value="1",
    )
    result = SimpleCausalInference(g).intervene(q)
    assert result.is_identifiable is False
    assert result.causal_path == []


def test_intervene_confidence_decreases_with_hop_count():
    # Longer path → lower confidence (1 / sqrt(len(path)))
    g2 = _make_graph({"a": "A", "b": "B"}, {"a": ["b"]})
    g3 = _make_graph({"a": "A", "b": "B", "c": "C"}, {"a": ["b"], "b": ["c"]})
    q2 = InterventionQuery(
        natural_language="q", target_variable="a", outcome_variable="b", intervention_value="x"
    )
    q3 = InterventionQuery(
        natural_language="q", target_variable="a", outcome_variable="c", intervention_value="x"
    )
    r2 = SimpleCausalInference(g2).intervene(q2)
    r3 = SimpleCausalInference(g3).intervene(q3)
    assert r2.confidence > r3.confidence


def test_intervene_never_raises():
    # Pathological graph — cycle — intervene should return gracefully
    g = _make_graph({"a": "A", "b": "B"}, {"a": ["b"], "b": ["a"]})
    q = InterventionQuery(
        natural_language="?",
        target_variable="a",
        outcome_variable="b",
        intervention_value="x",
    )
    result = SimpleCausalInference(g).intervene(q)
    assert result is not None
