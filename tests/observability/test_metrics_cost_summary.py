"""Tests for NexusMetrics.get_cost_summary()."""

from __future__ import annotations

import pytest

from nexus.observability.metrics import CostSummary, NexusMetrics


class TestGetCostSummaryEmpty:
    def test_returns_cost_summary_type(self) -> None:
        m = NexusMetrics(agent_id="agent-1")
        result = m.get_cost_summary()
        assert isinstance(result, CostSummary)

    def test_empty_metrics_zero_totals(self) -> None:
        m = NexusMetrics(agent_id="agent-1")
        s = m.get_cost_summary()
        assert s.total_cost_usd == pytest.approx(0.0)
        assert s.total_tokens == 0
        assert s.total_llm_calls == 0

    def test_empty_metrics_empty_tables(self) -> None:
        m = NexusMetrics(agent_id="agent-1")
        s = m.get_cost_summary()
        assert s.by_model == []
        assert s.by_agent == []


class TestGetCostSummaryWithData:
    def test_total_cost_matches_recorded(self) -> None:
        m = NexusMetrics(agent_id="agent-1")
        m.record_llm_call(
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            latency_ms=100.0,
        )
        s = m.get_cost_summary()
        assert s.total_cost_usd == pytest.approx(0.01)

    def test_by_model_sorted_by_cost_desc(self) -> None:
        m = NexusMetrics(agent_id="a")
        m.record_llm_call(
            model="cheap", input_tokens=10, output_tokens=5, cost_usd=0.001, latency_ms=10.0
        )
        m.record_llm_call(
            model="expensive", input_tokens=100, output_tokens=50, cost_usd=0.05, latency_ms=500.0
        )
        s = m.get_cost_summary()
        assert len(s.by_model) == 2
        assert s.by_model[0]["model"] == "expensive"
        assert s.by_model[1]["model"] == "cheap"

    def test_by_model_pct_sums_to_100(self) -> None:
        m = NexusMetrics(agent_id="a")
        m.record_llm_call(
            model="m1", input_tokens=100, output_tokens=0, cost_usd=0.04, latency_ms=10.0
        )
        m.record_llm_call(
            model="m2", input_tokens=100, output_tokens=0, cost_usd=0.01, latency_ms=10.0
        )
        s = m.get_cost_summary()
        total_pct = sum(r["pct"] for r in s.by_model)
        assert total_pct == pytest.approx(100.0)

    def test_by_model_tokens_match_input_plus_output(self) -> None:
        m = NexusMetrics(agent_id="a")
        m.record_llm_call(
            model="m1", input_tokens=100, output_tokens=50, cost_usd=0.01, latency_ms=10.0
        )
        s = m.get_cost_summary()
        row = s.by_model[0]
        assert row["tokens"] == 150

    def test_by_agent_single_entry(self) -> None:
        m = NexusMetrics(agent_id="my-agent")
        m.record_llm_call(
            model="gpt-4", input_tokens=10, output_tokens=5, cost_usd=0.005, latency_ms=50.0
        )
        s = m.get_cost_summary()
        assert len(s.by_agent) == 1
        assert s.by_agent[0]["agent_id"] == "my-agent"
        assert s.by_agent[0]["cost_usd"] == pytest.approx(0.005)
        assert s.by_agent[0]["pct"] == pytest.approx(100.0)

    def test_multiple_calls_to_same_model_accumulate(self) -> None:
        m = NexusMetrics(agent_id="a")
        m.record_llm_call(
            model="gpt-4", input_tokens=10, output_tokens=5, cost_usd=0.01, latency_ms=10.0
        )
        m.record_llm_call(
            model="gpt-4", input_tokens=20, output_tokens=10, cost_usd=0.02, latency_ms=20.0
        )
        s = m.get_cost_summary()
        assert len(s.by_model) == 1
        assert s.by_model[0]["cost_usd"] == pytest.approx(0.03)

    def test_reset_clears_cost_summary(self) -> None:
        m = NexusMetrics(agent_id="a")
        m.record_llm_call(
            model="gpt-4", input_tokens=10, output_tokens=5, cost_usd=0.01, latency_ms=10.0
        )
        m.reset()
        s = m.get_cost_summary()
        assert s.total_cost_usd == pytest.approx(0.0)
        assert s.by_model == []
