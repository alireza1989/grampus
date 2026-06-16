"""Tests for GrampusMetrics in-process Prometheus-compatible metrics collector."""

from __future__ import annotations

import re

import pytest

from grampus.observability.metrics import GrampusMetrics, MetricsSnapshot


class TestGrampusMetricsCounters:
    def setup_method(self) -> None:
        self.m = GrampusMetrics(agent_id="agent-1")

    def test_record_llm_call_increments_total_tokens(self) -> None:
        self.m.record_llm_call(
            model="claude", input_tokens=100, output_tokens=50, cost_usd=0.01, latency_ms=200.0
        )
        assert self.m.snapshot().total_tokens == 150

    def test_record_llm_call_increments_total_cost(self) -> None:
        self.m.record_llm_call(
            model="claude", input_tokens=100, output_tokens=50, cost_usd=0.05, latency_ms=100.0
        )
        assert self.m.snapshot().total_cost_usd == pytest.approx(0.05)

    def test_record_llm_call_increments_call_count(self) -> None:
        self.m.record_llm_call(
            model="claude", input_tokens=10, output_tokens=10, cost_usd=0.001, latency_ms=50.0
        )
        self.m.record_llm_call(
            model="claude", input_tokens=10, output_tokens=10, cost_usd=0.001, latency_ms=50.0
        )
        assert self.m.snapshot().llm_call_count == 2

    def test_record_llm_call_tracks_per_model_tokens(self) -> None:
        self.m.record_llm_call(
            model="gpt-4", input_tokens=200, output_tokens=100, cost_usd=0.01, latency_ms=300.0
        )
        self.m.record_llm_call(
            model="claude", input_tokens=50, output_tokens=25, cost_usd=0.005, latency_ms=100.0
        )
        snap = self.m.snapshot()
        assert snap.per_model_tokens["gpt-4"] == 300
        assert snap.per_model_tokens["claude"] == 75

    def test_record_tool_call_increments_tool_calls(self) -> None:
        self.m.record_tool_call(tool_name="search", success=True, latency_ms=120.0)
        self.m.record_tool_call(tool_name="calc", success=False, latency_ms=30.0)
        assert self.m.snapshot().total_tool_calls == 2

    def test_record_error_increments_error_count(self) -> None:
        self.m.record_error(error_type="ToolError")
        self.m.record_error(error_type="ModelError")
        assert self.m.snapshot().total_errors == 2

    def test_set_active_agents_updates_gauge(self) -> None:
        self.m.set_active_agents(5)
        assert self.m.snapshot().active_agents == 5
        self.m.set_active_agents(2)
        assert self.m.snapshot().active_agents == 2

    def test_reset_clears_all_counters(self) -> None:
        self.m.record_llm_call(
            model="claude", input_tokens=100, output_tokens=50, cost_usd=0.01, latency_ms=100.0
        )
        self.m.record_error(error_type="ToolError")
        self.m.reset()
        snap = self.m.snapshot()
        assert snap.total_tokens == 0
        assert snap.total_cost_usd == 0.0
        assert snap.total_errors == 0
        assert snap.llm_call_count == 0


class TestGrampusMetricsSnapshot:
    def test_snapshot_zero_initially(self) -> None:
        m = GrampusMetrics(agent_id="a")
        snap = m.snapshot()
        assert snap == MetricsSnapshot()

    def test_snapshot_reflects_recorded_data(self) -> None:
        m = GrampusMetrics(agent_id="a")
        m.record_llm_call(
            model="m1", input_tokens=10, output_tokens=5, cost_usd=0.002, latency_ms=80.0
        )
        m.record_tool_call(tool_name="t1", success=True, latency_ms=10.0)
        snap = m.snapshot()
        assert snap.total_tokens == 15
        assert snap.total_tool_calls == 1

    def test_per_agent_cost_tracked(self) -> None:
        m = GrampusMetrics(agent_id="agent-99")
        m.record_llm_call(
            model="m", input_tokens=100, output_tokens=50, cost_usd=0.03, latency_ms=200.0
        )
        snap = m.snapshot()
        assert snap.per_agent_cost["agent-99"] == pytest.approx(0.03)


class TestGrampusMetricsPrometheus:
    def setup_method(self) -> None:
        self.m = GrampusMetrics(agent_id="agent-x")
        self.m.record_llm_call(
            model="claude", input_tokens=100, output_tokens=50, cost_usd=0.01, latency_ms=150.0
        )
        self.m.record_tool_call(tool_name="search", success=True, latency_ms=80.0)
        self.m.record_error(error_type="ToolError")
        self.text = self.m.to_prometheus_text()

    def test_to_prometheus_text_contains_metric_names(self) -> None:
        assert "grampus_total_tokens" in self.text
        assert "grampus_total_cost_usd" in self.text
        assert "grampus_total_tool_calls" in self.text

    def test_to_prometheus_text_contains_agent_id_label(self) -> None:
        assert 'agent_id="agent-x"' in self.text

    def test_to_prometheus_text_contains_counter_type(self) -> None:
        assert "# TYPE grampus_total_tokens counter" in self.text

    def test_to_prometheus_text_contains_histogram_buckets(self) -> None:
        assert "_bucket{" in self.text
        assert 'le="10"' in self.text
        assert 'le="+Inf"' in self.text

    def test_prometheus_text_is_valid_format(self) -> None:
        for line in self.text.strip().split("\n"):
            if not line:
                continue
            # Lines are: # HELP ..., # TYPE ..., or metric{labels} value
            assert (
                line.startswith("#")
                or re.match(r"^\w[\w_]*\{", line)
                or re.match(r"^\w[\w_]* ", line)
            ), f"Unexpected line format: {line!r}"
