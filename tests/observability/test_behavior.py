"""Tests for BehaviorMonitor rolling-window anomaly detection."""

from __future__ import annotations

import pytest

from nexus.observability.behavior import AnomalyType, BehaviorMonitor


def _fill_window(monitor: BehaviorMonitor, n: int = 20) -> None:
    """Record n baseline turns with stable, low-cost behaviour."""
    for _ in range(n):
        monitor.record_turn(cost_usd=0.01, tool_names=["search", "calc"], error_count=0)


class TestBehaviorMonitorBaseline:
    def test_no_anomalies_before_window_full(self) -> None:
        m = BehaviorMonitor(agent_id="a1", cost_spike_threshold=3.0)
        # Inject a huge cost before the window fills — should still be empty
        for _ in range(19):
            anomalies = m.record_turn(cost_usd=100.0, tool_names=[], error_count=10)
            assert anomalies == []

    def test_profile_updates_after_each_turn(self) -> None:
        m = BehaviorMonitor(agent_id="a1")
        m.record_turn(cost_usd=0.5, tool_names=["t1"], error_count=1)
        assert m.profile().turn_count == 1

    def test_top_tools_reflects_most_used(self) -> None:
        m = BehaviorMonitor(agent_id="a1")
        _fill_window(m)  # uses ["search", "calc"] 20×
        top = m.profile().top_tools
        assert "search" in top
        assert "calc" in top

    def test_avg_cost_computed_correctly(self) -> None:
        m = BehaviorMonitor(agent_id="a1", cost_spike_threshold=3.0)
        for _ in range(20):
            m.record_turn(cost_usd=0.10, tool_names=[], error_count=0)
        assert m.profile().avg_cost_per_turn == pytest.approx(0.10)

    def test_avg_errors_computed_correctly(self) -> None:
        m = BehaviorMonitor(agent_id="a1")
        for _ in range(20):
            m.record_turn(cost_usd=0.01, tool_names=[], error_count=2)
        assert m.profile().avg_errors_per_turn == pytest.approx(2.0)


class TestBehaviorMonitorAnomalies:
    def test_cost_spike_detected_above_threshold(self) -> None:
        m = BehaviorMonitor(agent_id="a1", cost_spike_threshold=3.0)
        _fill_window(m)  # avg_cost ≈ 0.01
        anomalies = m.record_turn(cost_usd=1.0, tool_names=["search"], error_count=0)
        types = [a.anomaly_type for a in anomalies]
        assert AnomalyType.COST_SPIKE in types

    def test_cost_spike_not_detected_below_threshold(self) -> None:
        m = BehaviorMonitor(agent_id="a1", cost_spike_threshold=3.0)
        _fill_window(m)
        # cost = 0.02, avg = 0.01 → ratio = 2× < 3× threshold
        anomalies = m.record_turn(cost_usd=0.02, tool_names=["search"], error_count=0)
        types = [a.anomaly_type for a in anomalies]
        assert AnomalyType.COST_SPIKE not in types

    def test_error_spike_detected(self) -> None:
        m = BehaviorMonitor(agent_id="a1", error_spike_threshold=5.0)
        for _ in range(20):
            m.record_turn(cost_usd=0.01, tool_names=["search"], error_count=1)
        # avg_errors = 1.0; spike threshold = 5.0; this turn = 10 errors → ratio = 10
        anomalies = m.record_turn(cost_usd=0.01, tool_names=["search"], error_count=10)
        types = [a.anomaly_type for a in anomalies]
        assert AnomalyType.ERROR_RATE_SPIKE in types

    def test_tool_usage_shift_detected_when_new_tools_dominate(self) -> None:
        m = BehaviorMonitor(agent_id="a1", tool_shift_threshold=0.5)
        _fill_window(m)  # top_tools = ["search", "calc"]
        # All tools are new: ["new1", "new2", "new3"] → 100% new > 50%
        anomalies = m.record_turn(cost_usd=0.01, tool_names=["new1", "new2", "new3"], error_count=0)
        types = [a.anomaly_type for a in anomalies]
        assert AnomalyType.TOOL_USAGE_SHIFT in types

    def test_tool_usage_shift_not_detected_when_familiar_tools(self) -> None:
        m = BehaviorMonitor(agent_id="a1", tool_shift_threshold=0.5)
        _fill_window(m)  # top_tools = ["search", "calc"]
        # Mostly familiar tools
        anomalies = m.record_turn(
            cost_usd=0.01, tool_names=["search", "calc", "search"], error_count=0
        )
        types = [a.anomaly_type for a in anomalies]
        assert AnomalyType.TOOL_USAGE_SHIFT not in types

    def test_anomaly_severity_capped_at_1_0(self) -> None:
        m = BehaviorMonitor(agent_id="a1", cost_spike_threshold=3.0)
        _fill_window(m)
        # Extreme cost spike: 1000× the average
        anomalies = m.record_turn(cost_usd=10.0, tool_names=[], error_count=0)
        for a in anomalies:
            assert a.severity <= 1.0

    def test_anomalies_method_returns_all_detected(self) -> None:
        m = BehaviorMonitor(agent_id="a1", cost_spike_threshold=3.0)
        _fill_window(m)
        m.record_turn(cost_usd=1.0, tool_names=[], error_count=0)
        m.record_turn(cost_usd=1.0, tool_names=[], error_count=0)
        assert len(m.anomalies()) >= 2

    def test_reset_clears_all_state(self) -> None:
        m = BehaviorMonitor(agent_id="a1", cost_spike_threshold=3.0)
        _fill_window(m)
        m.record_turn(cost_usd=5.0, tool_names=[], error_count=0)
        m.reset()
        assert m.anomalies() == []
        assert m.profile().turn_count == 0
        # After reset, no anomalies until window fills again
        anomalies = m.record_turn(cost_usd=999.0, tool_names=[], error_count=0)
        assert anomalies == []
