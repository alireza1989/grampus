"""Integration tests for EventLog, GrampusMetrics, and BehaviorMonitor."""

from __future__ import annotations

import pytest

from grampus.observability.events import EventLog, EventType
from tests.integration.conftest import FakeStateStore


@pytest.mark.integration
class TestObservabilityIntegration:
    async def test_event_log_appends_and_replays_with_fake_store(
        self, fake_state_store: FakeStateStore
    ) -> None:
        log = EventLog(agent_id="obs-agent", session_id="s1", state_store=fake_state_store)
        await log.append(EventType.AGENT_STARTED, {"input": "test"})
        await log.append(EventType.LLM_CALLED, {"model": "mock"})
        await log.append(EventType.AGENT_COMPLETED, {"output": "done"})

        events = await log.replay()
        assert len(events) == 3
        assert events[0].event_type == EventType.AGENT_STARTED
        assert events[2].event_type == EventType.AGENT_COMPLETED

    async def test_event_log_in_memory_replay(self) -> None:
        log = EventLog(agent_id="obs-agent", session_id="s2")
        await log.append(EventType.LLM_CALLED)
        await log.append(EventType.TOOL_CALLED)
        events = await log.replay()
        assert len(events) == 2

    async def test_event_log_sequence_numbers_monotonic(
        self, fake_state_store: FakeStateStore
    ) -> None:
        log = EventLog(agent_id="obs-agent", session_id="s3", state_store=fake_state_store)
        for et in [EventType.AGENT_STARTED, EventType.LLM_CALLED, EventType.AGENT_COMPLETED]:
            await log.append(et)
        events = await log.replay()
        seqs = [e.sequence_number for e in events]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)

    async def test_event_log_replay_since_filters_correctly(
        self, fake_state_store: FakeStateStore
    ) -> None:
        log = EventLog(agent_id="obs-agent", session_id="s4", state_store=fake_state_store)
        for et in [EventType.AGENT_STARTED, EventType.LLM_CALLED, EventType.TOOL_CALLED]:
            await log.append(et)

        recent = await log.replay_since(1)
        assert all(e.sequence_number >= 1 for e in recent)
        assert len(recent) == 2

    async def test_event_count_is_accurate(self) -> None:
        log = EventLog(agent_id="obs-agent", session_id="s5")
        assert log.event_count() == 0
        await log.append(EventType.LLM_CALLED)
        await log.append(EventType.TOOL_CALLED)
        assert log.event_count() == 2

    async def test_metrics_accumulate_across_llm_calls(self) -> None:
        from grampus.observability.metrics import GrampusMetrics

        metrics = GrampusMetrics(agent_id="met-agent")
        metrics.record_llm_call(
            model="claude", input_tokens=100, output_tokens=50, cost_usd=0.01, latency_ms=200.0
        )
        metrics.record_llm_call(
            model="claude", input_tokens=80, output_tokens=40, cost_usd=0.008, latency_ms=150.0
        )
        snap = metrics.snapshot()
        assert snap.total_tokens == 270
        assert snap.total_cost_usd == pytest.approx(0.018)
        assert snap.llm_call_count == 2

    async def test_prometheus_text_valid_after_recording(self) -> None:
        from grampus.observability.metrics import GrampusMetrics

        metrics = GrampusMetrics(agent_id="prom-agent")
        metrics.record_llm_call(
            model="mock", input_tokens=10, output_tokens=5, cost_usd=0.001, latency_ms=50.0
        )
        metrics.record_tool_call(tool_name="echo", success=True, latency_ms=20.0)
        prom_text = metrics.to_prometheus_text()
        assert isinstance(prom_text, str)
        assert len(prom_text) > 0

    async def test_behavior_monitor_detects_cost_spike_after_window(self) -> None:
        from grampus.observability.behavior import AnomalyType, BehaviorMonitor

        monitor = BehaviorMonitor(
            agent_id="beh-agent",
            cost_spike_threshold=3.0,
        )
        for _ in range(20):
            monitor.record_turn(cost_usd=0.01, tool_names=["echo"], error_count=0)

        anomalies = monitor.record_turn(cost_usd=1.0, tool_names=["echo"], error_count=0)
        found_cost_spike = any(a.anomaly_type == AnomalyType.COST_SPIKE for a in anomalies)
        assert found_cost_spike, "Expected cost spike anomaly not detected"

    async def test_behavior_monitor_no_anomaly_before_window_full(self) -> None:
        from grampus.observability.behavior import BehaviorMonitor

        monitor = BehaviorMonitor(agent_id="beh2-agent")
        for _ in range(5):
            anomalies = monitor.record_turn(cost_usd=0.01, tool_names=["echo"], error_count=0)
        assert anomalies == []

    async def test_tracer_spans_created_without_error(self) -> None:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        from grampus.observability.tracer import GrampusTracer

        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        tracer = GrampusTracer(agent_id="trace-agent", session_id="ts1")

        with tracer.agent_run_span() as span:
            tracer.record_llm_call(
                span, model="mock", input_tokens=5, output_tokens=5, cost_usd=0.0, latency_ms=10.0
            )
