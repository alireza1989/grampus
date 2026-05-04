"""Tests for NexusTracer OTEL span wrappers."""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nexus.observability.tracer import NexusTracer, SpanKind


def _make_tracer_with_exporter() -> tuple[NexusTracer, InMemorySpanExporter]:
    """Return a NexusTracer wired to an in-memory exporter for assertions."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = NexusTracer(service_name="test-service", agent_id="agent-1")
    tracer._tracer = provider.get_tracer("test-service")
    return tracer, exporter


class TestNexusTracerSpans:
    def test_noop_provider_used_when_no_endpoint(self) -> None:
        tracer = NexusTracer(service_name="svc", agent_id="a1")
        # Should not raise; span methods must work with NoOp provider
        with tracer.agent_run(session_id="s1"):
            pass

    def test_spans_are_context_managers(self) -> None:
        tracer = NexusTracer(agent_id="a1")
        with tracer.agent_run(session_id="s1") as span:
            assert span is not None

    def test_agent_run_span_created(self) -> None:
        tracer, exporter = _make_tracer_with_exporter()
        with tracer.agent_run(session_id="sess-42"):
            pass
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == SpanKind.AGENT_RUN

    def test_llm_call_span_has_model_attribute(self) -> None:
        tracer, exporter = _make_tracer_with_exporter()
        with tracer.llm_call(
            model="claude-3-5-sonnet", input_tokens=10, output_tokens=20, cost_usd=0.01
        ):
            pass
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["model"] == "claude-3-5-sonnet"

    def test_llm_call_span_has_token_attributes(self) -> None:
        tracer, exporter = _make_tracer_with_exporter()
        with tracer.llm_call(model="gpt-4", input_tokens=100, output_tokens=50, cost_usd=0.005):
            pass
        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs["input_tokens"] == 100
        assert attrs["output_tokens"] == 50
        assert attrs["cost_usd"] == pytest.approx(0.005)

    def test_tool_call_span_has_tool_name(self) -> None:
        tracer, exporter = _make_tracer_with_exporter()
        with tracer.tool_call(tool_name="web_search", success=True, duration_ms=42.0):
            pass
        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs["tool_name"] == "web_search"
        assert attrs["success"] is True
        assert attrs["duration_ms"] == pytest.approx(42.0)

    def test_memory_read_span_has_memory_type(self) -> None:
        tracer, exporter = _make_tracer_with_exporter()
        with tracer.memory_read(memory_type="episodic", records_returned=5):
            pass
        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs["memory_type"] == "episodic"
        assert attrs["records_returned"] == 5

    def test_memory_write_span_has_content_length(self) -> None:
        tracer, exporter = _make_tracer_with_exporter()
        with tracer.memory_write(memory_type="semantic", content_length=200):
            pass
        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs["memory_type"] == "semantic"
        assert attrs["content_length"] == 200

    def test_decision_span_has_node_and_outcome(self) -> None:
        tracer, exporter = _make_tracer_with_exporter()
        with tracer.decision(node_name="router", outcome="branch_a"):
            pass
        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs["node_name"] == "router"
        assert attrs["outcome"] == "branch_a"

    def test_span_records_exception_on_error(self) -> None:
        tracer, exporter = _make_tracer_with_exporter()
        with pytest.raises(ValueError, match="boom"), tracer.agent_run(session_id="s1"):
            raise ValueError("boom")
        span = exporter.get_finished_spans()[0]
        from opentelemetry.trace import StatusCode

        assert span.status.status_code == StatusCode.ERROR
        assert len(span.events) >= 1  # exception event recorded
