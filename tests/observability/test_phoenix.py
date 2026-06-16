"""Tests for Arize Phoenix integration helpers."""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from grampus.observability.phoenix import PhoenixConfig, configure_phoenix_tracer, phoenix_tracer
from grampus.observability.tracer import GrampusTracer


def _make_recording_tracer(grampus_tracer: GrampusTracer) -> InMemorySpanExporter:
    """Wire an in-memory exporter onto an existing GrampusTracer for assertions."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    grampus_tracer._tracer = provider.get_tracer("test-service")
    return exporter


class TestPhoenixConfig:
    def test_phoenix_config_defaults(self) -> None:
        config = PhoenixConfig()
        assert config.endpoint == "http://localhost:6006"
        assert config.otlp_grpc_endpoint == "http://localhost:4317"
        assert config.project_name == "grampus"
        assert config.enabled is True

    def test_phoenix_config_custom_values(self) -> None:
        config = PhoenixConfig(
            endpoint="http://phoenix.example.com:6006",
            otlp_grpc_endpoint="http://phoenix.example.com:4317",
            project_name="my-project",
            enabled=False,
        )
        assert config.endpoint == "http://phoenix.example.com:6006"
        assert config.project_name == "my-project"
        assert config.enabled is False

    def test_phoenix_config_from_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PHOENIX_ENDPOINT", "http://custom-ui:6006")
        monkeypatch.setenv("PHOENIX_OTLP_ENDPOINT", "http://custom-otlp:4317")
        monkeypatch.setenv("PHOENIX_PROJECT_NAME", "env-project")
        # PhoenixConfig reads env vars via its factory path; test the context manager
        with phoenix_tracer() as tracer:
            assert isinstance(tracer, GrampusTracer)


class TestConfigurePhoenixTracer:
    def test_configure_phoenix_tracer_returns_grampus_tracer(self) -> None:
        config = PhoenixConfig(enabled=True)
        tracer = configure_phoenix_tracer(config)
        assert isinstance(tracer, GrampusTracer)

    def test_configure_phoenix_tracer_disabled_returns_noop(self) -> None:
        config = PhoenixConfig(enabled=False)
        tracer = configure_phoenix_tracer(config)
        assert isinstance(tracer, GrampusTracer)
        # disabled tracer has no OTLP endpoint — should work without error
        with tracer.agent_run(session_id="noop-session"):
            pass

    def test_configure_phoenix_tracer_passes_service_name(self) -> None:
        config = PhoenixConfig(enabled=True)
        tracer = configure_phoenix_tracer(config, service_name="my-service")
        assert tracer._service_name == "my-service"

    def test_configure_phoenix_tracer_passes_agent_id(self) -> None:
        config = PhoenixConfig(enabled=True)
        tracer = configure_phoenix_tracer(config, agent_id="agent-42")
        assert tracer._agent_id == "agent-42"

    def test_configure_phoenix_tracer_passes_session_id(self) -> None:
        config = PhoenixConfig(enabled=True)
        tracer = configure_phoenix_tracer(config, session_id="sess-99")
        assert tracer._session_id == "sess-99"


class TestPhoenixTracerContextManager:
    def test_phoenix_tracer_context_manager(self) -> None:
        with phoenix_tracer() as tracer:
            assert isinstance(tracer, GrampusTracer)

    def test_phoenix_tracer_uses_env_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PHOENIX_OTLP_ENDPOINT", "http://custom:4317")
        # Context manager should construct without error using custom endpoint
        with phoenix_tracer() as tracer:
            assert isinstance(tracer, GrampusTracer)

    def test_phoenix_tracer_accepts_explicit_config(self) -> None:
        config = PhoenixConfig(project_name="explicit-project", enabled=True)
        with phoenix_tracer(config=config) as tracer:
            assert isinstance(tracer, GrampusTracer)

    def test_phoenix_tracer_disabled_config(self) -> None:
        config = PhoenixConfig(enabled=False)
        with phoenix_tracer(config=config) as tracer:
            assert isinstance(tracer, GrampusTracer)
            with tracer.agent_run(session_id="disabled"):
                pass


class TestGrampusTracerExtraResourceAttrs:
    def test_grampus_tracer_extra_resource_attrs_accepted(self) -> None:
        tracer = GrampusTracer(extra_resource_attrs={"phoenix.project.name": "test", "env": "ci"})
        assert isinstance(tracer, GrampusTracer)

    def test_grampus_tracer_extra_resource_attrs_none_accepted(self) -> None:
        tracer = GrampusTracer(extra_resource_attrs=None)
        assert isinstance(tracer, GrampusTracer)

    def test_grampus_tracer_extra_resource_attrs_stored(self) -> None:
        tracer = GrampusTracer(extra_resource_attrs={"phoenix.project.name": "stored-test"})
        assert tracer._extra_resource_attrs == {"phoenix.project.name": "stored-test"}

    def test_grampus_tracer_no_extra_attrs_backward_compat(self) -> None:
        # No extra_resource_attrs arg — must still work exactly as before
        tracer = GrampusTracer(service_name="compat-svc", agent_id="a1")
        with tracer.agent_run(session_id="compat"):
            pass


class TestPhoenixTracerSpansStillWork:
    def test_all_span_methods_work_after_configure(self) -> None:
        config = PhoenixConfig(enabled=True)
        tracer = configure_phoenix_tracer(config, agent_id="span-test")
        exporter = _make_recording_tracer(tracer)

        with tracer.agent_run(session_id="s1"):
            pass
        with tracer.llm_call(model="claude-3-5-sonnet", input_tokens=10, output_tokens=5):
            pass
        with tracer.tool_call(tool_name="search", success=True, duration_ms=20.0):
            pass
        with tracer.memory_read(memory_type="episodic", records_returned=3):
            pass
        with tracer.memory_write(memory_type="semantic", content_length=100):
            pass
        with tracer.decision(node_name="router", outcome="branch_a"):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 6

    def test_agent_run_span_method_works(self) -> None:
        config = PhoenixConfig(enabled=False)
        tracer = configure_phoenix_tracer(config, session_id="sess-1")
        with tracer.agent_run_span() as span:
            assert span is not None
