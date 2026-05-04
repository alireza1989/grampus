"""OpenTelemetry tracer for Nexus agent spans."""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from enum import StrEnum
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, StatusCode

from nexus.core.logging import get_logger

logger = get_logger(__name__)

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
except ImportError:
    OTLPSpanExporter = None  # type: ignore[assignment,misc]


class SpanKind(StrEnum):
    """Names for Nexus custom OTEL span types."""

    AGENT_RUN = "agent.run"
    LLM_CALL = "agent.llm_call"
    TOOL_CALL = "agent.tool_call"
    MEMORY_READ = "agent.memory_read"
    MEMORY_WRITE = "agent.memory_write"
    DECISION = "agent.decision"


@contextlib.contextmanager
def _span_context(
    tracer: trace.Tracer, name: str, attrs: dict[str, Any]
) -> Generator[Span, None, None]:
    """Shared span lifecycle: set attributes, catch exceptions, record status."""
    with tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            span.set_attribute(k, v)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            raise


class NexusTracer:
    """Wraps the OpenTelemetry SDK to produce Nexus-specific agent spans.

    All span methods are synchronous context managers:

        with tracer.agent_run(session_id="x") as span:
            span.set_attribute("custom", "value")

    Args:
        service_name: OTEL service name (e.g. "nexus-agent").
        otlp_endpoint: Optional OTLP exporter endpoint (e.g. "http://localhost:4317").
            When None, uses a NoOpTracerProvider — no network calls.
        agent_id: Default agent_id attached to every span.
    """

    def __init__(
        self,
        *,
        service_name: str = "nexus-agent",
        otlp_endpoint: str | None = None,
        agent_id: str = "unknown",
    ) -> None:
        self._agent_id = agent_id
        self._service_name = service_name
        self._tracer = self._build_tracer(service_name, otlp_endpoint)

    def _build_tracer(self, service_name: str, otlp_endpoint: str | None) -> trace.Tracer:
        if otlp_endpoint is None:
            trace.set_tracer_provider(trace.NoOpTracerProvider())
            return trace.get_tracer(service_name)
        return self._build_otlp_tracer(service_name, otlp_endpoint)

    def _build_otlp_tracer(self, service_name: str, endpoint: str) -> trace.Tracer:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        if OTLPSpanExporter is None:
            logger.warning("otlp_exporter_unavailable", endpoint=endpoint)
            return trace.get_tracer(service_name)

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        return provider.get_tracer(service_name)

    def _base_attrs(self) -> dict[str, Any]:
        return {"agent_id": self._agent_id}

    @contextlib.contextmanager
    def agent_run(self, *, session_id: str, **extra_attrs: Any) -> Generator[Span, None, None]:
        """Span for a full agent execution turn.

        Args:
            session_id: Unique identifier for this session.
            **extra_attrs: Additional span attributes.

        Yields:
            The active OTEL Span.
        """
        attrs = {**self._base_attrs(), "session_id": session_id, **extra_attrs}
        with _span_context(self._tracer, SpanKind.AGENT_RUN, attrs) as span:
            yield span

    @contextlib.contextmanager
    def llm_call(
        self,
        *,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        **extra_attrs: Any,
    ) -> Generator[Span, None, None]:
        """Span for one LLM completion call.

        Args:
            model: Model identifier string.
            input_tokens: Number of prompt tokens consumed.
            output_tokens: Number of completion tokens produced.
            cost_usd: Estimated cost in USD.
            **extra_attrs: Additional span attributes.

        Yields:
            The active OTEL Span.
        """
        attrs = {
            **self._base_attrs(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            **extra_attrs,
        }
        with _span_context(self._tracer, SpanKind.LLM_CALL, attrs) as span:
            yield span

    @contextlib.contextmanager
    def tool_call(
        self,
        *,
        tool_name: str,
        success: bool = True,
        duration_ms: float = 0.0,
        **extra_attrs: Any,
    ) -> Generator[Span, None, None]:
        """Span for one tool execution.

        Args:
            tool_name: Registered name of the tool.
            success: Whether the tool call succeeded.
            duration_ms: Wall-clock execution time.
            **extra_attrs: Additional span attributes.

        Yields:
            The active OTEL Span.
        """
        attrs = {
            **self._base_attrs(),
            "tool_name": tool_name,
            "success": success,
            "duration_ms": duration_ms,
            **extra_attrs,
        }
        with _span_context(self._tracer, SpanKind.TOOL_CALL, attrs) as span:
            yield span

    @contextlib.contextmanager
    def memory_read(
        self,
        *,
        memory_type: str,
        records_returned: int = 0,
    ) -> Generator[Span, None, None]:
        """Span for a memory recall operation.

        Args:
            memory_type: One of "working", "episodic", "semantic", "procedural".
            records_returned: Number of records surfaced by the query.

        Yields:
            The active OTEL Span.
        """
        attrs = {
            **self._base_attrs(),
            "memory_type": memory_type,
            "records_returned": records_returned,
        }
        with _span_context(self._tracer, SpanKind.MEMORY_READ, attrs) as span:
            yield span

    @contextlib.contextmanager
    def memory_write(
        self,
        *,
        memory_type: str,
        content_length: int = 0,
    ) -> Generator[Span, None, None]:
        """Span for a memory store operation.

        Args:
            memory_type: Memory layer being written.
            content_length: Byte length of the content stored.

        Yields:
            The active OTEL Span.
        """
        attrs = {**self._base_attrs(), "memory_type": memory_type, "content_length": content_length}
        with _span_context(self._tracer, SpanKind.MEMORY_WRITE, attrs) as span:
            yield span

    @contextlib.contextmanager
    def decision(
        self,
        *,
        node_name: str,
        outcome: str,
    ) -> Generator[Span, None, None]:
        """Span for a graph decision/routing point.

        Args:
            node_name: Name of the graph node making the decision.
            outcome: The selected branch or outcome label.

        Yields:
            The active OTEL Span.
        """
        attrs = {**self._base_attrs(), "node_name": node_name, "outcome": outcome}
        with _span_context(self._tracer, SpanKind.DECISION, attrs) as span:
            yield span
