"""Observability layer: OTEL tracing, Prometheus metrics, behavior monitoring, event log."""

from nexus.observability.behavior import AgentBehaviorProfile, Anomaly, AnomalyType, BehaviorMonitor
from nexus.observability.events import AgentEvent, EventLog, EventType
from nexus.observability.metrics import MetricsSnapshot, NexusMetrics
from nexus.observability.phoenix import PhoenixConfig, configure_phoenix_tracer, phoenix_tracer
from nexus.observability.tracer import NexusTracer, SpanKind

__all__ = [
    "Anomaly",
    "AnomalyType",
    "AgentBehaviorProfile",
    "BehaviorMonitor",
    "AgentEvent",
    "EventLog",
    "EventType",
    "MetricsSnapshot",
    "NexusMetrics",
    "NexusTracer",
    "SpanKind",
    "PhoenixConfig",
    "configure_phoenix_tracer",
    "phoenix_tracer",
]
