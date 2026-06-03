"""Observability layer: OTEL tracing, Prometheus metrics, behavior monitoring, event log."""

from nexus.observability.alerts import (
    AlertEvaluator,
    AlertEvent,
    AlertRule,
    AlertSeverity,
    AlertState,
    ThresholdType,
)
from nexus.observability.behavior import AgentBehaviorProfile, Anomaly, AnomalyType, BehaviorMonitor
from nexus.observability.events import AgentEvent, EventLog, EventType
from nexus.observability.metrics import MetricsSnapshot, NexusMetrics
from nexus.observability.notification import (
    LogChannel,
    NotificationDispatcher,
    SlackChannel,
    SmtpChannel,
    WebhookChannel,
)
from nexus.observability.phoenix import PhoenixConfig, configure_phoenix_tracer, phoenix_tracer
from nexus.observability.tracer import NexusTracer, SpanKind

__all__ = [
    "AlertEvaluator",
    "AlertEvent",
    "AlertRule",
    "AlertSeverity",
    "AlertState",
    "ThresholdType",
    "Anomaly",
    "AnomalyType",
    "AgentBehaviorProfile",
    "BehaviorMonitor",
    "AgentEvent",
    "EventLog",
    "EventType",
    "LogChannel",
    "MetricsSnapshot",
    "NexusMetrics",
    "NotificationDispatcher",
    "NexusTracer",
    "SlackChannel",
    "SmtpChannel",
    "SpanKind",
    "WebhookChannel",
    "PhoenixConfig",
    "configure_phoenix_tracer",
    "phoenix_tracer",
]
