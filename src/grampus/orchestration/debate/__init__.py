"""Multi-agent debate subsystem for Nexus orchestration."""

from grampus.orchestration.debate.orchestrator import DebateOrchestrator
from grampus.orchestration.debate.types import (
    AggregationStrategy,
    DebateConfig,
    DebaterConfig,
    DebateResult,
    DebateRound,
    DebaterPosition,
    RoutingDecision,
)

__all__ = [
    "AggregationStrategy",
    "DebateConfig",
    "DebateOrchestrator",
    "DebaterConfig",
    "DebateResult",
    "DebaterPosition",
    "DebateRound",
    "RoutingDecision",
]
