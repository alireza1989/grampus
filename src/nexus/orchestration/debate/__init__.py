"""Multi-agent debate subsystem for Nexus orchestration."""

from nexus.orchestration.debate.orchestrator import DebateOrchestrator
from nexus.orchestration.debate.types import (
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
