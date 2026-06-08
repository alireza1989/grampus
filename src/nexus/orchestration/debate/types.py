"""Type definitions for the multi-agent debate system."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from nexus.core.types import TokenUsage


class AggregationStrategy(StrEnum):
    """How the final debate answer is determined."""

    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    JUDGE = "judge"


class RoutingDecision(StrEnum):
    """Whether a full debate ran or was bypassed by the router."""

    SINGLE_AGENT = "single_agent"
    DEBATE = "debate"


class DebaterConfig(BaseModel):
    """Configuration for a single debate participant."""

    model_client: Any
    model_id: str
    temperature: float = 0.7
    role_hint: str = ""
    weight: float = 1.0

    model_config = {"arbitrary_types_allowed": True}


class DebateConfig(BaseModel):
    """Full configuration governing a debate run.

    Args:
        debaters: At least two DebaterConfig entries.
        max_rounds: Maximum debate rounds before forced aggregation.
        aggregation: Strategy for picking the winner.
        convergence_threshold: Fraction of debaters that must agree to stop early.
        adaptive_routing: Skip debate when single-agent confidence is high.
        routing_confidence_threshold: Confidence level that triggers routing bypass.
        routing_model_client: Fast model for routing check; defaults to debaters[0].
        routing_model_id: Model ID for routing check.
        judge_config: Required when aggregation=JUDGE.
        cost_budget_usd: Hard spend ceiling; enforced before each round.
        escalate_threshold: If final convergence_score < this, set escalate_to_human=True.
    """

    debaters: list[DebaterConfig] = Field(min_length=2)
    max_rounds: int = 3
    aggregation: AggregationStrategy = AggregationStrategy.WEIGHTED_VOTE
    convergence_threshold: float = 0.8
    adaptive_routing: bool = True
    routing_confidence_threshold: float = 0.85
    routing_model_client: Any | None = None
    routing_model_id: str = ""
    judge_config: DebaterConfig | None = None
    cost_budget_usd: float | None = None
    escalate_threshold: float = 0.5

    model_config = {"arbitrary_types_allowed": True}


class DebaterPosition(BaseModel):
    """One debater's answer and reasoning for a single round."""

    debater_index: int
    model_id: str
    answer: str
    reasoning: str
    confidence: float = 0.5
    changed_from_previous: bool = False
    change_justification: str = ""
    token_usage: TokenUsage | None = None


class DebateRound(BaseModel):
    """All positions produced in one round of debate."""

    round_number: int
    positions: list[DebaterPosition]
    convergence_score: float
    stopped_early: bool = False


class DebateResult(BaseModel):
    """The final output of a completed debate pipeline."""

    question: str
    final_answer: str
    final_reasoning: str
    confidence: float
    escalate_to_human: bool = False
    rounds: list[DebateRound]
    aggregation_method: AggregationStrategy
    routing_decision: RoutingDecision
    total_rounds_run: int
    converged: bool
    final_convergence_score: float
    total_token_usage: TokenUsage
    total_cost_usd: float
    duration_seconds: float
