"""Pydantic v2 data models for the market-based task allocation system."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AgentTier(StrEnum):
    """Model capability tier advertised by a worker agent."""

    FAST = "fast"
    BALANCED = "balanced"
    POWERFUL = "powerful"


class AllocationStatus(StrEnum):
    """Lifecycle state of a task on the board."""

    PENDING = "pending"
    BIDDING = "bidding"
    ALLOCATED = "allocated"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"


class CapabilityProfile(BaseModel):
    """What a worker agent advertises it can do.

    Args:
        agent_id: Unique identifier for the worker agent.
        agent_name: Human-readable name.
        skill_tags: List of skill labels (e.g. ["web_search", "sql"]).
        model_tier: Capability tier of the underlying model.
        cost_per_step_usd: Self-reported cost estimate per step.
        max_steps: Maximum steps agent will attempt.
        latency_sla_ms: Optional latency SLA commitment in milliseconds.
        metadata: Arbitrary extra metadata.
        registered_at: UTC timestamp of registration.
    """

    agent_id: str
    agent_name: str
    skill_tags: list[str]
    model_tier: AgentTier = AgentTier.BALANCED
    cost_per_step_usd: float = 0.0
    max_steps: int = 20
    latency_sla_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TaskSpec(BaseModel):
    """A task the supervisor posts to the board.

    Args:
        task_id: Unique identifier.
        description: Natural language task description.
        required_skills: Skills that must be present (hard filter).
        preferred_skills: Skills that improve ranking (soft filter).
        budget_usd: Hard cost cap; None means unlimited.
        min_success_threshold: Minimum calibrated success probability to accept a bid.
        deadline_ms: Optional wall-clock budget in milliseconds.
        allow_partial: Whether a PARTIAL outcome counts as success.
        metadata: Arbitrary extra metadata.
    """

    task_id: str
    description: str
    required_skills: list[str]
    preferred_skills: list[str] = Field(default_factory=list)
    budget_usd: float | None = None
    min_success_threshold: float = 0.5
    deadline_ms: int | None = None
    allow_partial: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Bid(BaseModel):
    """A worker agent's response to a task posting.

    Args:
        bid_id: Unique identifier for this bid.
        task_id: The task this bid is for.
        agent_id: The bidding agent.
        self_reported_success_prob: Agent's own estimate (0–1); will be discounted.
        estimated_cost_usd: Self-reported cost estimate.
        estimated_steps: Estimated number of steps to complete.
        estimated_latency_ms: Optional latency estimate in milliseconds.
        rationale: Brief explanation (1–2 sentences).
        submitted_at: UTC timestamp of submission.
    """

    bid_id: str
    task_id: str
    agent_id: str
    self_reported_success_prob: float
    estimated_cost_usd: float
    estimated_steps: int
    estimated_latency_ms: int | None = None
    rationale: str = ""
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BidScore(BaseModel):
    """Composite score computed by BidScorer for a single bid.

    Args:
        bid_id: The bid this score corresponds to.
        agent_id: The bidding agent.
        raw_success_prob: Self-reported probability before calibration.
        calibrated_success_prob: After reputation calibration discount.
        reputation_score: Derived from ReputationTracker (0–1).
        cost_score: Inverse normalized cost (higher = cheaper).
        composite: Weighted blend of reputation, calibrated success, and cost.
        ucb_bonus: Exploration bonus for agents with few historical runs.
        final_score: composite + ucb_bonus.
    """

    bid_id: str
    agent_id: str
    raw_success_prob: float
    calibrated_success_prob: float
    reputation_score: float
    cost_score: float
    composite: float
    ucb_bonus: float
    final_score: float


class AllocationResult(BaseModel):
    """Returned by MarketAllocator.allocate().

    Args:
        task_id: The task that was allocated.
        status: Final allocation status.
        winning_agent_id: Agent that won, or None if rejected.
        winning_bid: The winning Bid object.
        winning_score: The winning BidScore object.
        all_scores: All computed BidScore objects (sorted descending).
        capability_filtered_out: Agent IDs filtered before bid solicitation.
        reject_reason: Human-readable reason when status is REJECTED.
        allocated_at: UTC timestamp of allocation decision.
    """

    task_id: str
    status: AllocationStatus
    winning_agent_id: str | None = None
    winning_bid: Bid | None = None
    winning_score: BidScore | None = None
    all_scores: list[BidScore] = Field(default_factory=list)
    capability_filtered_out: list[str] = Field(default_factory=list)
    reject_reason: str | None = None
    allocated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TaskOutcome(BaseModel):
    """Reported back to ReputationTracker after a task completes.

    Args:
        task_id: The completed task.
        agent_id: The agent that executed it.
        actual_success: Whether the task succeeded.
        actual_cost_usd: Actual cost incurred.
        actual_steps: Actual number of steps taken.
        actual_latency_ms: Actual latency in milliseconds, if measured.
        completed_at: UTC timestamp of completion.
    """

    task_id: str
    agent_id: str
    actual_success: bool
    actual_cost_usd: float
    actual_steps: int
    actual_latency_ms: int | None = None
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ReputationRecord(BaseModel):
    """Per-agent reputation stored in Dapr state.

    Args:
        agent_id: The agent this record belongs to.
        total_tasks: Number of tasks completed (success or failure).
        successful_tasks: Number of successful completions.
        success_rate: successful_tasks / total_tasks (running ratio).
        cost_accuracy: EMA of actual_cost / estimated_cost; 1.0 = perfect.
        latency_accuracy: EMA of actual_latency / estimated_latency; 1.0 = perfect.
        calibration_factor: EMA multiplier applied to self_reported_success_prob.
        ucb_confidence: Current UCB exploration bonus (sqrt term).
        last_updated: UTC timestamp of last update.
    """

    agent_id: str
    total_tasks: int = 0
    successful_tasks: int = 0
    success_rate: float = 0.0
    cost_accuracy: float = 1.0
    latency_accuracy: float = 1.0
    calibration_factor: float = 1.0
    ucb_confidence: float = 1.0
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))
