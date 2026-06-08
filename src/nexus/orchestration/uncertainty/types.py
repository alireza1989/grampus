"""Pydantic models and enums for uncertainty quantification state."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class UncertaintyLevel(StrEnum):
    """Confidence tier assigned to each agent step."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UncertaintySource(StrEnum):
    """Origin of a confidence signal."""

    VERBALIZED = "verbalized"
    P_TRUE = "p_true"
    SEMANTIC_ENTROPY = "semantic_entropy"
    PROPAGATED = "propagated"
    IRREVERSIBLE_TOOL = "irreversible_tool"


class UncertaintyAction(StrEnum):
    """Control action derived from the uncertainty level."""

    PROCEED = "proceed"
    PROCEED_WITH_LOG = "proceed_with_log"
    PAUSE_FOR_HUMAN = "pause_for_human"
    ABORT = "abort"


class StepUncertainty(BaseModel):
    """Per-step uncertainty record capturing all signals and the final decision.

    Args:
        step_id: Unique identifier for the step (e.g. "llm_0").
        step_type: Category — "llm_call", "tool_call", "memory_read", "decision".
        verbalized_confidence: Raw confidence extracted from LLM text (0–1).
        p_true_confidence: Self-evaluation score; -1.0 when not run.
        fused_confidence: Weighted, calibrated fusion of verbalized + P(True).
        propagated_confidence: fused_confidence after SAUP propagation.
        level: Tier derived from propagated_confidence.
        sources: Which estimation signals were used.
        action: Control action derived from level + policy.
        reflection_injected: Whether a System-2 reflection was injected.
        reasoning: Human-readable explanation.
        samples_used: Number of semantic entropy samples drawn (0 = not run).
        timestamp: UTC timestamp of observation.
    """

    step_id: str
    step_type: str
    verbalized_confidence: float
    p_true_confidence: float
    fused_confidence: float
    propagated_confidence: float
    level: UncertaintyLevel
    sources: list[UncertaintySource]
    action: UncertaintyAction
    reflection_injected: bool = False
    reasoning: str = ""
    samples_used: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentBeliefState(BaseModel):
    """Session-level belief state accumulating uncertainty across all steps.

    Args:
        session_id: Identifier for the current session.
        agent_id: Identifier for the agent.
        step_uncertainties: Ordered list of per-step records.
        cumulative_confidence: EMA-propagated confidence across all steps.
        overall_level: Tier derived from cumulative_confidence.
        total_steps: Count of observed steps.
        high_uncertainty_steps: Count of HIGH or CRITICAL steps.
        escalation_history: step_ids that triggered escalation.
    """

    session_id: str
    agent_id: str
    step_uncertainties: list[StepUncertainty] = Field(default_factory=list)
    cumulative_confidence: float = 1.0
    overall_level: UncertaintyLevel = UncertaintyLevel.LOW
    total_steps: int = 0
    high_uncertainty_steps: int = 0
    escalation_history: list[str] = Field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        """Return a compact dict suitable for state.metadata["uncertainty"]."""
        last_id = self.step_uncertainties[-1].step_id if self.step_uncertainties else ""
        return {
            "overall_level": str(self.overall_level),
            "cumulative_confidence": self.cumulative_confidence,
            "total_steps": self.total_steps,
            "high_uncertainty_steps": self.high_uncertainty_steps,
            "last_step_id": last_id,
        }
