"""SAUP-inspired uncertainty propagation across sequential agent steps.

Research basis: arXiv 2412.01033, ACL 2025 (pages 6064–6073).
A minor grounding error in step 1 biases all downstream steps (Spiral of
Hallucination). Per-step-type situational weights prevent one confident step
from erasing a chain of uncertain history.
"""

from __future__ import annotations

SITUATIONAL_WEIGHTS: dict[str, float] = {
    "decision": 0.70,
    "llm_call": 0.55,
    "tool_call": 0.45,
    "memory_read": 0.35,
}

_DEFAULT_WEIGHT = 0.50


class UncertaintyPropagator:
    """Propagates uncertainty across sequential agent steps using SAUP weights.

    Formula for propagated confidence at step t:
        propagated(t) = w * fused(t) + (1 - w) * cumulative(t-1)

    where w = SITUATIONAL_WEIGHTS[step_type] (default 0.50 for unknown types).

    This ensures a single confident step cannot erase a chain of uncertain
    prior steps — the history carries proportional weight.
    """

    def propagate(
        self,
        fused_confidence: float,
        step_type: str,
        cumulative_confidence: float,
    ) -> float:
        """Compute propagated confidence for the current step.

        Args:
            fused_confidence: Calibrated fused estimate for this step.
            step_type: Step category used to look up situational weight.
            cumulative_confidence: EMA cumulative from all prior steps.

        Returns:
            Propagated confidence float in [0.0, 1.0].
        """
        w = SITUATIONAL_WEIGHTS.get(step_type, _DEFAULT_WEIGHT)
        return float(max(0.0, min(1.0, w * fused_confidence + (1.0 - w) * cumulative_confidence)))

    def update_cumulative(
        self,
        propagated_confidence: float,
        current_cumulative: float,
        step_type: str,
    ) -> float:
        """EMA update of session-level cumulative confidence.

        Args:
            propagated_confidence: Propagated value for the current step.
            current_cumulative: Current cumulative before this step.
            step_type: Step category used to look up situational weight.

        Returns:
            New cumulative confidence float in [0.0, 1.0].
        """
        w = SITUATIONAL_WEIGHTS.get(step_type, _DEFAULT_WEIGHT)
        new_val = w * propagated_confidence + (1.0 - w) * current_cumulative
        return float(max(0.0, min(1.0, new_val)))
