"""Uncertainty policy: threshold classification and action decision.

Research basis: Three-tier escalation (Zylos Research April 2026).
PROCEED → PROCEED_WITH_LOG → PAUSE_FOR_HUMAN → ABORT.
Irreversible actions (send_email, delete, deploy) demand PAUSE_FOR_HUMAN at
MEDIUM uncertainty or worse. LOW is always safe.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from nexus.orchestration.uncertainty.types import (
    UncertaintyAction,
    UncertaintyLevel,
)


class UncertaintyPolicy(BaseModel):
    """Maps propagated_confidence to UncertaintyLevel and UncertaintyAction.

    Thresholds (applied to propagated_confidence):
        >= low_threshold    → LOW    → PROCEED
        >= medium_threshold → MEDIUM → PROCEED_WITH_LOG
        >= high_threshold   → HIGH   → PAUSE_FOR_HUMAN
        <  high_threshold   → CRITICAL → ABORT

    Irreversible tool override:
        If tool_name matches any entry in irreversible_tool_names (case-insensitive
        substring match) AND level >= MEDIUM → escalate to PAUSE_FOR_HUMAN.
        LOW uncertainty is always safe even for irreversible tools.

    Reflection injection:
        When level reaches HIGH and inject_reflection_on_high=True, a System-2
        reflection prompt is injected to give the model one self-correction chance.

    Args:
        low_threshold: Propagated confidence floor for LOW level (default 0.80).
        medium_threshold: Floor for MEDIUM level (default 0.60).
        high_threshold: Floor for HIGH level (default 0.40).
        enable_p_true: Passed to estimator to enable P(True) calls.
        enable_semantic_sampling: Passed to estimator to enable entropy sampling.
        irreversible_tool_names: Tool name substrings that trigger escalation.
        inject_reflection_on_high: Inject System-2 prompt on HIGH uncertainty.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    low_threshold: float = 0.80
    medium_threshold: float = 0.60
    high_threshold: float = 0.40

    enable_p_true: bool = True
    enable_semantic_sampling: bool = False

    irreversible_tool_names: list[str] = Field(default_factory=list)
    inject_reflection_on_high: bool = True

    def classify(self, propagated_confidence: float) -> UncertaintyLevel:
        """Map propagated_confidence to an UncertaintyLevel tier.

        Args:
            propagated_confidence: SAUP-propagated confidence value.

        Returns:
            UncertaintyLevel enum value.
        """
        if propagated_confidence >= self.low_threshold:
            return UncertaintyLevel.LOW
        if propagated_confidence >= self.medium_threshold:
            return UncertaintyLevel.MEDIUM
        if propagated_confidence >= self.high_threshold:
            return UncertaintyLevel.HIGH
        return UncertaintyLevel.CRITICAL

    def decide(
        self,
        level: UncertaintyLevel,
        step_type: str,
        tool_name: str | None = None,
    ) -> UncertaintyAction:
        """Map uncertainty level to a control action.

        Applies irreversible tool override: MEDIUM level on an irreversible
        tool escalates to PAUSE_FOR_HUMAN instead of PROCEED_WITH_LOG.

        Args:
            level: Classified uncertainty tier.
            step_type: Step category (unused in base policy, available for override).
            tool_name: Optional tool name to check against irreversible list.

        Returns:
            UncertaintyAction enum value.
        """
        _ = step_type
        if level == UncertaintyLevel.LOW:
            return UncertaintyAction.PROCEED
        if level == UncertaintyLevel.MEDIUM:
            if tool_name and self.is_irreversible(tool_name):
                return UncertaintyAction.PAUSE_FOR_HUMAN
            return UncertaintyAction.PROCEED_WITH_LOG
        if level == UncertaintyLevel.HIGH:
            return UncertaintyAction.PAUSE_FOR_HUMAN
        return UncertaintyAction.ABORT

    def is_irreversible(self, tool_name: str) -> bool:
        """True if tool_name contains any irreversible substring (case-insensitive).

        Args:
            tool_name: Tool name to test.

        Returns:
            True when the tool is considered irreversible.
        """
        lower = tool_name.lower()
        return any(entry.lower() in lower for entry in self.irreversible_tool_names)
