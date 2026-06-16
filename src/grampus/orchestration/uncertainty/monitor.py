"""Session-level uncertainty monitor implementing Dual-Process AUQ.

Research basis: arXiv 2601.15703 (Jan 2026) — System 1 (fast P(True) + verbalized
fusion) always runs; System 2 (reflection + semantic entropy) activates on HIGH.
"""

from __future__ import annotations

from typing import Any

from grampus.core.logging import get_logger
from grampus.orchestration.uncertainty.estimator import UncertaintyEstimator
from grampus.orchestration.uncertainty.policy import UncertaintyPolicy
from grampus.orchestration.uncertainty.propagator import UncertaintyPropagator
from grampus.orchestration.uncertainty.types import (
    AgentBeliefState,
    StepUncertainty,
    UncertaintyAction,
    UncertaintyLevel,
    UncertaintySource,
)

_log = get_logger(__name__)

REFLECTION_PROMPT = """
Before you continue, assess your own uncertainty explicitly.
List the specific things you are NOT confident about in your current reasoning.
For each uncertain point state: (1) what you know, (2) what you don't know,
(3) what additional information would resolve the uncertainty.
If you are fully confident, state that explicitly with your justification.
"""


class UncertaintyMonitor:
    """Session-level uncertainty tracking implementing Dual-Process AUQ.

    System 1 (fast, always):
        P(True) + verbalized fusion → SAUP propagation → classify → decide.

    System 2 (slow, triggered on uncertain zone):
        Semantic entropy sampling when fused confidence is in trigger zone.
        Reflection injection when HIGH uncertainty is detected.

    Args:
        estimator: UncertaintyEstimator instance; defaults to UncertaintyEstimator().
        propagator: UncertaintyPropagator instance; defaults to UncertaintyPropagator().
        policy: UncertaintyPolicy instance; defaults to UncertaintyPolicy().
        tracer: Optional GrampusTracer for OTEL spans (duck-typed).
    """

    def __init__(
        self,
        *,
        estimator: UncertaintyEstimator | None = None,
        propagator: UncertaintyPropagator | None = None,
        policy: UncertaintyPolicy | None = None,
        tracer: Any | None = None,
    ) -> None:
        self._estimator = estimator or UncertaintyEstimator()
        self._propagator = propagator or UncertaintyPropagator()
        self._policy = policy or UncertaintyPolicy()
        self._tracer = tracer
        self._belief: AgentBeliefState = AgentBeliefState(session_id="", agent_id="")

    def initialize(self, session_id: str, agent_id: str) -> None:
        """Reset belief state for a new session. Call at start of each run()."""
        self._belief = AgentBeliefState(session_id=session_id, agent_id=agent_id)

    async def observe_llm_response(
        self,
        response_text: str,
        step_id: str,
        *,
        prompt_messages: list[Any] | None = None,
        model_client: Any | None = None,
        model_id: str = "",
        step_type: str = "llm_call",
    ) -> tuple[StepUncertainty, UncertaintyAction]:
        """Full estimation pipeline for one LLM response.

        Runs System 1 always; System 2 when fused confidence in trigger zone.

        Returns:
            (StepUncertainty record, UncertaintyAction to take).
        """
        verbalized_raw = self._estimator.extract_verbalized(response_text)
        p_true_ran = False
        p_true_raw = -1.0

        if self._policy.enable_p_true and model_client is not None:
            p_true_raw = await self._estimator.p_true(
                original_messages=prompt_messages or [],
                response_text=response_text,
                model_client=model_client,
                model_id=model_id,
            )
            p_true_ran = True

        fused = self._estimator.fuse(verbalized_raw, p_true_raw, p_true_ran=p_true_ran)
        samples_used = 0

        if (
            self._policy.enable_semantic_sampling
            and self._estimator.should_sample(fused)
            and model_client is not None
        ):
            entropy_conf, samples_used = await self._estimator.semantic_entropy(
                messages=prompt_messages or [],
                model_client=model_client,
                model_id=model_id,
            )
            fused = min(fused, entropy_conf)

        return self._build_and_record(
            step_id=step_id,
            step_type=step_type,
            verbalized_raw=verbalized_raw,
            p_true_raw=p_true_raw,
            p_true_ran=p_true_ran,
            fused=fused,
            samples_used=samples_used,
            tool_name=None,
        )

    async def observe_tool_call(
        self,
        tool_name: str,
        step_id: str,
    ) -> tuple[StepUncertainty, UncertaintyAction]:
        """Check uncertainty before executing a tool using current cumulative state.

        Does not make any new LLM calls — uses existing cumulative_confidence.

        Returns:
            (StepUncertainty record, UncertaintyAction to take).
        """
        return self._build_and_record(
            step_id=step_id,
            step_type="tool_call",
            verbalized_raw=self._belief.cumulative_confidence,
            p_true_raw=-1.0,
            p_true_ran=False,
            fused=self._belief.cumulative_confidence,
            samples_used=0,
            tool_name=tool_name,
        )

    def get_reflection_message(self) -> Any:
        """Return a SYSTEM Message containing the System-2 reflection prompt."""
        from grampus.core.types import Message, Role

        return Message(role=Role.SYSTEM, content=REFLECTION_PROMPT)

    def get_belief_state(self) -> AgentBeliefState:
        """Return the current session belief state."""
        return self._belief

    def reset(self) -> None:
        """Reset belief state to empty."""
        self._belief = AgentBeliefState(session_id="", agent_id="")

    def summary_metadata(self) -> dict[str, Any]:
        """Return compact metadata dict for state.metadata["uncertainty"]."""
        return self._belief.to_summary()

    def _build_and_record(
        self,
        *,
        step_id: str,
        step_type: str,
        verbalized_raw: float,
        p_true_raw: float,
        p_true_ran: bool,
        fused: float,
        samples_used: int,
        tool_name: str | None,
    ) -> tuple[StepUncertainty, UncertaintyAction]:
        """Propagate, classify, decide, record, emit span."""
        propagated = self._propagator.propagate(
            fused, step_type, self._belief.cumulative_confidence
        )
        new_cumulative = self._propagator.update_cumulative(
            propagated, self._belief.cumulative_confidence, step_type
        )
        level = self._policy.classify(propagated)
        action = self._policy.decide(level, step_type, tool_name)

        sources = _build_sources(
            p_true_ran=p_true_ran,
            samples_used=samples_used,
            tool_name=tool_name,
            policy=self._policy,
        )

        step = StepUncertainty(
            step_id=step_id,
            step_type=step_type,
            verbalized_confidence=verbalized_raw,
            p_true_confidence=p_true_raw,
            fused_confidence=fused,
            propagated_confidence=propagated,
            level=level,
            sources=sources,
            action=action,
            samples_used=samples_used,
        )

        self._belief.step_uncertainties.append(step)
        self._belief.cumulative_confidence = new_cumulative
        self._belief.overall_level = self._policy.classify(new_cumulative)
        self._belief.total_steps += 1
        if level in (UncertaintyLevel.HIGH, UncertaintyLevel.CRITICAL):
            self._belief.high_uncertainty_steps += 1
            self._belief.escalation_history.append(step_id)

        self._emit_span(step)
        return step, action

    def _emit_span(self, step: StepUncertainty) -> None:
        """Emit OTEL uncertainty.estimate span (and uncertainty.escalate if needed)."""
        if self._tracer is None:
            return
        try:
            span_name = "uncertainty.estimate"
            attrs: dict[str, Any] = {
                "step_id": step.step_id,
                "step_type": step.step_type,
                "verbalized_confidence": step.verbalized_confidence,
                "p_true_confidence": step.p_true_confidence,
                "fused_confidence": step.fused_confidence,
                "propagated_confidence": step.propagated_confidence,
                "level": str(step.level),
                "action": str(step.action),
                "p_true_ran": step.p_true_confidence != -1.0,
                "samples_used": step.samples_used,
            }
            with self._tracer._tracer.start_as_current_span(span_name) as span:
                for k, v in attrs.items():
                    span.set_attribute(k, v)

            if step.samples_used > 0:
                with self._tracer._tracer.start_as_current_span("uncertainty.semantic") as span:
                    span.set_attribute("step_id", step.step_id)
                    span.set_attribute("sample_count", step.samples_used)

            if step.level in (UncertaintyLevel.HIGH, UncertaintyLevel.CRITICAL):
                with self._tracer._tracer.start_as_current_span("uncertainty.escalate") as span:
                    span.set_attribute("step_id", step.step_id)
                    span.set_attribute("step_type", step.step_type)
                    span.set_attribute("level", str(step.level))
                    span.set_attribute("cumulative_confidence", self._belief.cumulative_confidence)
                    span.set_attribute(
                        "irreversible", UncertaintySource.IRREVERSIBLE_TOOL in step.sources
                    )
        except Exception:  # noqa: BLE001
            _log.debug("uncertainty_span_failed", step_id=step.step_id)


def _build_sources(
    *,
    p_true_ran: bool,
    samples_used: int,
    tool_name: str | None,
    policy: UncertaintyPolicy,
) -> list[UncertaintySource]:
    """Assemble list of UncertaintySource values used for this step."""
    sources = [UncertaintySource.VERBALIZED]
    if p_true_ran:
        sources.append(UncertaintySource.P_TRUE)
    if samples_used > 0:
        sources.append(UncertaintySource.SEMANTIC_ENTROPY)
    if tool_name is not None:
        sources.append(UncertaintySource.PROPAGATED)
        if policy.is_irreversible(tool_name):
            sources.append(UncertaintySource.IRREVERSIBLE_TOOL)
    return sources
