"""MarketAllocator — end-to-end market allocation pipeline."""

from __future__ import annotations

import json
import uuid
from typing import Any

from grampus.core.logging import get_logger
from grampus.orchestration.market.board import TaskBoard
from grampus.orchestration.market.registry import CapabilityRegistry
from grampus.orchestration.market.reputation import ReputationTracker
from grampus.orchestration.market.scorer import BidScorer
from grampus.orchestration.market.types import (
    AllocationResult,
    AllocationStatus,
    Bid,
    BidScore,
    CapabilityProfile,
    TaskOutcome,
    TaskSpec,
)

_log = get_logger(__name__)

FALLBACK_SUCCESS_PROB = 0.3
FALLBACK_COST_MULTIPLIER = 10

BID_SOLICITATION_PROMPT = """You are agent '{agent_name}' with skills: {skill_tags}.
Your typical cost: ${cost_per_step_usd:.4f}/step.

Task: {description}
Required skills: {required_skills}
Budget: {budget}

Estimate your ability to complete this task. Respond with ONLY valid JSON:
{{
  "self_reported_success_prob": <float 0.0-1.0>,
  "estimated_cost_usd": <float>,
  "estimated_steps": <int>,
  "rationale": "<one sentence>"
}}"""


class MarketAllocator:
    """End-to-end market allocation pipeline.

    Steps:
    1. Filter to capable agents (CapabilityRegistry.filter_capable).
    2. Solicit bids from each capable agent via LLM call.
    3. Score all bids (BidScorer.score_all).
    4. Award to highest final_score above min_success_threshold.
    5. If no bids pass threshold → AllocationStatus.REJECTED.
    6. Update TaskBoard status.

    OTEL spans emitted:
    - market.allocate (parent): task_id, num_candidates, num_bids
    - market.bid_solicitation (child per agent): agent_id
    - market.award: winner_agent_id, final_score, calibrated_success_prob

    Args:
        registry: Agent capability registry.
        board: Task board for posting and status updates.
        scorer: Bid scoring engine.
        reputation: Reputation tracker for calibration data.
        model_client: ModelClient for bid solicitation LLM calls.
        tracer: Optional GrampusTracer for OTEL span emission.
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        board: TaskBoard,
        scorer: BidScorer,
        reputation: ReputationTracker,
        model_client: Any,
        tracer: Any | None = None,
    ) -> None:
        self._registry = registry
        self._board = board
        self._scorer = scorer
        self._reputation = reputation
        self._model_client = model_client
        self._tracer = tracer

    async def allocate(self, spec: TaskSpec) -> AllocationResult:
        """Run the full allocation pipeline.

        Args:
            spec: The task specification to allocate.

        Returns:
            AllocationResult with status and winning bid (if any).
        """
        span_ctx = _maybe_span(
            self._tracer,
            "market.allocate",
            {"task_id": spec.task_id},
        )
        with span_ctx as alloc_span:
            await self._board.post_task(spec)
            await self._board.update_task_status(spec.task_id, AllocationStatus.BIDDING)

            candidates = self._registry.filter_capable(spec.required_skills, spec.preferred_skills)
            filtered_out = [p.agent_id for p in self._registry.list_agents() if p not in candidates]

            if alloc_span is not None:
                alloc_span.set_attribute("num_candidates", len(candidates))

            if not candidates:
                await self._board.update_task_status(spec.task_id, AllocationStatus.REJECTED)
                return AllocationResult(
                    task_id=spec.task_id,
                    status=AllocationStatus.REJECTED,
                    capability_filtered_out=filtered_out,
                    reject_reason="No capable agents found for required skills.",
                )

            bids = await self._solicit_bids(candidates, spec)
            if alloc_span is not None:
                alloc_span.set_attribute("num_bids", len(bids))

            for b in bids:
                await self._board.submit_bid(b)
                self._reputation.record_self_report(b.agent_id, b.self_reported_success_prob)

            scores = await self._scorer.score_all(bids, spec)
            winners = [s for s in scores if s.final_score > 0.0]

            if not winners:
                await self._board.update_task_status(spec.task_id, AllocationStatus.REJECTED)
                return AllocationResult(
                    task_id=spec.task_id,
                    status=AllocationStatus.REJECTED,
                    all_scores=scores,
                    capability_filtered_out=filtered_out,
                    reject_reason="All bids below minimum success threshold after calibration.",
                )

            best_score = winners[0]
            winning_bid = next(b for b in bids if b.bid_id == best_score.bid_id)

            if alloc_span is not None:
                _emit_award_span(self._tracer, best_score)

            await self._board.update_task_status(spec.task_id, AllocationStatus.ALLOCATED)

            _log.info(
                "market_allocated",
                task_id=spec.task_id,
                winner=best_score.agent_id,
                final_score=best_score.final_score,
            )
            return AllocationResult(
                task_id=spec.task_id,
                status=AllocationStatus.ALLOCATED,
                winning_agent_id=best_score.agent_id,
                winning_bid=winning_bid,
                winning_score=best_score,
                all_scores=scores,
                capability_filtered_out=filtered_out,
            )

    async def report_outcome(self, outcome: TaskOutcome) -> None:
        """Update board status and reputation after a task completes.

        Args:
            outcome: The completed task outcome.
        """
        await self._board.mark_outcome(outcome)
        await self._reputation.update(outcome)
        _log.debug("market_outcome_reported", task_id=outcome.task_id, agent_id=outcome.agent_id)

    async def _solicit_bids(self, candidates: list[CapabilityProfile], spec: TaskSpec) -> list[Bid]:
        """Solicit bids from all capable candidates concurrently.

        Args:
            candidates: Agents to solicit.
            spec: Task specification.

        Returns:
            List of Bid objects (one per candidate).
        """
        import asyncio

        return list(await asyncio.gather(*[self._solicit_bid(p, spec) for p in candidates]))

    async def _solicit_bid(self, profile: CapabilityProfile, spec: TaskSpec) -> Bid:
        """Send a structured prompt asking the agent to produce a Bid.

        Uses model_client.complete() with max_tokens=200 and parses the JSON
        response. Falls back to a conservative default on parse failure so
        allocation is never aborted by a single bad response.

        Args:
            profile: The agent profile (provides name, skills, cost).
            spec: Task specification.

        Returns:
            Bid — either parsed from model response or conservative fallback.
        """
        span_ctx = _maybe_span(
            self._tracer,
            "market.bid_solicitation",
            {"agent_id": profile.agent_id},
        )
        with span_ctx:
            budget_str = f"${spec.budget_usd:.4f}" if spec.budget_usd is not None else "unlimited"
            prompt = BID_SOLICITATION_PROMPT.format(
                agent_name=profile.agent_name,
                skill_tags=", ".join(profile.skill_tags),
                cost_per_step_usd=profile.cost_per_step_usd,
                description=spec.description,
                required_skills=", ".join(spec.required_skills),
                budget=budget_str,
            )
            from grampus.core.types import Message, Role

            try:
                response = await self._model_client.complete(
                    messages=[Message(role=Role.USER, content=prompt)],
                    model=getattr(self._model_client, "default_model", ""),
                    max_tokens=200,
                )
                content = response.content or ""
                parsed = json.loads(content)
                return Bid(
                    bid_id=str(uuid.uuid4()),
                    task_id=spec.task_id,
                    agent_id=profile.agent_id,
                    self_reported_success_prob=float(
                        parsed.get("self_reported_success_prob", FALLBACK_SUCCESS_PROB)
                    ),
                    estimated_cost_usd=float(
                        parsed.get(
                            "estimated_cost_usd",
                            profile.cost_per_step_usd * FALLBACK_COST_MULTIPLIER,
                        )
                    ),
                    estimated_steps=int(parsed.get("estimated_steps", 10)),
                    rationale=str(parsed.get("rationale", "")),
                )
            except Exception:  # noqa: BLE001
                _log.debug("bid_solicitation_fallback", agent_id=profile.agent_id)
                return _fallback_bid(profile, spec)


def _fallback_bid(profile: CapabilityProfile, spec: TaskSpec) -> Bid:
    """Conservative default bid used when model response cannot be parsed."""
    return Bid(
        bid_id=str(uuid.uuid4()),
        task_id=spec.task_id,
        agent_id=profile.agent_id,
        self_reported_success_prob=FALLBACK_SUCCESS_PROB,
        estimated_cost_usd=profile.cost_per_step_usd * FALLBACK_COST_MULTIPLIER,
        estimated_steps=10,
        rationale="Fallback bid (parse error).",
    )


def _emit_award_span(tracer: Any | None, score: BidScore) -> None:
    """Emit market.award OTEL span with winner details."""
    if tracer is None:
        return
    ctx = _maybe_span(
        tracer,
        "market.award",
        {
            "winner_agent_id": score.agent_id,
            "final_score": score.final_score,
            "calibrated_success_prob": score.calibrated_success_prob,
        },
    )
    with ctx:
        pass


class _NullSpan:
    """Minimal no-op span returned when tracer is None."""

    def set_attribute(self, key: str, value: object) -> None:  # noqa: ARG002
        pass

    def __enter__(self) -> _NullSpan:
        return self

    def __exit__(self, *_: object) -> None:
        pass


def _maybe_span(tracer: Any | None, name: str, attrs: dict[str, Any]) -> Any:
    """Return a real span context or a no-op if tracer is None."""
    if tracer is None:
        return _NullSpan()
    return tracer._tracer.start_as_current_span(name)
