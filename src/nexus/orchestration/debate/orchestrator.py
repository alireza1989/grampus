"""Multi-agent debate orchestrator — the main pipeline entry point.

Architecture grounded in:
  Du et al. ICML 2024 — foundational debate framework.
  M3MAD-Bench ICLR 2025 — heterogeneous models beat temperature-only diversity.
  "From Debate to Decision" April 2026 — act-vs-escalate with coverage guarantee.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Generator
from typing import Any

from nexus.core.logging import get_logger
from nexus.core.types import TokenUsage
from nexus.orchestration.debate.aggregator import build_aggregator
from nexus.orchestration.debate.convergence import ConvergenceDetector
from nexus.orchestration.debate.debater import Debater
from nexus.orchestration.debate.router import DebateRouter
from nexus.orchestration.debate.types import (
    DebateConfig,
    DebateResult,
    DebateRound,
    DebaterPosition,
    RoutingDecision,
)

_log = get_logger(__name__)


class DebateOrchestrator:
    """Runs multi-agent debate and returns a DebateResult.

    Args:
        config: DebateConfig specifying debaters, rounds, aggregation, and budgets.
        cost_tracker: Optional CostTracker; ``check_budget()`` is called before each round.
        tracer: Optional tracer with a ``span(name, **attrs)`` context manager method.
    """

    def __init__(
        self,
        config: DebateConfig,
        cost_tracker: Any | None = None,
        tracer: Any | None = None,
    ) -> None:
        self._config = config
        self._cost_tracker = cost_tracker
        self._tracer = tracer
        self._debaters = [Debater(i, cfg) for i, cfg in enumerate(config.debaters)]
        self._router = DebateRouter(config)
        self._aggregator = build_aggregator(config.aggregation, config.judge_config)
        self._detector = ConvergenceDetector(threshold=config.convergence_threshold)

    async def run(self, question: str) -> DebateResult:
        """Execute the full debate pipeline and return a DebateResult.

        Steps:
        1. Adaptive routing check — skip to single-agent if confidence is high.
        2. Round 1: all debaters answer independently (asyncio.gather).
        3. Convergence check → stop early if threshold met.
        4. Rounds 2..max_rounds: debaters respond to peers (asyncio.gather per round).
        5. Aggregation via configured strategy.
        6. Act-vs-escalate: escalate_to_human when convergence < escalate_threshold.
        """
        start = time.monotonic()
        with _maybe_span(
            self._tracer,
            "debate.run",
            question_len=len(question),
            num_debaters=len(self._debaters),
            max_rounds=self._config.max_rounds,
            aggregation=str(self._config.aggregation),
            adaptive_routing=self._config.adaptive_routing,
        ):
            return await self._execute(question, start)

    async def _execute(self, question: str, start: float) -> DebateResult:
        """Internal pipeline; separated so the outer span wraps cleanly."""
        if self._config.adaptive_routing:
            bypass = await self._router.route(question, self._cost_tracker, self._tracer)
            if bypass is not None:
                _log.info("debate.routed_single_agent", question_len=len(question))
                return bypass

        rounds = await self._run_all_rounds(question)
        return await self._build_result(question, rounds, start)

    async def _run_all_rounds(self, question: str) -> list[DebateRound]:
        """Run debate rounds until convergence or max_rounds."""
        rounds: list[DebateRound] = []
        previous_round: DebateRound | None = None

        for round_number in range(1, self._config.max_rounds + 1):
            if self._cost_tracker is not None:
                self._cost_tracker.check_budget()

            rnd = await self._run_round(question, round_number, previous_round)
            rounds.append(rnd)
            previous_round = rnd
            if rnd.stopped_early:
                break

        return rounds

    async def _run_round(
        self,
        question: str,
        round_number: int,
        previous_round: DebateRound | None,
    ) -> DebateRound:
        """Run all debaters concurrently and compute convergence for one round."""
        with _maybe_span(self._tracer, "debate.round", round_number=round_number):
            positions = await self._gather_positions(question, round_number, previous_round)
            conv_score = self._detector.score(positions)
            stopped_early = self._detector.should_stop(positions)
            _log.debug("debate.round_done", round=round_number, convergence=conv_score)
            return DebateRound(
                round_number=round_number,
                positions=positions,
                convergence_score=conv_score,
                stopped_early=stopped_early,
            )

    async def _gather_positions(
        self,
        question: str,
        round_number: int,
        previous_round: DebateRound | None,
    ) -> list[DebaterPosition]:
        """Invoke all debaters concurrently via asyncio.gather."""
        tasks = [
            self._run_one_debater(d, question, round_number, previous_round) for d in self._debaters
        ]
        return list(await asyncio.gather(*tasks))

    async def _run_one_debater(
        self,
        debater: Debater,
        question: str,
        round_number: int,
        previous_round: DebateRound | None,
    ) -> DebaterPosition:
        """Run one debater inside a span; attach post-call attrs to the span."""
        with _maybe_span(
            self._tracer,
            "debate.debater",
            debater_index=debater.index,
            model_id=debater.model_id,
        ) as span:
            position = await debater.respond(
                question, round_number, previous_round, self._cost_tracker, self._tracer
            )
            _set_span_attr(span, "confidence", position.confidence)
            _set_span_attr(span, "changed", position.changed_from_previous)
        return position

    async def _build_result(
        self,
        question: str,
        rounds: list[DebateRound],
        start: float,
    ) -> DebateResult:
        """Aggregate all rounds and construct the final DebateResult."""
        final_round = rounds[-1]
        final_conv = final_round.convergence_score
        converged = final_round.stopped_early

        with _maybe_span(
            self._tracer, "debate.aggregate", strategy=str(self._config.aggregation)
        ) as span:
            answer, reasoning, confidence = await self._aggregator.aggregate(
                question, rounds, self._config.debaters
            )
            _set_span_attr(span, "final_confidence", confidence)
            escalate = final_conv < self._config.escalate_threshold
            _set_span_attr(span, "escalate_to_human", escalate)

        total_usage = _accumulate_all_usage(rounds)
        return DebateResult(
            question=question,
            final_answer=answer,
            final_reasoning=reasoning,
            confidence=confidence,
            escalate_to_human=escalate,
            rounds=rounds,
            aggregation_method=self._config.aggregation,
            routing_decision=RoutingDecision.DEBATE,
            total_rounds_run=len(rounds),
            converged=converged,
            final_convergence_score=final_conv,
            total_token_usage=total_usage,
            total_cost_usd=total_usage.cost_usd,
            duration_seconds=time.monotonic() - start,
        )


@contextlib.contextmanager
def _maybe_span(tracer: Any | None, name: str, **attrs: Any) -> Generator[Any, None, None]:
    """Emit a named span if a tracer is provided; yield None otherwise."""
    if tracer is None:
        yield None
        return
    with tracer.span(name, **attrs) as span:
        yield span


def _set_span_attr(span: Any | None, key: str, value: Any) -> None:
    """Safely set an attribute on a span that may be None or a no-op object."""
    if span is not None and hasattr(span, "set_attribute"):
        span.set_attribute(key, value)


def _accumulate_all_usage(rounds: list[DebateRound]) -> TokenUsage:
    """Sum token usage across all positions in all rounds."""
    total = TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0, cost_usd=0.0, model="")
    for rnd in rounds:
        for pos in rnd.positions:
            if pos.token_usage is not None:
                u = pos.token_usage
                total = TokenUsage(
                    input_tokens=total.input_tokens + u.input_tokens,
                    output_tokens=total.output_tokens + u.output_tokens,
                    total_tokens=total.total_tokens + u.total_tokens,
                    cost_usd=total.cost_usd + u.cost_usd,
                    model=u.model,
                )
    return total
