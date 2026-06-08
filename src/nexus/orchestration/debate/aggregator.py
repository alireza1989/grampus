"""Aggregation strategies for determining the winning position from a debate.

Three strategies (Du et al. ICML 2024; "Demystifying Multi-Agent Debate"):
  MAJORITY_VOTE  — largest Jaccard cluster, highest-confidence representative.
  WEIGHTED_VOTE  — cluster scored by sum(debater.weight * confidence).
  JUDGE          — separate judge model synthesises all positions; falls back to majority.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from nexus.core.logging import get_logger
from nexus.core.types import Message, Role
from nexus.orchestration.debate.convergence import ConvergenceDetector
from nexus.orchestration.debate.types import (
    AggregationStrategy,
    DebaterConfig,
    DebateRound,
    DebaterPosition,
)

_log = get_logger(__name__)

_JACCARD_MIN = 0.4


class BaseAggregator(ABC):
    """Abstract base for all debate aggregators."""

    @abstractmethod
    async def aggregate(
        self,
        question: str,
        rounds: list[DebateRound],
        debater_configs: list[DebaterConfig],
    ) -> tuple[str, str, float]:
        """Return (final_answer, final_reasoning, confidence)."""


class MajorityVoteAggregator(BaseAggregator):
    """Selects the largest Jaccard cluster; representative is highest-confidence member."""

    async def aggregate(
        self,
        question: str,
        rounds: list[DebateRound],
        debater_configs: list[DebaterConfig],
    ) -> tuple[str, str, float]:
        """Return the winning cluster's best answer."""
        positions = rounds[-1].positions
        detector = ConvergenceDetector(jaccard_min=_JACCARD_MIN)
        clusters = detector.cluster_positions(positions)
        best_cluster = max(clusters, key=len)
        rep = max(best_cluster, key=lambda p: p.confidence)
        avg_conf = sum(p.confidence for p in best_cluster) / len(best_cluster)
        return rep.answer, rep.reasoning, avg_conf


class WeightedVoteAggregator(BaseAggregator):
    """Scores clusters by sum(debater.weight * confidence); picks highest-scoring cluster."""

    async def aggregate(
        self,
        question: str,
        rounds: list[DebateRound],
        debater_configs: list[DebaterConfig],
    ) -> tuple[str, str, float]:
        """Return the highest-weighted cluster's best answer."""
        positions = rounds[-1].positions
        detector = ConvergenceDetector(jaccard_min=_JACCARD_MIN)
        clusters = detector.cluster_positions(positions)

        def _cluster_score(cluster: list[DebaterPosition]) -> float:
            return sum(debater_configs[p.debater_index].weight * p.confidence for p in cluster)

        best_cluster = max(clusters, key=_cluster_score)
        rep = max(best_cluster, key=lambda p: p.confidence)
        avg_conf = sum(p.confidence for p in best_cluster) / len(best_cluster)
        return rep.answer, rep.reasoning, avg_conf


class JudgeAggregator(BaseAggregator):
    """Calls a separate judge model to synthesise all final-round positions.

    Falls back to MajorityVoteAggregator if JSON parsing fails.

    Args:
        judge_config: DebaterConfig for the judge model.
    """

    def __init__(self, judge_config: DebaterConfig) -> None:
        self._judge_config = judge_config
        self._fallback = MajorityVoteAggregator()

    async def aggregate(
        self,
        question: str,
        rounds: list[DebateRound],
        debater_configs: list[DebaterConfig],
    ) -> tuple[str, str, float]:
        """Call judge model; parse JSON; fall back to majority vote on failure."""
        positions = rounds[-1].positions
        positions_text = "\n".join(
            f"Agent {p.debater_index}: {p.answer}\nReasoning: {p.reasoning}" for p in positions
        )
        system_msg = Message(
            role=Role.SYSTEM,
            content=(
                "You are a debate judge. Given the following agent positions, "
                "synthesise the best final answer.\n"
                'Output JSON: {"answer": "...", "reasoning": "...", "confidence": 0.0-1.0}'
            ),
        )
        user_msg = Message(
            role=Role.USER,
            content=f"Question: {question}\n\nPositions:\n{positions_text}",
        )
        try:
            response = await self._judge_config.model_client.complete(
                messages=[system_msg, user_msg],
                model=self._judge_config.model_id,
                temperature=self._judge_config.temperature,
            )
            parsed = _parse_json(response.content or "")
            if not parsed or "answer" not in parsed:
                raise ValueError("missing answer key")
            return (
                str(parsed["answer"]),
                str(parsed.get("reasoning", "")),
                float(parsed.get("confidence", 0.5)),
            )
        except Exception as exc:
            _log.warning("judge_aggregator_fallback", error=str(exc))
            return await self._fallback.aggregate(question, rounds, debater_configs)


def build_aggregator(
    strategy: AggregationStrategy,
    judge_config: DebaterConfig | None = None,
) -> BaseAggregator:
    """Factory: return the correct aggregator for the given strategy."""
    if strategy == AggregationStrategy.MAJORITY_VOTE:
        return MajorityVoteAggregator()
    if strategy == AggregationStrategy.WEIGHTED_VOTE:
        return WeightedVoteAggregator()
    if strategy == AggregationStrategy.JUDGE:
        if judge_config is None:
            raise ValueError("judge_config is required for JUDGE aggregation strategy")
        return JudgeAggregator(judge_config)
    raise ValueError(f"Unknown aggregation strategy: {strategy}")  # pragma: no cover


def _parse_json(content: str) -> dict[str, Any]:
    """Extract and parse the first JSON object from LLM output."""
    text = content.strip()
    try:
        return dict(json.loads(text))
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return dict(json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass
    match2 = re.search(r"\{.*\}", text, re.DOTALL)
    if match2:
        try:
            return dict(json.loads(match2.group(0)))
        except json.JSONDecodeError:
            pass
    return {}
