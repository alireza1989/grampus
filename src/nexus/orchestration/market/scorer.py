"""BidScorer — composite bid scoring with calibration discount."""

from __future__ import annotations

from nexus.core.logging import get_logger
from nexus.orchestration.market.reputation import ReputationTracker
from nexus.orchestration.market.types import Bid, BidScore, TaskSpec

_log = get_logger(__name__)

_DEFAULT_ALPHA = 0.35  # reputation weight
_DEFAULT_BETA = 0.45  # calibrated success weight
_DEFAULT_GAMMA = 0.20  # cost efficiency weight

_REJECTED_SCORE = -1.0


class BidScorer:
    """Composite scoring with calibration discount (MarketBench, arXiv 2604.23897).

    Scoring formula:
        calibrated_success = clamp(raw_success * calibration_factor, 0, 1)
        cost_score = 1 / (1 + normalized_cost)
        composite = α×reputation + β×calibrated_success + γ×cost_score
        final = composite + ucb_bonus

    Bids with calibrated_success_prob below task_spec.min_success_threshold
    receive final_score = -1.0 and are filtered by the allocator (moral hazard
    guard against systematic over-reporters).

    Args:
        reputation_tracker: For fetching calibration factors and UCB bonuses.
        alpha: Weight for reputation score. Default 0.35.
        beta: Weight for calibrated success probability. Default 0.45.
        gamma: Weight for cost efficiency. Default 0.20.

    Raises:
        ValueError: If alpha + beta + gamma != 1.0.
    """

    def __init__(
        self,
        reputation_tracker: ReputationTracker,
        alpha: float = _DEFAULT_ALPHA,
        beta: float = _DEFAULT_BETA,
        gamma: float = _DEFAULT_GAMMA,
    ) -> None:
        if abs(alpha + beta + gamma - 1.0) > 1e-6:
            raise ValueError(f"alpha + beta + gamma must equal 1.0, got {alpha + beta + gamma:.6f}")
        self._reputation = reputation_tracker
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

    async def score(self, bid: Bid, task_spec: TaskSpec) -> BidScore:
        """Compute a BidScore for one bid.

        Args:
            bid: The bid to score.
            task_spec: The task specification (provides budget and threshold).

        Returns:
            BidScore with composite and final scores. final_score == -1.0 when
            the calibrated success probability falls below the threshold.
        """
        cal_factor = await self._reputation.calibration_factor(bid.agent_id)
        ucb = await self._reputation.ucb_bonus(bid.agent_id)
        rep_record = await self._reputation.get(bid.agent_id)

        calibrated = min(max(bid.self_reported_success_prob * cal_factor, 0.0), 1.0)

        reputation_score = _reputation_to_score(rep_record)
        cost_score = _cost_score(bid.estimated_cost_usd, task_spec.budget_usd)

        composite = (
            self._alpha * reputation_score + self._beta * calibrated + self._gamma * cost_score
        )

        final = _REJECTED_SCORE if calibrated < task_spec.min_success_threshold else composite + ucb

        _log.debug(
            "bid_scored",
            bid_id=bid.bid_id,
            agent_id=bid.agent_id,
            calibrated=calibrated,
            final=final,
        )
        return BidScore(
            bid_id=bid.bid_id,
            agent_id=bid.agent_id,
            raw_success_prob=bid.self_reported_success_prob,
            calibrated_success_prob=calibrated,
            reputation_score=reputation_score,
            cost_score=cost_score,
            composite=composite,
            ucb_bonus=ucb,
            final_score=final,
        )

    async def score_all(self, bids: list[Bid], task_spec: TaskSpec) -> list[BidScore]:
        """Score all bids and return sorted descending by final_score.

        Args:
            bids: All bids to score.
            task_spec: Task context for normalization and thresholds.

        Returns:
            BidScore list sorted descending by final_score.
        """
        scores = [await self.score(bid, task_spec) for bid in bids]
        scores.sort(key=lambda s: s.final_score, reverse=True)
        return scores


def _reputation_to_score(record: object) -> float:
    """Convert a ReputationRecord to a [0, 1] score.

    New agents (total_tasks == 0) default to 0.5 (neutral prior).
    """
    total = getattr(record, "total_tasks", 0)
    if total == 0:
        return 0.5
    return float(getattr(record, "success_rate", 0.5))


def _cost_score(cost_usd: float, budget_usd: float | None) -> float:
    """Convert estimated cost to a [0, 1] efficiency score.

    Args:
        cost_usd: Estimated cost in USD.
        budget_usd: Task budget; None treats each dollar as a unit.

    Returns:
        1 / (1 + normalized_cost) — higher = cheaper relative to budget.
    """
    normalized = cost_usd / budget_usd if budget_usd is not None and budget_usd > 0 else cost_usd
    return 1.0 / (1.0 + normalized)
