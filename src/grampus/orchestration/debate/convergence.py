"""Convergence detection for multi-agent debate using Jaccard word-overlap clustering.

Grounded in: "Adaptive Stability Detection" (NeurIPS 2025) — stop debate early when
>= convergence_threshold fraction of debaters agree, using positional clustering rather
than exact string match.
"""

from __future__ import annotations

from grampus.orchestration.debate.types import DebaterPosition


class ConvergenceDetector:
    """Detects when debaters have reached sufficient agreement to stop early.

    Two positions are in the same cluster if their answer Jaccard word overlap
    >= jaccard_min. The convergence score is:
        size_of_largest_cluster / total_debaters

    Args:
        threshold: Fraction of debaters that must agree (default 0.8).
        jaccard_min: Minimum Jaccard similarity to consider two answers equivalent.
    """

    def __init__(self, threshold: float = 0.8, jaccard_min: float = 0.4) -> None:
        self._threshold = threshold
        self._jaccard_min = jaccard_min

    def cluster_positions(self, positions: list[DebaterPosition]) -> list[list[DebaterPosition]]:
        """Group positions into Jaccard-similarity clusters.

        Returns:
            List of clusters, each a list of DebaterPosition objects.
            Positions are greedily assigned to the first cluster they match.
        """
        if not positions:
            return []

        clusters: list[list[DebaterPosition]] = []
        assigned = [False] * len(positions)

        for i, pos in enumerate(positions):
            if assigned[i]:
                continue
            cluster = [pos]
            assigned[i] = True
            for j in range(i + 1, len(positions)):
                if (
                    not assigned[j]
                    and self._jaccard(pos.answer, positions[j].answer) >= self._jaccard_min
                ):
                    cluster.append(positions[j])
                    assigned[j] = True
            clusters.append(cluster)

        return clusters

    def score(self, positions: list[DebaterPosition]) -> float:
        """Return fraction of debaters in the largest agreement cluster.

        Args:
            positions: All positions from one debate round.

        Returns:
            Float in [0, 1]. 1.0 means unanimous agreement.
        """
        if not positions:
            return 0.0
        if len(positions) == 1:
            return 1.0
        clusters = self.cluster_positions(positions)
        largest = max(len(c) for c in clusters)
        return largest / len(positions)

    def should_stop(self, positions: list[DebaterPosition]) -> bool:
        """Return True when the largest cluster meets or exceeds the threshold."""
        return self.score(positions) >= self._threshold

    def _jaccard(self, a: str, b: str) -> float:
        """Jaccard similarity between word sets of a and b."""
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if not wa and not wb:
            return 1.0
        return len(wa & wb) / len(wa | wb)
