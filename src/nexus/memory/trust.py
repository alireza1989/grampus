"""Trust scoring: time-decayed, access-boosted scores for memory records."""

from __future__ import annotations

import math

from nexus.memory.provenance import Provenance


class TrustScorer:
    """Computes a dynamic trust score from provenance, age, and access frequency.

    Formula: ``clamp(base * exp(-decay_rate * age_days) + min(access_count * access_boost, max_boost), 0, 1)``

    Args:
        decay_rate: Exponential decay rate per day (higher = faster decay).
        access_boost: Per-access increment added to the decayed base.
        max_boost: Upper cap on the total access boost term.
    """

    def __init__(
        self,
        decay_rate: float = 0.01,
        access_boost: float = 0.01,
        max_boost: float = 0.2,
    ) -> None:
        self._decay_rate = decay_rate
        self._access_boost = access_boost
        self._max_boost = max_boost

    def score(
        self,
        provenance: Provenance,
        *,
        access_count: int,
        age_days: float,
    ) -> float:
        """Compute a trust score in [0, 1].

        Args:
            provenance: Source provenance (provides baseline trust level).
            access_count: Number of times this record has been retrieved.
            age_days: Age of the record in fractional days.

        Returns:
            A float in [0.0, 1.0].
        """
        decayed = provenance.trust_level * math.exp(-self._decay_rate * age_days)
        boost = min(access_count * self._access_boost, self._max_boost)
        return max(0.0, min(1.0, decayed + boost))
