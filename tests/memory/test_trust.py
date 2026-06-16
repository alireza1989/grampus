"""Tests for grampus.memory.trust — TrustScorer."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from grampus.memory.provenance import Provenance, SourceType
from grampus.memory.trust import TrustScorer


def _make_provenance(source_type: SourceType = SourceType.USER_INPUT) -> Provenance:
    return Provenance(
        id=str(uuid.uuid4()),
        source_type=source_type,
        source_id="test",
        trust_level=source_type.trust_level,
        timestamp=datetime.now(UTC),
        content_hash_sha256="abc",
    )


class TestTrustScorer:
    def test_score_returns_float(self) -> None:
        scorer = TrustScorer()
        score = scorer.score(_make_provenance(SourceType.SYSTEM), access_count=0, age_days=0.0)
        assert isinstance(score, float)

    def test_score_clamped_to_unit_interval_lower(self) -> None:
        scorer = TrustScorer()
        score = scorer.score(
            _make_provenance(SourceType.EXTERNAL_DATA), access_count=0, age_days=100_000.0
        )
        assert score >= 0.0

    def test_score_clamped_to_unit_interval_upper(self) -> None:
        scorer = TrustScorer()
        score = scorer.score(
            _make_provenance(SourceType.SYSTEM), access_count=999_999, age_days=0.0
        )
        assert score <= 1.0

    def test_score_zero_age_zero_access_equals_base(self) -> None:
        scorer = TrustScorer()
        p = _make_provenance(SourceType.SYSTEM)
        score = scorer.score(p, access_count=0, age_days=0.0)
        # base * exp(0) + min(0, max_boost) = 1.0 + 0 = 1.0, clamped to 1.0
        assert score == pytest.approx(1.0)

    def test_score_decreases_with_age(self) -> None:
        scorer = TrustScorer()
        p = _make_provenance(SourceType.USER_INPUT)
        score_fresh = scorer.score(p, access_count=0, age_days=0.0)
        score_old = scorer.score(p, access_count=0, age_days=30.0)
        assert score_old < score_fresh

    def test_score_increases_with_access_count(self) -> None:
        scorer = TrustScorer()
        p = _make_provenance(SourceType.LLM_GENERATED)
        score_none = scorer.score(p, access_count=0, age_days=5.0)
        score_many = scorer.score(p, access_count=10, age_days=5.0)
        assert score_many >= score_none

    def test_access_boost_is_capped_at_max_boost(self) -> None:
        scorer = TrustScorer()
        p = _make_provenance(SourceType.EXTERNAL_DATA)
        score_high = scorer.score(p, access_count=10_000, age_days=0.0)
        score_extreme = scorer.score(p, access_count=999_999, age_days=0.0)
        assert score_high == pytest.approx(score_extreme, abs=1e-6)

    def test_system_higher_than_external_same_conditions(self) -> None:
        scorer = TrustScorer()
        s_sys = scorer.score(_make_provenance(SourceType.SYSTEM), access_count=0, age_days=1.0)
        s_ext = scorer.score(
            _make_provenance(SourceType.EXTERNAL_DATA), access_count=0, age_days=1.0
        )
        assert s_sys > s_ext

    def test_custom_decay_rate(self) -> None:
        fast = TrustScorer(decay_rate=0.5)
        slow = TrustScorer(decay_rate=0.001)
        p = _make_provenance(SourceType.USER_INPUT)
        assert fast.score(p, access_count=0, age_days=10.0) < slow.score(
            p, access_count=0, age_days=10.0
        )

    def test_custom_access_boost(self) -> None:
        big_boost = TrustScorer(access_boost=0.1, max_boost=1.0)
        small_boost = TrustScorer(access_boost=0.001, max_boost=1.0)
        p = _make_provenance(SourceType.EXTERNAL_DATA)
        assert big_boost.score(p, access_count=5, age_days=100.0) > small_boost.score(
            p, access_count=5, age_days=100.0
        )
