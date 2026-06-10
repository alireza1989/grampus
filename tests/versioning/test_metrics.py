"""Tests for QualityTracker and VersionMetrics."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.versioning.metrics import QualityTracker, VersionMetrics


def _make_state_store() -> MagicMock:
    store: dict[tuple[str, str], object] = {}

    mock = MagicMock()

    async def save(entity_type: str, entity_id: str, model: object, **kwargs: object) -> None:
        store[(entity_type, entity_id)] = model

    async def get(entity_type: str, entity_id: str, cls: type) -> tuple[object | None, str]:
        val = store.get((entity_type, entity_id))
        if val is None:
            return None, ""
        if hasattr(val, "model_dump_json"):
            restored = cls.model_validate_json(val.model_dump_json())  # type: ignore[attr-defined]
            return restored, "etag-1"
        return val, "etag-1"

    mock.save = AsyncMock(side_effect=save)
    mock.get = AsyncMock(side_effect=get)
    return mock


class TestVersionMetrics:
    def test_eval_pass_rate_computed_correctly(self) -> None:
        m = VersionMetrics(version_id="v1", eval_pass_count=8, eval_total=10)
        assert m.eval_pass_rate == pytest.approx(0.8)

    def test_eval_pass_rate_none_when_total_zero(self) -> None:
        m = VersionMetrics(version_id="v1")
        assert m.eval_pass_rate is None

    def test_avg_cost_usd_computed(self) -> None:
        m = VersionMetrics(version_id="v1", total_runs=4, total_cost_usd=0.04)
        assert m.avg_cost_usd == pytest.approx(0.01)

    def test_avg_cost_usd_none_when_no_runs(self) -> None:
        m = VersionMetrics(version_id="v1")
        assert m.avg_cost_usd is None

    def test_avg_latency_computed(self) -> None:
        m = VersionMetrics(version_id="v1", total_runs=2, total_latency_seconds=4.0)
        assert m.avg_latency_seconds == pytest.approx(2.0)

    def test_error_rate_computed(self) -> None:
        m = VersionMetrics(version_id="v1", total_runs=10, error_count=2)
        assert m.error_rate == pytest.approx(0.2)

    def test_round_trip_serialization(self) -> None:
        m = VersionMetrics(
            version_id="abc",
            total_runs=5,
            error_count=1,
            total_cost_usd=0.05,
            total_latency_seconds=10.0,
            eval_pass_count=4,
            eval_total=5,
        )
        restored = VersionMetrics.model_validate_json(m.model_dump_json())
        assert restored.total_runs == 5
        assert restored.eval_pass_count == 4


class TestQualityTracker:
    @pytest.mark.asyncio
    async def test_record_run_increments_total_runs(self) -> None:
        tracker = QualityTracker(_make_state_store())
        await tracker.record_run("v1", cost_usd=0.01, latency_seconds=1.0)
        await tracker.record_run("v1", cost_usd=0.02, latency_seconds=2.0)
        m = await tracker.get_metrics("v1")
        assert m.total_runs == 2

    @pytest.mark.asyncio
    async def test_record_run_accumulates_cost(self) -> None:
        tracker = QualityTracker(_make_state_store())
        await tracker.record_run("v1", cost_usd=0.01)
        await tracker.record_run("v1", cost_usd=0.02)
        m = await tracker.get_metrics("v1")
        assert m.total_cost_usd == pytest.approx(0.03)

    @pytest.mark.asyncio
    async def test_record_run_accumulates_latency(self) -> None:
        tracker = QualityTracker(_make_state_store())
        await tracker.record_run("v1", latency_seconds=1.5)
        await tracker.record_run("v1", latency_seconds=2.5)
        m = await tracker.get_metrics("v1")
        assert m.total_latency_seconds == pytest.approx(4.0)

    @pytest.mark.asyncio
    async def test_record_run_error_flag(self) -> None:
        tracker = QualityTracker(_make_state_store())
        await tracker.record_run("v1", error=True)
        await tracker.record_run("v1", error=False)
        m = await tracker.get_metrics("v1")
        assert m.error_count == 1
        assert m.total_runs == 2

    @pytest.mark.asyncio
    async def test_record_eval_result_pass(self) -> None:
        tracker = QualityTracker(_make_state_store())
        await tracker.record_eval_result("v1", passed=True)
        await tracker.record_eval_result("v1", passed=True)
        await tracker.record_eval_result("v1", passed=False)
        m = await tracker.get_metrics("v1")
        assert m.eval_total == 3
        assert m.eval_pass_count == 2

    @pytest.mark.asyncio
    async def test_get_metrics_fresh_version_returns_zeros(self) -> None:
        tracker = QualityTracker(_make_state_store())
        m = await tracker.get_metrics("never-seen")
        assert m.total_runs == 0
        assert m.eval_total == 0
        assert m.version_id == "never-seen"
