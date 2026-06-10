"""Tests for A/B testing — ABTestManager and VersionRouter with A/B routing."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.core.errors import VersioningError
from nexus.core.types import AgentDefinition
from nexus.versioning.ab_testing import ABTestManager
from nexus.versioning.manager import VersionManager
from nexus.versioning.metrics import QualityTracker
from nexus.versioning.router import VersionRouter
from nexus.versioning.store import VersionStore
from nexus.versioning.types import ABTestConfig, SuccessMetric


def _make_def(prompt: str, name: str = "ab-agent") -> AgentDefinition:
    return AgentDefinition(name=name, model="m", system_prompt=prompt, tools=[])


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


def _make_infra() -> tuple[VersionStore, QualityTracker]:
    ss = _make_state_store()
    return VersionStore(ss), QualityTracker(ss)


class TestABTestConfig:
    def test_round_trip_serialization(self) -> None:
        cfg = ABTestConfig(
            experiment_id="exp-1",
            agent_id="my-agent",
            control_version_id="ctrl",
            treatment_version_id="trt",
            traffic_split=0.2,
            created_at=datetime.now(UTC),
        )
        restored = ABTestConfig.model_validate_json(cfg.model_dump_json())
        assert restored.experiment_id == "exp-1"
        assert restored.traffic_split == 0.2
        assert restored.active is True

    def test_default_values(self) -> None:
        cfg = ABTestConfig(
            experiment_id="e",
            agent_id="a",
            control_version_id="c",
            treatment_version_id="t",
            traffic_split=0.1,
            created_at=datetime.now(UTC),
        )
        assert cfg.success_metric == SuccessMetric.eval_pass_rate
        assert cfg.auto_promote_threshold == 0.05
        assert cfg.min_samples == 100


class TestABTestManager:
    @pytest.mark.asyncio
    async def test_start_test_creates_and_persists(self) -> None:
        store, tracker = _make_infra()
        mgr = ABTestManager(store, tracker)
        cfg = await mgr.start_test(
            "ab-agent",
            control_version_id="ctrl",
            treatment_version_id="trt",
        )
        assert cfg.agent_id == "ab-agent"
        assert cfg.control_version_id == "ctrl"
        assert cfg.treatment_version_id == "trt"
        assert cfg.active is True

        retrieved = await mgr.get_test(cfg.experiment_id)
        assert retrieved is not None
        assert retrieved.experiment_id == cfg.experiment_id

    @pytest.mark.asyncio
    async def test_start_second_test_raises(self) -> None:
        store, tracker = _make_infra()
        mgr = ABTestManager(store, tracker)
        await mgr.start_test("dup-agent", control_version_id="c", treatment_version_id="t")
        with pytest.raises(VersioningError) as exc_info:
            await mgr.start_test("dup-agent", control_version_id="c2", treatment_version_id="t2")
        assert exc_info.value.code == "TEST_ALREADY_ACTIVE"

    @pytest.mark.asyncio
    async def test_stop_test_marks_inactive(self) -> None:
        store, tracker = _make_infra()
        mgr = ABTestManager(store, tracker)
        cfg = await mgr.start_test("stop-agent", control_version_id="c", treatment_version_id="t")
        stopped = await mgr.stop_test(cfg.experiment_id)
        assert stopped.active is False

    @pytest.mark.asyncio
    async def test_get_active_test_returns_active(self) -> None:
        store, tracker = _make_infra()
        mgr = ABTestManager(store, tracker)
        cfg = await mgr.start_test("active-agent", control_version_id="c", treatment_version_id="t")
        active = await mgr.get_active_test("active-agent")
        assert active is not None
        assert active.experiment_id == cfg.experiment_id

    @pytest.mark.asyncio
    async def test_get_active_test_none_when_no_test(self) -> None:
        store, tracker = _make_infra()
        mgr = ABTestManager(store, tracker)
        assert await mgr.get_active_test("no-agent") is None

    @pytest.mark.asyncio
    async def test_get_active_test_none_after_stop(self) -> None:
        store, tracker = _make_infra()
        mgr = ABTestManager(store, tracker)
        cfg = await mgr.start_test("ended-agent", control_version_id="c", treatment_version_id="t")
        await mgr.stop_test(cfg.experiment_id)
        assert await mgr.get_active_test("ended-agent") is None


class TestVersionRouterABRouting:
    @pytest.mark.asyncio
    async def test_sticky_routing_same_user_same_version(self) -> None:
        store, tracker = _make_infra()
        ab_mgr = ABTestManager(store, tracker)

        defn_ctrl = _make_def("Control prompt.")
        defn_trt = _make_def("Treatment prompt.")
        version_mgr = VersionManager(store, agent_id="ab-agent")
        v_ctrl = await version_mgr.create_version(defn_ctrl, version_tag="ctrl")
        v_trt = await version_mgr.create_version(defn_trt, version_tag="trt")
        await version_mgr.deploy(v_ctrl.version_id)

        await ab_mgr.start_test(
            "ab-agent",
            control_version_id=v_ctrl.version_id,
            treatment_version_id=v_trt.version_id,
            traffic_split=0.5,
        )

        router = VersionRouter(store, ab_manager=ab_mgr)
        result1 = await router.resolve("ab-agent", user_id="alice")
        result2 = await router.resolve("ab-agent", user_id="alice")
        # Same user always gets same result
        assert result1 is not None
        assert result2 is not None
        assert result1.system_prompt == result2.system_prompt

    @pytest.mark.asyncio
    async def test_traffic_split_zero_all_control(self) -> None:
        store, tracker = _make_infra()
        ab_mgr = ABTestManager(store, tracker)

        defn_ctrl = _make_def("Control.", "split-zero-agent")
        defn_trt = _make_def("Treatment.", "split-zero-agent")
        version_mgr = VersionManager(store, agent_id="split-zero-agent")
        v_ctrl = await version_mgr.create_version(defn_ctrl, version_tag="ctrl")
        v_trt = await version_mgr.create_version(defn_trt, version_tag="trt")
        await version_mgr.deploy(v_ctrl.version_id)

        await ab_mgr.start_test(
            "split-zero-agent",
            control_version_id=v_ctrl.version_id,
            treatment_version_id=v_trt.version_id,
            traffic_split=0.0,
        )

        router = VersionRouter(store, ab_manager=ab_mgr)
        for i in range(20):
            result = await router.resolve("split-zero-agent", user_id=f"user-{i}")
            assert result is not None
            assert result.system_prompt == "Control."

    @pytest.mark.asyncio
    async def test_traffic_split_one_all_treatment(self) -> None:
        store, tracker = _make_infra()
        ab_mgr = ABTestManager(store, tracker)

        defn_ctrl = _make_def("Control.", "split-one-agent")
        defn_trt = _make_def("Treatment.", "split-one-agent")
        version_mgr = VersionManager(store, agent_id="split-one-agent")
        v_ctrl = await version_mgr.create_version(defn_ctrl, version_tag="ctrl")
        v_trt = await version_mgr.create_version(defn_trt, version_tag="trt")
        await version_mgr.deploy(v_ctrl.version_id)

        await ab_mgr.start_test(
            "split-one-agent",
            control_version_id=v_ctrl.version_id,
            treatment_version_id=v_trt.version_id,
            traffic_split=1.0,
        )

        router = VersionRouter(store, ab_manager=ab_mgr)
        for i in range(20):
            result = await router.resolve("split-one-agent", user_id=f"user-{i}")
            assert result is not None
            assert result.system_prompt == "Treatment."

    @pytest.mark.asyncio
    async def test_traffic_split_approximate_distribution(self) -> None:
        store, tracker = _make_infra()
        ab_mgr = ABTestManager(store, tracker)
        target_split = 0.3

        defn_ctrl = _make_def("Control.", "dist-agent")
        defn_trt = _make_def("Treatment.", "dist-agent")
        version_mgr = VersionManager(store, agent_id="dist-agent")
        v_ctrl = await version_mgr.create_version(defn_ctrl, version_tag="ctrl")
        v_trt = await version_mgr.create_version(defn_trt, version_tag="trt")
        await version_mgr.deploy(v_ctrl.version_id)

        await ab_mgr.start_test(
            "dist-agent",
            control_version_id=v_ctrl.version_id,
            treatment_version_id=v_trt.version_id,
            traffic_split=target_split,
        )

        router = VersionRouter(store, ab_manager=ab_mgr)
        treatment_count = 0
        n = 1000
        for i in range(n):
            result = await router.resolve("dist-agent", user_id=f"synthetic-user-{i}")
            if result is not None and result.system_prompt == "Treatment.":
                treatment_count += 1

        fraction = treatment_count / n
        assert abs(fraction - target_split) < 0.05
