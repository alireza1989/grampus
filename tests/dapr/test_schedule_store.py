"""Unit tests for ScheduleStore (in-memory mode — no Dapr required)."""

from __future__ import annotations

import pytest

from nexus.dapr.schedule_store import ScheduleConfig, ScheduleStore


def _make_config(name: str = "test-job") -> ScheduleConfig:
    return ScheduleConfig(name=name, cron="@daily", input_text="go")


@pytest.mark.asyncio
async def test_save_and_get() -> None:
    store = ScheduleStore(state_store=None)
    config = _make_config("my-job")
    await store.save(config)
    result = await store.get("my-job")
    assert result is not None
    assert result.name == "my-job"
    assert result.cron == "@daily"


@pytest.mark.asyncio
async def test_get_missing_returns_none() -> None:
    store = ScheduleStore(state_store=None)
    result = await store.get("missing")
    assert result is None


@pytest.mark.asyncio
async def test_delete_found() -> None:
    store = ScheduleStore(state_store=None)
    config = _make_config("del-job")
    await store.save(config)
    deleted = await store.delete("del-job")
    assert deleted is True
    assert await store.get("del-job") is None


@pytest.mark.asyncio
async def test_delete_not_found() -> None:
    store = ScheduleStore(state_store=None)
    deleted = await store.delete("missing")
    assert deleted is False


@pytest.mark.asyncio
async def test_list_all_empty() -> None:
    store = ScheduleStore(state_store=None)
    result = await store.list_all()
    assert result == []


@pytest.mark.asyncio
async def test_list_all_multiple() -> None:
    store = ScheduleStore(state_store=None)
    await store.save(_make_config("job-a"))
    await store.save(_make_config("job-b"))
    await store.save(_make_config("job-c"))
    result = await store.list_all()
    assert len(result) == 3
    names = {c.name for c in result}
    assert names == {"job-a", "job-b", "job-c"}


@pytest.mark.asyncio
async def test_save_overwrites_existing() -> None:
    store = ScheduleStore(state_store=None)
    config = _make_config("job-x")
    await store.save(config)
    updated = config.model_copy(update={"cron": "@hourly"})
    await store.save(updated)
    result = await store.get("job-x")
    assert result is not None
    assert result.cron == "@hourly"


def test_schedule_config_defaults() -> None:
    config = ScheduleConfig(name="x", cron="@daily", input_text="go")
    assert config.enabled is True
    assert config.trigger_count == 0
    assert config.session_prefix == "sched"
    assert config.last_triggered_at is None


def test_schedule_config_round_trip() -> None:
    config = ScheduleConfig(name="round", cron="0 8 * * 1", input_text="hello")
    data = config.model_dump_json()
    restored = ScheduleConfig.model_validate_json(data)
    assert restored.name == config.name
    assert restored.cron == config.cron
    assert restored.input_text == config.input_text
