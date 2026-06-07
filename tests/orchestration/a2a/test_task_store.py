"""Tests for NexusTaskStore."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from a2a.server.context import ServerCallContext
from a2a.types.a2a_pb2 import Task, TaskState


def _make_task(task_id: str = "task-1", state: int = TaskState.TASK_STATE_WORKING) -> Task:
    task = Task()
    task.id = task_id
    task.context_id = "ctx-1"
    task.status.state = state
    return task


def _make_call_context() -> ServerCallContext:
    return ServerCallContext()


async def test_save_and_get_task() -> None:
    from nexus.orchestration.a2a.task_store import NexusTaskStore

    store = NexusTaskStore()
    ctx = _make_call_context()
    task = _make_task("t1")

    await store.save(task, ctx)
    retrieved = await store.get("t1", ctx)

    assert retrieved is not None
    assert retrieved.id == "t1"


async def test_get_missing_returns_none() -> None:
    from nexus.orchestration.a2a.task_store import NexusTaskStore

    store = NexusTaskStore()
    ctx = _make_call_context()

    result = await store.get("nonexistent", ctx)
    assert result is None


async def test_delete_task() -> None:
    from nexus.orchestration.a2a.task_store import NexusTaskStore

    store = NexusTaskStore()
    ctx = _make_call_context()
    task = _make_task("t2")

    await store.save(task, ctx)
    await store.delete("t2", ctx)

    result = await store.get("t2", ctx)
    assert result is None


async def test_in_memory_fallback_when_no_state_store() -> None:
    from nexus.orchestration.a2a.task_store import NexusTaskStore

    store = NexusTaskStore(state_store=None)
    ctx = _make_call_context()
    task = _make_task("t3")

    await store.save(task, ctx)
    retrieved = await store.get("t3", ctx)
    assert retrieved is not None
    assert retrieved.id == "t3"


async def test_state_store_failure_does_not_raise() -> None:
    from nexus.orchestration.a2a.task_store import NexusTaskStore

    bad_store = MagicMock()
    bad_store.set = AsyncMock(side_effect=RuntimeError("dapr down"))
    bad_store.get = AsyncMock(side_effect=RuntimeError("dapr down"))

    store = NexusTaskStore(state_store=bad_store)
    ctx = _make_call_context()
    task = _make_task("t4")

    # Neither save nor get should raise even if the backing store explodes
    await store.save(task, ctx)
    result = await store.get("t4", ctx)
    # Falls back to in-memory
    assert result is not None


async def test_list_tasks_returns_stored() -> None:
    from a2a.types.a2a_pb2 import ListTasksRequest

    from nexus.orchestration.a2a.task_store import NexusTaskStore

    store = NexusTaskStore()
    ctx = _make_call_context()
    await store.save(_make_task("ta"), ctx)
    await store.save(_make_task("tb"), ctx)

    req = ListTasksRequest()
    response = await store.list(req, ctx)
    task_ids = {t.id for t in response.tasks}
    assert "ta" in task_ids
    assert "tb" in task_ids
