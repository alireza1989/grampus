"""GrampusTaskStore — TaskStore implementation backed by Dapr state with in-memory fallback."""

from __future__ import annotations

from typing import Any

try:
    from a2a.server.context import ServerCallContext
    from a2a.server.tasks import TaskStore
    from a2a.types.a2a_pb2 import ListTasksRequest, ListTasksResponse, Task

    _HAS_A2A = True
except ImportError:  # pragma: no cover
    _HAS_A2A = False
    TaskStore = object
    ServerCallContext = object

from grampus.core.logging import get_logger

_log = get_logger(__name__)
_DAPR_KEY_PREFIX = "a2a:task:"


class GrampusTaskStore(TaskStore):  # type: ignore[misc]
    """TaskStore persisting to Dapr state with an in-memory fallback.

    When ``state_store`` is None (or when a Dapr write fails), all task data
    is kept in a process-local dict. A store failure never raises — it logs and
    continues so the server stays operational.

    Args:
        state_store: Optional DaprStateStore for durable persistence.
    """

    def __init__(self, state_store: Any | None = None) -> None:
        self._state_store = state_store
        self._memory: dict[str, bytes] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, task_id: str) -> str:
        return f"{_DAPR_KEY_PREFIX}{task_id}"

    def _serialize(self, task: Task) -> bytes:
        return bytes(task.SerializeToString())

    def _deserialize(self, data: bytes) -> Task:
        task: Task = Task()
        task.ParseFromString(data)
        return task

    # ------------------------------------------------------------------
    # TaskStore interface
    # ------------------------------------------------------------------

    async def save(self, task: Task, context: ServerCallContext) -> None:
        """Persist a task; silently falls back to in-memory on Dapr failure."""
        key = self._key(task.id)
        data = self._serialize(task)
        self._memory[key] = data

        if self._state_store is not None:
            try:
                await self._state_store.set(key, task.SerializeToString())
            except Exception as exc:
                _log.warning("a2a_task_store_dapr_save_failed", task_id=task.id, error=str(exc))

    async def get(self, task_id: str, context: ServerCallContext) -> Task | None:
        """Retrieve a task by ID; returns None if not found."""
        key = self._key(task_id)

        if self._state_store is not None:
            try:
                data = await self._state_store.get(key)
                if data:
                    return self._deserialize(data if isinstance(data, bytes) else data.encode())
            except Exception as exc:
                _log.warning("a2a_task_store_dapr_get_failed", task_id=task_id, error=str(exc))

        data = self._memory.get(key)
        if data is None:
            return None
        return self._deserialize(data)

    async def list(self, params: ListTasksRequest, context: ServerCallContext) -> ListTasksResponse:
        """List all tasks currently held in memory."""
        response = ListTasksResponse()
        for raw in self._memory.values():
            task = self._deserialize(raw)
            response.tasks.append(task)
        return response

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        """Remove a task from storage."""
        key = self._key(task_id)
        self._memory.pop(key, None)

        if self._state_store is not None:
            try:
                await self._state_store.delete(key)
            except Exception as exc:
                _log.warning("a2a_task_store_dapr_delete_failed", task_id=task_id, error=str(exc))
