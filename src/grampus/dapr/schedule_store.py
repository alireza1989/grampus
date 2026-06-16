"""Persistence layer for ScheduleConfig records using Dapr state store."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from grampus.core.logging import get_logger

_log = get_logger(__name__)


class ScheduleConfig(BaseModel):
    """Persistent record of one scheduled agent job."""

    name: str
    cron: str
    input_text: str
    session_prefix: str = "sched"
    enabled: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_triggered_at: datetime | None = None
    trigger_count: int = 0


class _IndexRecord(BaseModel):
    """Internal index of all schedule names stored in Dapr state."""

    names: list[str] = Field(default_factory=list)


class ScheduleStore:
    """CRUD for ScheduleConfig backed by DaprStateStore.

    Args:
        state_store: A DaprStateStore instance (namespace-scoped).
                     When None, operates in-memory only (for testing / no-Dapr dev).
    """

    _ENTITY = "schedules"
    _INDEX_KEY = "_index"

    def __init__(self, state_store: Any | None = None) -> None:
        self._store = state_store
        self._mem: dict[str, ScheduleConfig] = {}

    async def save(self, config: ScheduleConfig) -> None:
        """Persist a ScheduleConfig.

        Args:
            config: The schedule configuration to save.
        """
        self._mem[config.name] = config
        if self._store is not None:
            await self._store.save(self._ENTITY, config.name, config)
            await self._save_index()

    async def get(self, name: str) -> ScheduleConfig | None:
        """Retrieve a ScheduleConfig by name.

        Args:
            name: Job name to look up.

        Returns:
            ScheduleConfig if found, None otherwise.
        """
        if name in self._mem:
            return self._mem[name]
        if self._store is None:
            return None
        raw, _ = await self._store.get(self._ENTITY, name, ScheduleConfig)
        result: ScheduleConfig | None = raw
        if result is not None:
            self._mem[name] = result
        return result

    async def delete(self, name: str) -> bool:
        """Delete a ScheduleConfig.

        Args:
            name: Job name to delete.

        Returns:
            True if the record existed and was deleted, False if not found.
        """
        found = name in self._mem or (await self.get(name)) is not None
        self._mem.pop(name, None)
        if self._store is not None:
            await self._store.delete(self._ENTITY, name)
            await self._save_index()
        return found

    async def list_all(self) -> list[ScheduleConfig]:
        """Return all stored ScheduleConfig records.

        Returns:
            List of all schedule configurations.
        """
        if self._store is None:
            return list(self._mem.values())
        index, _ = await self._store.get(self._ENTITY, self._INDEX_KEY, _IndexRecord)
        names = index.names if index else []
        results: list[ScheduleConfig] = []
        for name in names:
            cfg = await self.get(name)
            if cfg is not None:
                results.append(cfg)
        return results

    async def _save_index(self) -> None:
        if self._store is None:
            return
        names = list(self._mem.keys())
        await self._store.save(self._ENTITY, self._INDEX_KEY, _IndexRecord(names=names))
