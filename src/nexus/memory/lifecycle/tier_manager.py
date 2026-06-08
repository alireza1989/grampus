"""LifecycleTierManager — hot/warm/cold promotion and demotion (MemOS, arXiv 2505.22101)."""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta
from typing import Any

from nexus.core.logging import get_logger
from nexus.memory.lifecycle.types import (
    LifecycleStats,
    MemoryTier,
    MemoryType,
    TierRecord,
)

_log = get_logger(__name__)

_WARM_TO_HOT_THRESHOLD_7D = 3
_COLD_TO_WARM_THRESHOLD_7D = 1
_HOT_SESSION_TTL_SECS = 3600
_WARM_TTL_DAYS = 7
_TIER_RECORD_ENTITY = "lifecycle_tier"
_HOT_INDEX_SUFFIX = "_hot_index"


class LifecycleTierManager:
    """Hot/warm/cold lifecycle management for memory records (MemOS, arXiv 2505.22101).

    Tracks access patterns via TierRecord metadata stored in Dapr.
    Promotion runs on access (lazy); demotion runs on a periodic sweep.

    Args:
        state_store: DaprStateStore for persisting TierRecords.
        agent_id: Scopes all records to this agent.
    """

    def __init__(self, state_store: Any, *, agent_id: str) -> None:
        self._store = state_store
        self._agent_id = agent_id

    async def record_access(
        self,
        record_id: str,
        memory_type: MemoryType,
    ) -> MemoryTier:
        """Record an access event. Runs lazy promotion if threshold crossed.

        Returns the current (possibly newly promoted) tier. Never raises.
        """
        with contextlib.suppress(Exception):
            record = await self._load_tier_record(record_id)
            if record is None:
                record = TierRecord(
                    record_id=record_id,
                    memory_type=memory_type,
                    agent_id=self._agent_id,
                )

            now = datetime.now(UTC)
            record = record.model_copy(
                update={
                    "access_count_total": record.access_count_total + 1,
                    "access_count_7d": _compute_7d_count(record, now),
                    "last_accessed": now,
                }
            )

            new_tier = self._should_promote(record)
            if new_tier is not None and new_tier != record.current_tier:
                record = record.model_copy(
                    update={
                        "current_tier": new_tier,
                        "promoted_at": now,
                    }
                )
                _log.debug(
                    "tier_promoted",
                    record_id=record_id,
                    new_tier=new_tier.value,
                )
                if new_tier == MemoryTier.HOT:
                    await self._add_to_hot_index(record_id)

            await self._save_tier_record(record)
            return record.current_tier

        return MemoryTier.COLD

    async def get_tier(self, record_id: str) -> MemoryTier:
        """Return current tier. Returns COLD if TierRecord not found."""
        with contextlib.suppress(Exception):
            record = await self._load_tier_record(record_id)
            if record is not None:
                return record.current_tier
        return MemoryTier.COLD

    async def get_hot_record_ids(self) -> list[str]:
        """Return all record_ids currently in HOT tier for this agent."""
        with contextlib.suppress(Exception):
            index = await self._load_hot_index()
            return list(index)
        return []

    async def sweep(self) -> LifecycleStats:
        """Demotion sweep: find stale HOT and WARM records, demote them.

        Returns stats summary. Never raises.
        """
        stats = LifecycleStats(agent_id=self._agent_id, last_run=datetime.now(UTC))
        with contextlib.suppress(Exception):
            hot_ids = await self._load_hot_index()
            now = datetime.now(UTC)
            hot_ttl_cutoff = now - timedelta(seconds=_HOT_SESSION_TTL_SECS)
            warm_cutoff = now - timedelta(days=_WARM_TTL_DAYS)
            demotions = 0
            new_hot_ids: list[str] = []

            for record_id in hot_ids:
                record = await self._load_tier_record(record_id)
                if record is None:
                    continue

                if record.current_tier == MemoryTier.HOT:
                    last = record.last_accessed
                    if last is None or last < hot_ttl_cutoff:
                        record = record.model_copy(
                            update={
                                "current_tier": MemoryTier.WARM,
                                "demoted_at": now,
                            }
                        )
                        await self._save_tier_record(record)
                        demotions += 1
                    else:
                        new_hot_ids.append(record_id)
                        stats.hot_count += 1

            await self._save_hot_index(new_hot_ids)

            all_records = await self._list_all_tier_records()
            for record in all_records:
                if record.current_tier == MemoryTier.WARM:
                    stats.warm_count += 1
                    if self._is_7d_window_stale(record) and (
                        record.last_accessed is None or record.last_accessed < warm_cutoff
                    ):
                        demoted = record.model_copy(
                            update={
                                "current_tier": MemoryTier.COLD,
                                "demoted_at": now,
                            }
                        )
                        await self._save_tier_record(demoted)
                        demotions += 1
                        stats.warm_count -= 1
                        stats.cold_count += 1
                elif record.current_tier == MemoryTier.COLD:
                    stats.cold_count += 1

            stats.total_demotions = demotions

        return stats

    async def _load_tier_record(self, record_id: str) -> TierRecord | None:
        key = f"{self._agent_id}:{record_id}"
        with contextlib.suppress(Exception):
            result, _ = await self._store.get(_TIER_RECORD_ENTITY, key, TierRecord)
            return result  # type: ignore[no-any-return]
        return None

    async def _save_tier_record(self, record: TierRecord) -> None:
        key = f"{self._agent_id}:{record.record_id}"
        await self._store.save(_TIER_RECORD_ENTITY, key, record)

    async def _load_hot_index(self) -> list[str]:
        key = f"{self._agent_id}{_HOT_INDEX_SUFFIX}"
        with contextlib.suppress(Exception):
            result, _ = await self._store.get(_TIER_RECORD_ENTITY, key, list)
            if isinstance(result, list):
                return result
        return []

    async def _save_hot_index(self, ids: list[str]) -> None:
        key = f"{self._agent_id}{_HOT_INDEX_SUFFIX}"
        await self._store.save(_TIER_RECORD_ENTITY, key, ids)

    async def _add_to_hot_index(self, record_id: str) -> None:
        index = await self._load_hot_index()
        if record_id not in index:
            index.append(record_id)
        await self._save_hot_index(index)

    async def _list_all_tier_records(self) -> list[TierRecord]:
        """Load all TierRecords for this agent from the hot index and warm set.

        This is a best-effort scan; only records we know about are returned.
        """
        hot_ids = await self._load_hot_index()
        records: list[TierRecord] = []
        for record_id in hot_ids:
            rec = await self._load_tier_record(record_id)
            if rec is not None:
                records.append(rec)
        return records

    def _should_promote(self, record: TierRecord) -> MemoryTier | None:
        """Return the tier to promote to, or None if no promotion warranted."""
        count_7d = record.access_count_7d
        if record.current_tier == MemoryTier.COLD and count_7d >= _COLD_TO_WARM_THRESHOLD_7D:
            return MemoryTier.WARM
        if record.current_tier == MemoryTier.WARM and count_7d >= _WARM_TO_HOT_THRESHOLD_7D:
            return MemoryTier.HOT
        return None

    def _is_7d_window_stale(self, record: TierRecord) -> bool:
        """Return True if last_accessed was more than 7 days ago."""
        if record.last_accessed is None:
            return True
        age = datetime.now(UTC) - record.last_accessed
        return age.total_seconds() > _WARM_TTL_DAYS * 86400


def _compute_7d_count(record: TierRecord, now: datetime) -> int:
    """Increment 7d count, but reset to 1 if window is stale."""
    if record.last_accessed is None:
        return 1
    age = now - record.last_accessed
    if age.total_seconds() > 7 * 86400:
        return 1
    return record.access_count_7d + 1
