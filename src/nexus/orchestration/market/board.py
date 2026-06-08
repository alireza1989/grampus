"""TaskBoard — durable task and bid storage for market-based allocation."""

from __future__ import annotations

from typing import Any

from nexus.core.logging import get_logger
from nexus.orchestration.market.types import (
    AllocationStatus,
    Bid,
    TaskOutcome,
    TaskSpec,
)

_log = get_logger(__name__)


class TaskBoard:
    """Supervisor posts TaskSpecs; workers retrieve open tasks and submit bids.

    In-memory dicts provide fast within-session access; Dapr state provides
    durability across restarts.

    Task lifecycle: PENDING → BIDDING → ALLOCATED → COMPLETED / FAILED

    Args:
        state_store: Optional DaprStateStore. When None, board is in-memory only.
    """

    def __init__(self, state_store: Any | None = None) -> None:
        self._store = state_store
        self._tasks: dict[str, TaskSpec] = {}
        self._task_statuses: dict[str, AllocationStatus] = {}
        self._bids: dict[str, list[Bid]] = {}

    async def post_task(self, spec: TaskSpec) -> str:
        """Post a new task to the board.

        Args:
            spec: The TaskSpec to post.

        Returns:
            The task_id of the posted task.
        """
        self._tasks[spec.task_id] = spec
        self._task_statuses[spec.task_id] = AllocationStatus.PENDING
        self._bids[spec.task_id] = []
        if self._store is not None:
            await self._store.save("task", spec.task_id, spec)
        _log.debug("task_posted", task_id=spec.task_id)
        return spec.task_id

    async def get_task(self, task_id: str) -> TaskSpec | None:
        """Retrieve a TaskSpec by ID.

        Args:
            task_id: The task to look up.

        Returns:
            The TaskSpec, or None if not found.
        """
        if task_id in self._tasks:
            return self._tasks[task_id]
        if self._store is not None:
            stored, _ = await self._store.get("task", task_id, TaskSpec)
            if stored is not None:
                loaded: TaskSpec = stored
                self._tasks[task_id] = loaded
            return stored if stored is not None else None
        return None

    async def submit_bid(self, bid: Bid) -> None:
        """Worker agent submits a bid for a task.

        Args:
            bid: The Bid to submit.
        """
        if bid.task_id not in self._bids:
            self._bids[bid.task_id] = []
        existing_ids = {b.bid_id for b in self._bids[bid.task_id]}
        if bid.bid_id not in existing_ids:
            self._bids[bid.task_id].append(bid)
        if self._store is not None:
            from pydantic import BaseModel, Field

            class _BidList(BaseModel):
                bids: list[Bid] = Field(default_factory=list)

            await self._store.save("bids", bid.task_id, _BidList(bids=self._bids[bid.task_id]))
        _log.debug("bid_submitted", task_id=bid.task_id, agent_id=bid.agent_id)

    async def get_bids_for_task(self, task_id: str) -> list[Bid]:
        """Return all bids submitted for a task.

        Args:
            task_id: The task to look up bids for.

        Returns:
            List of Bid objects (may be empty).
        """
        return list(self._bids.get(task_id, []))

    async def update_task_status(self, task_id: str, status: AllocationStatus) -> None:
        """Update the lifecycle status of a task.

        Args:
            task_id: The task to update.
            status: The new AllocationStatus.
        """
        self._task_statuses[task_id] = status
        _log.debug("task_status_updated", task_id=task_id, status=status)

    async def mark_outcome(self, outcome: TaskOutcome) -> None:
        """Record the final outcome and update task status accordingly.

        Args:
            outcome: TaskOutcome describing what happened.
        """
        new_status = (
            AllocationStatus.COMPLETED if outcome.actual_success else AllocationStatus.FAILED
        )
        await self.update_task_status(outcome.task_id, new_status)
        _log.debug(
            "task_outcome_recorded",
            task_id=outcome.task_id,
            agent_id=outcome.agent_id,
            success=outcome.actual_success,
        )
