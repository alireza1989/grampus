"""MarketCrew — Crew subclass with optional market-based task allocation."""

from __future__ import annotations

import uuid
from typing import Any, cast

from nexus.core.errors import MarketAllocationError
from nexus.core.logging import get_logger
from nexus.core.types import ExecutionResult
from nexus.orchestration.crew import Crew, CrewMember, CrewPattern
from nexus.orchestration.market.allocator import MarketAllocator
from nexus.orchestration.market.types import (
    AllocationStatus,
    TaskOutcome,
    TaskSpec,
)

_log = get_logger(__name__)


class MarketCrew(Crew):
    """Extends Crew with optional market-based task allocation.

    When use_market=True, each task is posted to the TaskBoard, MarketAllocator
    selects the best-fit worker agent, the selected agent executes via its
    AgentRunner, and the outcome is reported back to ReputationTracker.

    When use_market=False (default), falls back to standard Crew execution
    with no market overhead — suitable for small crews.

    Args:
        members: List of crew members (each has its own AgentRunner).
        pattern: Execution pattern (sequential, parallel, hierarchical).
        shared_state_store: Optional Dapr state store for cross-agent shared state.
        lock: Optional DaprLock for coordinating shared state writes.
        session_id: Shared session identifier for the crew run.
        allocator: MarketAllocator instance. Required when use_market=True.
        use_market: Enable market-based allocation. Default False.
    """

    def __init__(
        self,
        members: list[CrewMember],
        *,
        pattern: CrewPattern = CrewPattern.SEQUENTIAL,
        shared_state_store: Any | None = None,
        lock: Any | None = None,
        session_id: str,
        allocator: MarketAllocator | None = None,
        use_market: bool = False,
    ) -> None:
        super().__init__(
            members,
            pattern=pattern,
            shared_state_store=shared_state_store,
            lock=lock,
            session_id=session_id,
        )
        self._allocator = allocator
        self._use_market = use_market
        self._member_by_agent_id = {m.agent_def.name: m for m in members}

    async def run_task_with_market(
        self,
        task_description: str,
        required_skills: list[str],
        budget_usd: float | None = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Post task to board, allocate best worker, run it, report outcome.

        Args:
            task_description: Natural language task description.
            required_skills: Skills required for capability filtering.
            budget_usd: Hard cost cap for the task, or None for unlimited.
            **kwargs: Extra metadata forwarded to TaskSpec.

        Returns:
            ExecutionResult from the winning agent.

        Raises:
            MarketAllocationError: When no capable agent wins allocation.
        """
        if not self._use_market or self._allocator is None:
            return await self._run_first_member(task_description)

        spec = TaskSpec(
            task_id=str(uuid.uuid4()),
            description=task_description,
            required_skills=required_skills,
            budget_usd=budget_usd,
            metadata=kwargs,
        )
        result = await self._allocator.allocate(spec)

        if result.status != AllocationStatus.ALLOCATED or result.winning_agent_id is None:
            raise MarketAllocationError(
                f"Market allocation rejected for task '{task_description[:60]}': "
                f"{result.reject_reason or 'no capable bidder'}",
                code="MARKET_ALLOCATION_REJECTED",
                details={"task_id": spec.task_id, "status": result.status},
            )

        member = self._member_by_agent_id.get(result.winning_agent_id)
        if member is None:
            raise MarketAllocationError(
                f"Winning agent '{result.winning_agent_id}' is not a crew member.",
                code="MARKET_WINNER_NOT_MEMBER",
            )

        exec_result: ExecutionResult = cast(
            ExecutionResult,
            await member.runner.run(
                member.agent_def, task_description, session_id=self._session_id
            ),
        )

        actual_cost = 0.0
        if exec_result.token_usage:
            actual_cost = exec_result.token_usage.cost_usd

        outcome = TaskOutcome(
            task_id=spec.task_id,
            agent_id=result.winning_agent_id,
            actual_success=exec_result.status.value not in ("failed",),
            actual_cost_usd=actual_cost,
            actual_steps=exec_result.steps_taken,
        )
        await self._allocator.report_outcome(outcome)
        _log.info(
            "market_task_complete",
            task_id=spec.task_id,
            agent=result.winning_agent_id,
            success=outcome.actual_success,
        )
        return exec_result

    async def _run_first_member(self, task: str) -> ExecutionResult:
        """Fallback: run the first crew member when market is disabled."""
        if not self._members:
            raise MarketAllocationError(
                "MarketCrew has no members.",
                code="MARKET_NO_MEMBERS",
            )
        return await self._run_member(self._members[0], task)
