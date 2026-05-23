"""Crew — multi-agent coordination: sequential, parallel, and hierarchical patterns."""

from __future__ import annotations

import asyncio
import json
import time
from enum import StrEnum
from typing import Any, cast

from pydantic import BaseModel

from nexus.core.errors import OrchestrationError
from nexus.core.logging import get_logger
from nexus.core.types import AgentDefinition, ExecutionResult

_log = get_logger(__name__)


class CrewPattern(StrEnum):
    """Execution pattern for a crew of agents."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"


class CrewMember(BaseModel):
    """A single member of a crew, pairing an AgentDefinition with its runner.

    Args:
        agent_def: Blueprint for this agent.
        runner: AgentRunner (duck-typed as Any to avoid circular imports).
        role: Semantic label — "supervisor", "worker", or custom.
    """

    agent_def: AgentDefinition
    runner: Any
    role: str

    model_config = {"arbitrary_types_allowed": True}


class CrewResult(BaseModel):
    """Aggregated output from a crew run."""

    outputs: dict[str, str]
    total_cost_usd: float
    duration_seconds: float
    pattern: CrewPattern


class Crew:
    """Coordinates multiple AgentRunner instances.

    Args:
        members: List of crew members (each has its own AgentRunner).
        pattern: Execution pattern (sequential, parallel, hierarchical).
        shared_state_store: Optional Dapr state store for cross-agent shared state.
        lock: Optional DaprLock for coordinating shared state writes.
        session_id: Shared session identifier for the crew run.
    """

    def __init__(
        self,
        members: list[CrewMember],
        *,
        pattern: CrewPattern = CrewPattern.SEQUENTIAL,
        shared_state_store: Any | None = None,
        lock: Any | None = None,
        session_id: str,
    ) -> None:
        self._members = members
        self._pattern = pattern
        self._shared_state_store = shared_state_store
        self._lock = lock
        self._session_id = session_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, initial_input: str) -> CrewResult:
        """Execute the crew with the configured pattern.

        Args:
            initial_input: Prompt or task passed to the first agent (sequential)
                or all agents (parallel / supervisor in hierarchical).

        Returns:
            CrewResult with per-agent outputs, total cost, duration, and pattern.

        Raises:
            OrchestrationError: code="CREW_MEMBER_FAILED" if any member raises.
        """
        start = time.monotonic()
        dispatch = {
            CrewPattern.SEQUENTIAL: self._run_sequential,
            CrewPattern.PARALLEL: self._run_parallel,
            CrewPattern.HIERARCHICAL: self._run_hierarchical,
        }
        outputs = await dispatch[self._pattern](initial_input)
        duration = time.monotonic() - start
        total_cost = self._total_cost()
        _log.debug(
            "crew_run_complete",
            pattern=self._pattern,
            members=len(self._members),
            cost=total_cost,
        )
        return CrewResult(
            outputs=outputs,
            total_cost_usd=total_cost,
            duration_seconds=duration,
            pattern=self._pattern,
        )

    # ------------------------------------------------------------------
    # Pattern implementations
    # ------------------------------------------------------------------

    async def _run_sequential(self, initial_input: str) -> dict[str, str]:
        outputs: dict[str, str] = {}
        current_input = initial_input
        for member in self._members:
            result = await self._run_member(member, current_input)
            output = result.output or ""
            outputs[member.agent_def.name] = output
            current_input = output
        return outputs

    async def _run_parallel(self, initial_input: str) -> dict[str, str]:
        async def _run(m: CrewMember) -> tuple[str, str]:
            result = await self._run_member(m, initial_input)
            return m.agent_def.name, result.output or ""

        pairs = cast(
            list[tuple[str, str]],
            await asyncio.gather(*[_run(m) for m in self._members]),
        )
        return dict(pairs)

    async def _run_hierarchical(self, initial_input: str) -> dict[str, str]:
        supervisor = self._members[0]
        workers = self._members[1:]

        sup_result = await self._run_member(supervisor, initial_input)
        sup_output = sup_result.output or ""

        task_assignments = self._parse_supervisor_output(sup_output, workers)
        worker_results = await self._dispatch_workers(task_assignments, workers)

        summary = _build_worker_summary(worker_results)
        final_result = await self._run_member(supervisor, summary)

        outputs: dict[str, str] = {supervisor.agent_def.name: final_result.output or ""}
        outputs.update(worker_results)
        return outputs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _run_member(self, member: CrewMember, user_input: str) -> ExecutionResult:
        try:
            result: ExecutionResult = await member.runner.run(
                member.agent_def, user_input, session_id=self._session_id
            )
            return result
        except Exception as exc:
            raise OrchestrationError(
                f"Crew member '{member.agent_def.name}' failed: {exc}",
                code="CREW_MEMBER_FAILED",
                hint="Check that agent names are unique and no agent depends on itself.",
            ) from exc

    def _parse_supervisor_output(self, output: str, workers: list[CrewMember]) -> dict[str, str]:
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
            raise ValueError("Not a dict")
        except (json.JSONDecodeError, ValueError):
            _log.debug("crew_hierarchical_fallback", reason="invalid_json")
            return {w.agent_def.name: output for w in workers}

    async def _dispatch_workers(
        self,
        task_assignments: dict[str, str],
        workers: list[CrewMember],
    ) -> dict[str, str]:
        worker_by_name = {w.agent_def.name: w for w in workers}

        async def _run_assigned(name: str, task: str) -> tuple[str, str]:
            if name not in worker_by_name:
                return name, ""
            result = await self._run_member(worker_by_name[name], task)
            return name, result.output or ""

        pairs = await asyncio.gather(*[_run_assigned(n, t) for n, t in task_assignments.items()])
        return dict(pairs)

    def _total_cost(self) -> float:
        total = 0.0
        for member in self._members:
            fn = getattr(member.runner, "cost_summary", None)
            if fn is None:
                continue
            summary = fn()
            if summary is not None:
                total += summary.total_cost_usd
        return total


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _build_worker_summary(worker_results: dict[str, str]) -> str:
    lines = ["Worker results:"]
    for name, output in worker_results.items():
        lines.append(f"{name}: {output}")
    return "\n".join(lines)
