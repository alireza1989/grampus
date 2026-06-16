"""PlanningRunner — top-level orchestrator for long-horizon task execution."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from grampus.core.logging import get_logger
from grampus.orchestration.planning.executor import SubGoalExecutor
from grampus.orchestration.planning.lookahead import LookaheadSimulator
from grampus.orchestration.planning.planner import Planner
from grampus.orchestration.planning.replanner import Replanner
from grampus.orchestration.planning.types import (
    Plan,
    PlanningConfig,
    PlanResult,
    SubGoal,
    SubGoalStatus,
    build_completed_summary,
)
from grampus.orchestration.planning.verifier import PostconditionVerifier

_log = get_logger(__name__)

SYNTHESIS_PROMPT = """You have completed a multi-step task. Synthesize the results into a final comprehensive answer.

Original task: {task}

Completed steps and their outputs:
{subgoal_outputs}

Provide a complete, well-organized final answer."""


class PlanningRunner:
    """Orchestrates long-horizon task execution via structured planning.

    Architecture:
    1. Estimate complexity. Below threshold → delegate to AgentRunner directly.
    2. Call Planner.create_plan() to get a SubGoal DAG.
    3. Topological sort → execution waves.
    4. Execute each wave: independent subgoals run via asyncio.gather.
    5. After each subgoal: verify. On FAIL: try fallback, then trigger Replanner.
    6. On replan: re-sort new plan, continue from first new wave.
    7. After all subgoals complete: synthesize final output via one LLM call.
    8. Return PlanResult.

    Args:
        agent_runner: AgentRunner instance for subgoal execution (duck-typed).
        model_client: LLM client for planning/verification/synthesis calls.
        model_id: Model identifier for planning calls.
        config: PlanningConfig tuning parameters.
        cost_tracker: Optional cost tracker shared across all calls.
        tracer: Optional OTEL tracer for planning.run spans.
    """

    def __init__(
        self,
        agent_runner: Any,
        model_client: Any,
        model_id: str,
        *,
        config: PlanningConfig | None = None,
        cost_tracker: Any | None = None,
        tracer: Any | None = None,
    ) -> None:
        self._runner = agent_runner
        self._model_client = model_client
        self._model_id = model_id
        self._config = config or PlanningConfig()
        self._cost_tracker = cost_tracker
        self._tracer = tracer

        self._planner = Planner(model_client, model_id, cost_tracker=cost_tracker, tracer=tracer)
        self._verifier = PostconditionVerifier(model_client, model_id, cost_tracker=cost_tracker)
        self._lookahead: LookaheadSimulator | None = None
        if self._config.enable_lookahead:
            self._lookahead = LookaheadSimulator(
                model_client,
                model_id,
                n_paths=self._config.lookahead_paths,
                cost_tracker=cost_tracker,
            )
        self._replanner = Replanner(
            model_client, model_id, cost_tracker=cost_tracker, tracer=tracer
        )
        self._executor = SubGoalExecutor(
            agent_runner,
            self._verifier,
            self._lookahead,
            tracer=tracer,
            cost_tracker=cost_tracker,
        )

    async def run(
        self,
        task: str,
        agent_def: Any,
        *,
        tool_names: list[str] | None = None,
        memory_context: str = "",
    ) -> PlanResult:
        """Execute the full planning + execution pipeline.

        Args:
            task: User task description.
            agent_def: AgentDefinition for subgoal execution.
            tool_names: Available tool names for planning hints.
            memory_context: Optional memory context for the planner.

        Returns:
            PlanResult with outcome, subgoal results, token usage, and timing.
        """
        start = time.monotonic()
        tools = tool_names or []

        complexity = await self._planner.estimate_complexity(task)
        _log.info("planning.complexity", task_len=len(task), estimate=complexity)

        if complexity <= self._config.complexity_threshold:
            return await self._run_direct(task, agent_def, tools, start)

        plan = await self._planner.create_plan(
            task, tool_names=tools, memory_context=memory_context, config=self._config
        )
        return await self._execute_plan(plan, task, agent_def, tools, start)

    async def _run_direct(
        self,
        task: str,
        agent_def: Any,
        tool_names: list[str],
        start: float,
    ) -> PlanResult:
        """Delegate directly to AgentRunner without planning (low-complexity path).

        Args:
            task: User task description.
            agent_def: AgentDefinition for execution.
            tool_names: Available tool names.
            start: Monotonic start time for duration calculation.

        Returns:
            PlanResult wrapping the direct AgentRunner result.
        """
        import uuid
        from datetime import UTC, datetime

        from grampus.orchestration.planning.types import Plan, SubGoal

        session_id = f"direct-{uuid.uuid4().hex[:8]}"
        result = await self._runner.run(agent_def, task, session_id=session_id)

        dummy_sg = SubGoal(
            id="execute_task",
            description=task,
            success_criterion="Task completed",
            status=SubGoalStatus.COMPLETED,
            output_summary=result.output or "",
        )
        dummy_plan = Plan(
            task=task,
            subgoals=[dummy_sg],
            total_estimated_steps=1,
            created_at=datetime.now(UTC),
        )
        return PlanResult(
            task=task,
            plan=dummy_plan,
            final_output=result.output or "",
            completed_subgoals=["execute_task"],
            failed_subgoals=[],
            replans_triggered=0,
            total_token_usage=result.token_usage,
            duration_seconds=time.monotonic() - start,
            success=True,
        )

    async def _execute_plan(
        self,
        plan: Plan,
        task: str,
        agent_def: Any,
        tool_names: list[str],
        start: float,
    ) -> PlanResult:
        """Execute a plan through its waves, replanning on failure.

        Args:
            plan: The initial Plan from the planner.
            task: User task description.
            agent_def: AgentDefinition for subgoal execution.
            tool_names: Available tool names.
            start: Monotonic start time.

        Returns:
            PlanResult with full execution outcome.
        """
        completed: list[SubGoal] = []
        failed: list[SubGoal] = []
        total_usage = None
        replans_triggered = 0
        current_plan = plan

        waves = self._planner._topological_sort(current_plan.subgoals)

        wave_idx = 0
        while wave_idx < len(waves):
            wave = [sg for sg in waves[wave_idx] if sg.status == SubGoalStatus.PENDING]
            if not wave:
                wave_idx += 1
                continue

            updated = await self._execute_wave(wave, task, completed, tool_names, agent_def)

            new_failures = [sg for sg in updated if sg.status == SubGoalStatus.FAILED]
            new_completed = [sg for sg in updated if sg.status == SubGoalStatus.COMPLETED]
            completed.extend(new_completed)

            if new_failures:
                failed_sg = new_failures[0]
                failed.append(failed_sg)
                try:
                    current_plan = await self._replanner.replan(
                        current_plan, failed_sg, completed, self._config
                    )
                    replans_triggered += 1
                    waves = self._planner._topological_sort(current_plan.subgoals)
                    wave_idx = 0
                    continue
                except Exception as exc:
                    _log.error("planning.replan_failed", error=str(exc))
                    raise

            wave_idx += 1

        final_output = await self._synthesize_output(task, completed)
        return PlanResult(
            task=task,
            plan=current_plan,
            final_output=final_output,
            completed_subgoals=[sg.id for sg in completed],
            failed_subgoals=[sg.id for sg in failed],
            replans_triggered=replans_triggered,
            total_token_usage=total_usage,
            duration_seconds=time.monotonic() - start,
            success=len(completed) > 0 and len(failed) == 0,
        )

    async def _execute_wave(
        self,
        wave: list[SubGoal],
        task: str,
        completed: list[SubGoal],
        tool_names: list[str],
        agent_def: Any,
    ) -> list[SubGoal]:
        """Execute all subgoals in a wave concurrently via asyncio.gather.

        Args:
            wave: Subgoals to execute (independent, no mutual dependencies).
            task: Global task description.
            completed: Already-completed subgoals for context.
            tool_names: Available tool names.
            agent_def: AgentDefinition for execution.

        Returns:
            Updated subgoal list with statuses set.
        """
        if self._config.enable_parallel_subgoals and len(wave) > 1:
            results = await asyncio.gather(
                *[self._executor.execute(sg, task, completed, tool_names, agent_def) for sg in wave]
            )
            return list(results)
        else:
            updated = []
            for sg in wave:
                result = await self._executor.execute(sg, task, completed, tool_names, agent_def)
                updated.append(result)
                if result.status == SubGoalStatus.COMPLETED:
                    completed = completed + [result]
            return updated

    async def _synthesize_output(
        self,
        task: str,
        completed_subgoals: list[SubGoal],
    ) -> str:
        """Combine all subgoal outputs into a coherent final answer.

        Args:
            task: Original user task.
            completed_subgoals: Successfully completed subgoals with output_summary filled.

        Returns:
            Synthesized final answer string.
        """
        from grampus.core.types import Message, Role

        if not completed_subgoals:
            return "No subgoals completed successfully."

        subgoal_outputs = "\n".join(
            f"{i + 1}. [{sg.id}] {sg.description}\n   Output: {sg.output_summary}"
            for i, sg in enumerate(completed_subgoals)
        )
        prompt = SYNTHESIS_PROMPT.format(task=task, subgoal_outputs=subgoal_outputs)
        try:
            response = await self._model_client.complete(
                messages=[Message(role=Role.USER, content=prompt)],
                model=self._model_id,
            )
            if self._cost_tracker and response.token_usage:
                await self._cost_tracker.record(response.token_usage, step_name="synthesize")
            return response.content or ""
        except Exception:  # noqa: BLE001
            return build_completed_summary(completed_subgoals)
