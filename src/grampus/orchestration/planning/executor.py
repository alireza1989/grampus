"""SubGoalExecutor — executes a single subgoal with scoped context."""

from __future__ import annotations

from typing import Any

from grampus.core.logging import get_logger
from grampus.orchestration.planning.types import (
    SubGoal,
    SubGoalStatus,
    VerificationResult,
    build_completed_summary,
)

_log = get_logger(__name__)

SUBGOAL_SYSTEM_PROMPT = """You are executing one step of a larger task. Focus ONLY on the current subgoal.

Global task: {task}

Completed subgoals summary:
{completed_summary}

Current subgoal: {description}
Success criterion: {success_criterion}
{approach_hint}
Use the available tools to complete this subgoal. When done, provide a concise \
summary of what you accomplished and the key output (1-2 sentences)."""


class SubGoalExecutor:
    """Executes a single subgoal using AgentRunner with scoped context.

    TDP key design: the executor only sees global task + completed summaries +
    current subgoal, NOT the full conversation history. This achieves the 82%
    token reduction from Task-Decoupled Planning (arXiv 2601.07577).

    Steps per subgoal:
    1. If lookahead enabled: get approach hint from LookaheadSimulator.
    2. Build scoped messages: system prompt with context + user message.
    3. Call AgentRunner.run() with a fresh session (no prior state).
    4. Extract output_summary from ExecutionResult.
    5. Call PostconditionVerifier.verify().
    6. If PARTIAL and attempts < max_retries: retry with failure feedback.
    7. If FAIL: check fallback_strategy; retry once with fallback hint.
    8. Return updated SubGoal with status/output_summary/attempts set.

    Args:
        runner: AgentRunner instance for subgoal execution (duck-typed).
        verifier: PostconditionVerifier to check success criteria.
        lookahead: Optional LookaheadSimulator; None disables lookahead.
        tracer: Optional OTEL tracer for planning.subgoal spans.
        cost_tracker: Optional cost tracker for recording usage.
    """

    def __init__(
        self,
        runner: Any,
        verifier: Any,
        lookahead: Any | None,
        *,
        tracer: Any | None = None,
        cost_tracker: Any | None = None,
    ) -> None:
        self._runner = runner
        self._verifier = verifier
        self._lookahead = lookahead
        self._tracer = tracer
        self._cost_tracker = cost_tracker

    async def execute(
        self,
        subgoal: SubGoal,
        task: str,
        completed_subgoals: list[SubGoal],
        tool_names: list[str],
        agent_def: Any,
    ) -> SubGoal:
        """Execute subgoal and return it with updated status, output_summary, attempts.

        Args:
            subgoal: SubGoal to execute.
            task: Global task description.
            completed_subgoals: Already-completed subgoals for context building.
            tool_names: Available tool names (used by lookahead).
            agent_def: AgentDefinition passed to AgentRunner.run().

        Returns:
            Updated SubGoal with status COMPLETED or FAILED.
        """
        sg = subgoal.model_copy(deep=True)
        sg.status = SubGoalStatus.RUNNING
        completed_summary = build_completed_summary(completed_subgoals)

        approach_hint = ""
        if self._lookahead is not None:
            approach_hint = await self._lookahead.select_approach(
                task, sg, completed_summary, tool_names
            )

        used_fallback = False
        for attempt in range(sg.max_retries + 2):
            sg.attempts = attempt + 1
            hint_line = f"\nRecommended approach: {approach_hint}" if approach_hint else ""

            output = await self._run_once(sg, task, completed_summary, agent_def, hint_line)
            result, reason = await self._verifier.verify(sg, output)

            _log.debug(
                "executor.verify",
                subgoal_id=sg.id,
                attempt=sg.attempts,
                result=str(result),
            )

            if result == VerificationResult.PASS:
                sg.status = SubGoalStatus.COMPLETED
                sg.output_summary = output[:400] if output else reason
                return sg

            if result == VerificationResult.PARTIAL and attempt < sg.max_retries:
                approach_hint = f"Previous attempt partially succeeded. {reason}. Try again."
                continue

            if result == VerificationResult.FAIL and not used_fallback and sg.fallback_strategy:
                approach_hint = (
                    f"Primary approach failed. Use this strategy: {sg.fallback_strategy}"
                )
                used_fallback = True
                continue

            sg.failure_reason = reason
            sg.status = SubGoalStatus.FAILED
            return sg

        sg.failure_reason = "Max retries reached without passing verification"
        sg.status = SubGoalStatus.FAILED
        return sg

    async def _run_once(
        self,
        subgoal: SubGoal,
        task: str,
        completed_summary: str,
        agent_def: Any,
        approach_hint: str,
    ) -> str:
        """Run the AgentRunner for one attempt and return the output string."""
        import uuid

        from grampus.core.types import Message, Role

        system_content = SUBGOAL_SYSTEM_PROMPT.format(
            task=task,
            completed_summary=completed_summary,
            description=subgoal.description,
            success_criterion=subgoal.success_criterion,
            approach_hint=approach_hint,
        )
        _prefix = [Message(role=Role.SYSTEM, content=system_content)]
        session_id = f"subgoal-{subgoal.id}-{uuid.uuid4().hex[:8]}"
        result = await self._runner.run(
            agent_def,
            subgoal.description,
            session_id=session_id,
            _prefix_messages=_prefix,
        )
        return result.output or ""

    def _build_scoped_messages(
        self,
        task: str,
        subgoal: SubGoal,
        completed_summary: str,
        approach_hint: str,
    ) -> list[Any]:
        """Return [Message(SYSTEM, scoped_system_prompt), Message(USER, subgoal.description)].

        Args:
            task: Global task description.
            subgoal: The current subgoal.
            completed_summary: Formatted summary of completed subgoals.
            approach_hint: Approach hint from lookahead (may be empty).

        Returns:
            Two-element message list for scoped execution context.
        """
        from grampus.core.types import Message, Role

        system_content = SUBGOAL_SYSTEM_PROMPT.format(
            task=task,
            completed_summary=completed_summary,
            description=subgoal.description,
            success_criterion=subgoal.success_criterion,
            approach_hint=f"\nRecommended approach: {approach_hint}" if approach_hint else "",
        )
        return [
            Message(role=Role.SYSTEM, content=system_content),
            Message(role=Role.USER, content=subgoal.description),
        ]
