"""Replanner — partial replan for unfinished subgoals after a failure."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from nexus.core.errors import PlanningError
from nexus.core.logging import get_logger
from nexus.orchestration.planning.types import Plan, PlanningConfig, SubGoal

_log = get_logger(__name__)

REPLAN_PROMPT = """
A subgoal in your plan has failed and cannot be retried.

Original task: {task}
Plan version: {version}

Completed subgoals (do NOT re-plan these):
{completed_subgoals}

Failed subgoal:
  ID: {failed_id}
  Description: {failed_description}
  Failure reason: {failure_reason}

Remaining planned subgoals (now invalidated):
{remaining_subgoals}

Generate a revised plan for ONLY the remaining work, taking the failure into account.
Do NOT include already-completed subgoals.
Output same JSON schema as original plan (subgoals array only, no wrapper):
{{
  "subgoals": [
    {{
      "id": "<short_slug>",
      "description": "<what to accomplish>",
      "success_criterion": "<verifiable condition>",
      "dependencies": ["<id>", ...],
      "tool_hints": ["<tool_name>", ...],
      "fallback_strategy": "<alternative>"
    }}
  ]
}}
"""

_FIX_PROMPT = (
    "Your previous response could not be parsed. Error: {error}. "
    "Output ONLY a valid JSON object with a 'subgoals' array."
)


class Replanner:
    """Generates a partial replan for unfinished subgoals after a failure.

    Only regenerates downstream work — completed subgoals are preserved
    unchanged per the Google DeepMind Subgoal Framework design.

    Args:
        model_client: LLM client (duck-typed); must expose async complete().
        model_id: Model identifier to use for replan calls.
        cost_tracker: Optional cost tracker; record() called after each call.
        tracer: Optional OTEL tracer for planning.replan spans.
    """

    def __init__(
        self,
        model_client: Any,
        model_id: str,
        *,
        cost_tracker: Any | None = None,
        tracer: Any | None = None,
    ) -> None:
        self._client = model_client
        self._model_id = model_id
        self._cost_tracker = cost_tracker
        self._tracer = tracer

    async def replan(
        self,
        original_plan: Plan,
        failed_subgoal: SubGoal,
        completed_subgoals: list[SubGoal],
        config: PlanningConfig,
    ) -> Plan:
        """Return a new Plan with version incremented.

        The new plan contains completed subgoals (status=COMPLETED, unchanged)
        plus newly generated subgoals for the remaining work.

        Args:
            original_plan: The plan that failed.
            failed_subgoal: The subgoal that could not be completed.
            completed_subgoals: Subgoals already successfully completed.
            config: Planning configuration with max_replans limit.

        Returns:
            New Plan with version = original_plan.version + 1.

        Raises:
            PlanningError: code='MAX_REPLANS_EXCEEDED' when plan.version >= max_replans.
            PlanningError: code='REPLAN_PARSE_FAILED' when LLM output is unparseable.
        """
        if original_plan.version >= config.max_replans:
            raise PlanningError(
                f"Maximum replans ({config.max_replans}) exceeded",
                code="MAX_REPLANS_EXCEEDED",
            )

        completed_ids = {sg.id for sg in completed_subgoals}
        remaining = [
            sg
            for sg in original_plan.subgoals
            if sg.id not in completed_ids and sg.id != failed_subgoal.id
        ]

        prompt = REPLAN_PROMPT.format(
            task=original_plan.task,
            version=original_plan.version,
            completed_subgoals=_format_subgoals(completed_subgoals),
            failed_id=failed_subgoal.id,
            failed_description=failed_subgoal.description,
            failure_reason=failed_subgoal.failure_reason or "unknown",
            remaining_subgoals=_format_subgoals(remaining),
        )
        from nexus.core.types import Message, Role

        messages = [Message(role=Role.USER, content=prompt)]
        last_error = ""

        for attempt in range(2):
            if attempt == 1:
                messages.append(
                    Message(role=Role.USER, content=_FIX_PROMPT.format(error=last_error))
                )
            response = await self._client.complete(messages=messages, model=self._model_id)
            if self._cost_tracker and response.token_usage:
                await self._cost_tracker.record(response.token_usage, step_name="replan")
            messages.append(Message(role=Role.ASSISTANT, content=response.content))

            try:
                new_subgoals = _parse_subgoals(response.content or "")
                all_subgoals = list(completed_subgoals) + new_subgoals
                new_plan = Plan(
                    task=original_plan.task,
                    subgoals=all_subgoals,
                    total_estimated_steps=len(all_subgoals),
                    created_at=datetime.now(UTC),
                    version=original_plan.version + 1,
                )
                _log.info(
                    "replanner.replan_created",
                    version=new_plan.version,
                    new_subgoals=len(new_subgoals),
                )
                return new_plan
            except Exception as exc:
                last_error = str(exc)
                _log.warning("replanner.parse_failed", attempt=attempt, error=last_error)

        raise PlanningError(
            f"Failed to parse replan response after 2 attempts: {last_error}",
            code="REPLAN_PARSE_FAILED",
        )


def _format_subgoals(subgoals: list[SubGoal]) -> str:
    """Format subgoals as a readable list for the replan prompt."""
    if not subgoals:
        return "  (none)"
    lines = []
    for sg in subgoals:
        lines.append(f"  - {sg.id}: {sg.description}")
        if sg.output_summary:
            lines.append(f"    output: {sg.output_summary}")
    return "\n".join(lines)


def _parse_subgoals(content: str) -> list[SubGoal]:
    """Extract and parse a subgoals list from LLM response text."""
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response")
    data = json.loads(match.group(0))
    subgoals_data = data.get("subgoals", [])
    if not subgoals_data:
        raise ValueError("No subgoals in replan response")
    return [SubGoal(**sg) for sg in subgoals_data]
