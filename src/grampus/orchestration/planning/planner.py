"""Planner — generates a structured SubGoal DAG from a task description."""

from __future__ import annotations

import json
import re
from collections import deque
from datetime import UTC, datetime
from typing import Any

from grampus.core.errors import PlanningError
from grampus.core.logging import get_logger
from grampus.orchestration.planning.types import Plan, PlanningConfig, SubGoal

_log = get_logger(__name__)

PLAN_PROMPT = """
You are a strategic task planner. Break the following task into a structured plan.

Task: {task}

Available tools: {tool_list}

Context from memory: {memory_context}

Output a JSON plan:
{{
  "total_estimated_steps": <int>,
  "subgoals": [
    {{
      "id": "<short_slug>",
      "description": "<what to accomplish>",
      "success_criterion": "<verifiable completion condition>",
      "dependencies": ["<id>", ...],
      "tool_hints": ["<tool_name>", ...],
      "fallback_strategy": "<alternative if primary fails>"
    }}
  ]
}}

Rules:
- Maximum {max_subgoals} subgoals.
- Each id must be unique, snake_case, <= 20 chars.
- Dependencies must reference ids defined earlier in the list.
- success_criterion must be concise and checkable (not "do a good job").
- fallback_strategy should suggest a genuinely different approach, not a retry.
- Independent subgoals (no shared dependencies) can run in parallel.
"""

COMPLEXITY_PROMPT = """
Estimate how many tool calls this task will require. Reply with only a JSON:
{{"estimated_steps": <int>, "reason": "<one sentence>"}}

Task: {task}
"""

_FIX_PROMPT = (
    "Your previous response could not be parsed as valid JSON. "
    "Error: {error}. "
    "Please output ONLY valid JSON matching the schema, with no extra text."
)


class Planner:
    """Generates a structured SubGoal DAG from a task description.

    Args:
        model_client: LLM client (duck-typed); must expose async complete().
        model_id: Model identifier to pass to the client.
        cost_tracker: Optional cost tracker; record() called after each LLM call.
        tracer: Optional OTEL tracer for planning.create_plan spans.
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

    async def estimate_complexity(self, task: str) -> int:
        """Return estimated tool call count for a task. One cheap LLM call.

        Args:
            task: User task description.

        Returns:
            Estimated number of tool calls; returns 999 on any parse failure.
        """
        prompt = COMPLEXITY_PROMPT.format(task=task)
        from grampus.core.types import Message, Role

        try:
            response = await self._client.complete(
                messages=[Message(role=Role.USER, content=prompt)],
                model=self._model_id,
            )
            if self._cost_tracker and response.token_usage:
                await self._cost_tracker.record(
                    response.token_usage, step_name="complexity_estimate"
                )
            raw = _extract_json(response.content or "")
            data = json.loads(raw)
            return int(data["estimated_steps"])
        except Exception:  # noqa: BLE001
            return 999

    async def create_plan(
        self,
        task: str,
        *,
        tool_names: list[str],
        memory_context: str = "",
        config: PlanningConfig,
    ) -> Plan:
        """Generate a Plan from the planning prompt.

        Retries once with an error-correction prompt on parse failure.
        Falls back to a single-subgoal degenerate plan if both attempts fail.

        Args:
            task: User task description.
            tool_names: Names of available tools.
            memory_context: Optional context string from memory recall.
            config: Planning configuration.

        Returns:
            A validated Plan with topologically consistent subgoals.
        """
        prompt = PLAN_PROMPT.format(
            task=task,
            tool_list=", ".join(tool_names) if tool_names else "none",
            memory_context=memory_context or "none",
            max_subgoals=config.max_subgoals,
        )
        from grampus.core.types import Message, Role

        messages = [Message(role=Role.USER, content=prompt)]
        last_error = ""

        for attempt in range(2):
            if attempt == 1:
                messages.append(
                    Message(
                        role=Role.USER,
                        content=_FIX_PROMPT.format(error=last_error),
                    )
                )
            response = await self._client.complete(messages=messages, model=self._model_id)
            if self._cost_tracker and response.token_usage:
                await self._cost_tracker.record(response.token_usage, step_name="create_plan")
            messages.append(Message(role=Role.ASSISTANT, content=response.content))

            try:
                plan = self._parse_plan(task, response.content or "")
                _log.debug(
                    "planner.plan_created", subgoals=len(plan.subgoals), version=plan.version
                )
                return plan
            except Exception as exc:
                last_error = str(exc)
                _log.warning("planner.parse_failed", attempt=attempt, error=last_error)

        return self._degenerate_plan(task)

    def _parse_plan(self, task: str, content: str) -> Plan:
        """Parse and validate a plan from LLM output.

        Args:
            task: Original task string.
            content: Raw LLM response text.

        Returns:
            A validated Plan object.

        Raises:
            ValueError: On JSON parse failure or validation error.
        """
        raw = _extract_json(content)
        data = json.loads(raw)
        subgoals_data = data.get("subgoals", [])
        if not subgoals_data:
            raise ValueError("No subgoals in response")

        subgoals = [SubGoal(**sg) for sg in subgoals_data]
        self._validate_subgoals(subgoals)

        return Plan(
            task=task,
            subgoals=subgoals,
            total_estimated_steps=int(data.get("total_estimated_steps", 0)),
            created_at=datetime.now(UTC),
        )

    def _validate_subgoals(self, subgoals: list[SubGoal]) -> None:
        """Validate IDs are unique and dependencies form a valid DAG.

        Args:
            subgoals: List of SubGoal objects to validate.

        Raises:
            ValueError: On duplicate IDs or missing dependency references.
            PlanningError: code='CIRCULAR_DEPENDENCY' on cycle detection.
        """
        ids = [sg.id for sg in subgoals]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate subgoal IDs detected")

        id_set = set(ids)
        for sg in subgoals:
            for dep in sg.dependencies:
                if dep not in id_set:
                    raise ValueError(f"Dependency '{dep}' not found in subgoal list")

        self._topological_sort(subgoals)

    def _topological_sort(self, subgoals: list[SubGoal]) -> list[list[SubGoal]]:
        """Group subgoals into parallel execution waves via Kahn's algorithm.

        Args:
            subgoals: List of SubGoal objects with dependency info.

        Returns:
            List of waves; each wave is a list of subgoals with satisfied deps.

        Raises:
            PlanningError: code='CIRCULAR_DEPENDENCY' when a cycle is detected.
        """
        index = {sg.id: sg for sg in subgoals}
        in_degree: dict[str, int] = {sg.id: 0 for sg in subgoals}
        children: dict[str, list[str]] = {sg.id: [] for sg in subgoals}

        for sg in subgoals:
            for dep in sg.dependencies:
                in_degree[sg.id] += 1
                children[dep].append(sg.id)

        waves: list[list[SubGoal]] = []
        queue: deque[str] = deque(sg_id for sg_id, deg in in_degree.items() if deg == 0)

        while queue:
            wave = list(queue)
            queue.clear()
            waves.append([index[sg_id] for sg_id in wave])
            for sg_id in wave:
                for child in children[sg_id]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        if sum(len(w) for w in waves) != len(subgoals):
            raise PlanningError(
                "Circular dependency detected in plan subgoals",
                code="CIRCULAR_DEPENDENCY",
            )
        return waves

    def _degenerate_plan(self, task: str) -> Plan:
        """Return a minimal single-subgoal plan when parsing fails.

        Args:
            task: Original user task.

        Returns:
            Plan with one subgoal covering the entire task.
        """
        _log.warning("planner.degenerate_fallback", task_len=len(task))
        return Plan(
            task=task,
            subgoals=[
                SubGoal(
                    id="execute_task",
                    description=task,
                    success_criterion="Task completed successfully",
                )
            ],
            total_estimated_steps=1,
            created_at=datetime.now(UTC),
        )


def _extract_json(text: str) -> str:
    """Extract the first JSON object or array from a text string."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return match.group(0)
    return text.strip()
