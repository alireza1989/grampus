"""LookaheadSimulator — FLARE-inspired candidate path selection before subgoal execution."""

from __future__ import annotations

import json
from typing import Any

from grampus.core.logging import get_logger
from grampus.orchestration.planning.types import SubGoal

_log = get_logger(__name__)

LOOKAHEAD_PROMPT = """
You are planning how to execute the following subgoal.

Global task: {task}
Completed so far: {completed_summary}
Current subgoal: {description}
Success criterion: {success_criterion}
Available tools: {tool_list}

Propose {n_paths} distinct execution approaches (different tool sequences or strategies).
For each, provide a brief plan and an estimated success probability (0.0-1.0).

Reply with JSON:
{{
  "paths": [
    {{"approach": "<description>", "tool_sequence": ["<tool>", ...], "estimated_success": 0.X}},
    ...
  ]
}}
"""


class LookaheadSimulator:
    """Generates and scores candidate execution paths before committing.

    Implements a lightweight version of FLARE's trajectory simulation:
    1. Generate n_paths candidate approaches via a single LLM call.
    2. Select the approach with the highest estimated_success score.
    3. Return the selected approach as a hint string for the subgoal executor.

    Cost: one extra LLM call per subgoal (using fast model).
    Only runs when PlanningConfig.enable_lookahead=True.

    Args:
        model_client: LLM client (duck-typed); must expose async complete().
        model_id: Model identifier to use for lookahead calls.
        n_paths: Number of candidate paths to generate per call.
        cost_tracker: Optional cost tracker; record() called after each call.
    """

    def __init__(
        self,
        model_client: Any,
        model_id: str,
        *,
        n_paths: int = 2,
        cost_tracker: Any | None = None,
    ) -> None:
        self._client = model_client
        self._model_id = model_id
        self._n_paths = n_paths
        self._cost_tracker = cost_tracker

    async def select_approach(
        self,
        task: str,
        subgoal: SubGoal,
        completed_summary: str,
        tool_names: list[str],
    ) -> str:
        """Return the best approach description as a hint string.

        Generates candidate paths and selects the one with the highest
        estimated_success score. Returns "" on any parse failure so that
        lookahead degrading gracefully never blocks execution.

        Args:
            task: Global task description.
            subgoal: The subgoal about to be executed.
            completed_summary: Summary of already-completed subgoals.
            tool_names: Available tool names.

        Returns:
            Best approach description string, or "" on failure.
        """
        from grampus.core.types import Message, Role

        prompt = LOOKAHEAD_PROMPT.format(
            task=task,
            completed_summary=completed_summary,
            description=subgoal.description,
            success_criterion=subgoal.success_criterion,
            tool_list=", ".join(tool_names) if tool_names else "none",
            n_paths=self._n_paths,
        )
        try:
            response = await self._client.complete(
                messages=[Message(role=Role.USER, content=prompt)],
                model=self._model_id,
            )
            if self._cost_tracker and response.token_usage:
                await self._cost_tracker.record(response.token_usage, step_name="lookahead")

            data = json.loads(_extract_json(response.content or ""))
            paths = data.get("paths", [])
            if not paths:
                return ""

            best = max(paths, key=lambda p: float(p.get("estimated_success", 0.0)))
            approach = str(best.get("approach", ""))
            _log.debug(
                "lookahead.selected",
                subgoal_id=subgoal.id,
                score=best.get("estimated_success"),
                approach=approach[:80],
            )
            return approach
        except Exception:  # noqa: BLE001
            _log.debug("lookahead.parse_failed", subgoal_id=subgoal.id)
            return ""


def _extract_json(text: str) -> str:
    """Extract the first JSON object from a text string."""
    import re

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text.strip()
