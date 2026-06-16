"""PostconditionVerifier — lightweight LLM check for subgoal success criteria."""

from __future__ import annotations

import json
from typing import Any

from grampus.core.logging import get_logger
from grampus.orchestration.planning.types import SubGoal, VerificationResult

_log = get_logger(__name__)

VERIFY_PROMPT = """
Task subgoal: {description}
Success criterion: {success_criterion}
Execution output: {output}

Did the execution meet the success criterion?
Reply with only JSON: {{"result": "pass"|"fail"|"partial", "reason": "<one sentence>"}}

- "pass": criterion clearly met
- "partial": some progress but criterion not fully met (worth retrying)
- "fail": criterion not met, retry unlikely to help
"""


class PostconditionVerifier:
    """Verifies whether a subgoal's success criterion was met after execution.

    Uses a fast model tier — this runs after every subgoal completion.

    Args:
        model_client: LLM client (duck-typed); must expose async complete().
        model_id: Model identifier to use for verification calls.
        cost_tracker: Optional cost tracker; record() called after each call.
    """

    def __init__(
        self,
        model_client: Any,
        model_id: str,
        *,
        cost_tracker: Any | None = None,
    ) -> None:
        self._client = model_client
        self._model_id = model_id
        self._cost_tracker = cost_tracker

    async def verify(
        self,
        subgoal: SubGoal,
        execution_output: str,
    ) -> tuple[VerificationResult, str]:
        """Check whether a subgoal's success criterion was satisfied.

        Args:
            subgoal: The subgoal that was executed.
            execution_output: Text output from the executor.

        Returns:
            Tuple of (VerificationResult, reason string).
            Falls back to (PASS, 'parse_fallback') on any error.
        """
        from grampus.core.types import Message, Role

        prompt = VERIFY_PROMPT.format(
            description=subgoal.description,
            success_criterion=subgoal.success_criterion,
            output=execution_output[:2000],
        )
        try:
            response = await self._client.complete(
                messages=[Message(role=Role.USER, content=prompt)],
                model=self._model_id,
            )
            if self._cost_tracker and response.token_usage:
                await self._cost_tracker.record(response.token_usage, step_name="verify")

            content = response.content or ""
            # Strip markdown code fences if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            data = json.loads(content.strip())
            raw_result = str(data.get("result", "pass")).lower()
            reason = str(data.get("reason", ""))

            result_map = {
                "pass": VerificationResult.PASS,
                "fail": VerificationResult.FAIL,
                "partial": VerificationResult.PARTIAL,
            }
            result = result_map.get(raw_result, VerificationResult.PASS)
            _log.debug(
                "verifier.result",
                subgoal_id=subgoal.id,
                result=str(result),
                reason=reason[:100],
            )
            return result, reason
        except Exception:  # noqa: BLE001
            _log.warning("verifier.parse_failed", subgoal_id=subgoal.id)
            return VerificationResult.PASS, "parse_fallback"
