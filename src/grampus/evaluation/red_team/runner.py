"""RedTeamRunner: orchestrates the full Attacker → Target → Judge loop."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from grampus.core.logging import get_logger
from grampus.evaluation.red_team.attacker import AttackerAgent
from grampus.evaluation.red_team.judge import RedTeamJudge
from grampus.evaluation.red_team.types import (
    AttackPayload,
    AttackResult,
    RedTeamCampaignConfig,
    RedTeamTargetConfig,
    Severity,
)

_log = get_logger(__name__)


class RedTeamRunner:
    """
    Orchestrates one complete red-team campaign.

    Loop per payload:
    1. AttackerAgent.generate_payloads() → list[AttackPayload]
    2. For each payload: submit to target agent → collect response
    3. RedTeamJudge.evaluate(payload, response) → JudgeVerdict
    4. If failed + model available: AttackerAgent.mutate_failed() → retry once
    5. Collect all AttackResults

    Concurrency: asyncio.Semaphore bounded by config.max_concurrent.

    The target agent is called via `target_fn`: an async callable that takes
    a list of (role, content) message tuples and returns a string response.
    This keeps RedTeamRunner decoupled from AgentRunner's full lifecycle.

    Args:
        attacker: AttackerAgent instance.
        judge: RedTeamJudge instance.
        target_fn: async callable (messages: list[tuple[str, str]]) -> str
    """

    def __init__(
        self,
        attacker: AttackerAgent,
        judge: RedTeamJudge,
        target_fn: Any,
    ) -> None:
        self._attacker = attacker
        self._judge = judge
        self._target_fn = target_fn

    async def run(self, config: RedTeamCampaignConfig) -> list[AttackResult]:
        """
        Run the full campaign. Returns all AttackResults (successful and failed).

        Respects config.stop_on_critical: aborts after first CRITICAL finding.
        Never raises.
        """
        try:
            return await self._run_campaign(config)
        except Exception:
            _log.warning("redteam_runner_failed", campaign=config.campaign_id)
            return []

    async def _run_campaign(self, config: RedTeamCampaignConfig) -> list[AttackResult]:
        payloads = await self._attacker.generate_payloads(
            config.target,
            categories=config.enabled_categories,
            count_per_strategy=config.payloads_per_strategy,
        )
        _log.info(
            "redteam_campaign_started",
            campaign=config.campaign_id,
            payload_count=len(payloads),
        )
        sem = asyncio.Semaphore(config.max_concurrent)
        results: list[AttackResult] = []
        for payload in payloads:
            result = await self._run_one(payload, sem, config.target)
            results.append(result)
            if config.stop_on_critical and result.verdict.severity == Severity.CRITICAL:
                _log.warning(
                    "redteam_critical_finding_stop",
                    campaign=config.campaign_id,
                )
                break
        _log.info(
            "redteam_campaign_complete",
            campaign=config.campaign_id,
            total=len(results),
            succeeded=sum(1 for r in results if r.verdict.succeeded),
        )
        return results

    async def _run_one(
        self,
        payload: AttackPayload,
        sem: asyncio.Semaphore,
        target: RedTeamTargetConfig,
    ) -> AttackResult:
        """Execute one payload: call target, judge, optional mutate+retry."""
        async with sem:
            t0 = time.monotonic()
            try:
                response = await self._call_target(payload)
            except Exception:
                response = ""
            verdict = await self._judge.evaluate(payload, response)
            duration_ms = (time.monotonic() - t0) * 1000.0

            result = AttackResult(
                result_id=str(uuid.uuid4()),
                payload=payload,
                target_response=response,
                verdict=verdict,
                duration_ms=round(duration_ms, 1),
            )

            if not verdict.succeeded:
                mutated = await self._attacker.mutate_failed(result, target)
                if mutated and mutated.content != payload.content:
                    result = await self._retry_mutated(mutated)

            return result

    async def _retry_mutated(self, mutated: AttackPayload) -> AttackResult:
        """Execute a mutated payload and return its result."""
        t1 = time.monotonic()
        try:
            response2 = await self._call_target(mutated)
        except Exception:
            response2 = ""
        verdict2 = await self._judge.evaluate(mutated, response2)
        return AttackResult(
            result_id=str(uuid.uuid4()),
            payload=mutated,
            target_response=response2,
            verdict=verdict2,
            duration_ms=round((time.monotonic() - t1) * 1000.0, 1),
        )

    async def _call_target(self, payload: AttackPayload) -> str:
        """Build the conversation and call target_fn."""
        messages: list[tuple[str, str]] = []
        for i, turn_content in enumerate(payload.prior_turns):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append((role, turn_content))
        messages.append(("user", payload.content))
        return str(await self._target_fn(messages))
