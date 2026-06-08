"""Adaptive debate router — bypasses full debate when a single agent is already confident.

Grounded in: "Debate Only When Necessary" (arXiv 2504.05047).
Skipping debate when first-agent confidence >= threshold eliminates ~40% of unnecessary
calls with no quality loss.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from collections.abc import Generator
from typing import Any

from nexus.core.logging import get_logger
from nexus.core.types import Message, Role, TokenUsage
from nexus.orchestration.debate.types import (
    AggregationStrategy,
    DebateConfig,
    DebateResult,
    DebateRound,
    DebaterPosition,
    RoutingDecision,
)

_log = get_logger(__name__)

_ZERO_USAGE = TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0, cost_usd=0.0, model="")


class DebateRouter:
    """Checks whether a single fast model can answer confidently enough to skip debate.

    Args:
        config: The debate configuration supplying routing thresholds and model.
    """

    def __init__(self, config: DebateConfig) -> None:
        self._config = config

    async def route(
        self,
        question: str,
        cost_tracker: Any | None,
        tracer: Any | None,
    ) -> DebateResult | None:
        """Return a single-agent DebateResult if bypassing debate, else None.

        A None return means the caller should proceed with the full debate.
        """
        start = time.monotonic()
        client = self._config.routing_model_client or self._config.debaters[0].model_client
        model_id = self._config.routing_model_id or self._config.debaters[0].model_id

        with _maybe_span(tracer, "debate.route_check", routing_model_id=model_id):
            parsed, usage = await self._single_call(client, model_id, question)

        confidence = float(parsed.get("confidence", 0.0))
        _log.debug("debate.route_check", model=model_id, confidence=confidence)

        if confidence < self._config.routing_confidence_threshold:
            return None

        position = DebaterPosition(
            debater_index=0,
            model_id=model_id,
            answer=str(parsed.get("answer", "")),
            reasoning=str(parsed.get("reasoning", "")),
            confidence=confidence,
            token_usage=usage,
        )
        single_round = DebateRound(
            round_number=1,
            positions=[position],
            convergence_score=1.0,
            stopped_early=False,
        )
        return DebateResult(
            question=question,
            final_answer=position.answer,
            final_reasoning=position.reasoning,
            confidence=confidence,
            escalate_to_human=False,
            rounds=[single_round],
            aggregation_method=AggregationStrategy.MAJORITY_VOTE,
            routing_decision=RoutingDecision.SINGLE_AGENT,
            total_rounds_run=1,
            converged=True,
            final_convergence_score=1.0,
            total_token_usage=usage or _ZERO_USAGE,
            total_cost_usd=(usage.cost_usd if usage else 0.0),
            duration_seconds=time.monotonic() - start,
        )

    async def _single_call(
        self,
        client: Any,
        model_id: str,
        question: str,
    ) -> tuple[dict[str, Any], TokenUsage | None]:
        """Call the routing model once and parse the JSON response."""
        system_msg = Message(
            role=Role.SYSTEM,
            content=(
                "You are a careful analytical reasoner.\n"
                "Generate your best answer to the question below. Think step by step.\n"
                'Output JSON: {"answer": "<concise answer>", "reasoning": "<reasoning>", "confidence": <0.0-1.0>}'
            ),
        )
        user_msg = Message(role=Role.USER, content=question)
        try:
            response = await client.complete(messages=[system_msg, user_msg], model=model_id)
        except Exception as exc:
            _log.warning("debate.route_model_error", error=str(exc))
            return {}, None

        usage: TokenUsage | None = getattr(response, "token_usage", None)
        parsed = _parse_json(getattr(response, "content", "") or "")
        return parsed, usage


@contextlib.contextmanager
def _maybe_span(tracer: Any | None, name: str, **attrs: Any) -> Generator[None, None, None]:
    """Emit a span if a tracer is provided; no-op otherwise."""
    if tracer is None:
        yield
        return
    with tracer.span(name, **attrs):
        yield


def _parse_json(content: str) -> dict[str, Any]:
    """Extract and parse the first JSON object from LLM output."""
    text = content.strip()
    try:
        return dict(json.loads(text))
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return dict(json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass
    match2 = re.search(r"\{.*\}", text, re.DOTALL)
    if match2:
        try:
            return dict(json.loads(match2.group(0)))
        except json.JSONDecodeError:
            pass
    return {}
