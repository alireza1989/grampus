"""Sycophancy-resistant debater that generates a position for each debate round.

Prompt design grounded in:
  CONSENSAGENT (ACL 2025) — agents must restate their prior answer before evaluating peers.
  "Peacemaker or Troublemaker" (arXiv 2509.23055) — position changes require explicit evidence.
"""

from __future__ import annotations

import json
import re
from typing import Any

from nexus.core.logging import get_logger
from nexus.core.types import Message, Role, TokenUsage
from nexus.orchestration.debate.types import DebaterConfig, DebateRound, DebaterPosition

_log = get_logger(__name__)

_ROUND1_SYSTEM = (
    "You are a careful analytical reasoner.{role_hint}\n"
    "Generate your best answer to the question below. Think step by step.\n"
    'Output JSON: {{"answer": "<concise answer>", "reasoning": "<step-by-step reasoning>", "confidence": <0.0-1.0>}}'
)

_ROUND2_SYSTEM = (
    "You are a careful analytical reasoner.{role_hint}\n"
    "You previously answered this question. Other agents have also answered.\n"
    "Your task:\n"
    "1. FIRST: Restate your previous answer and reasoning verbatim.\n"
    "2. THEN: Evaluate each peer position critically. What evidence or logic supports "
    "or contradicts it?\n"
    "3. FINALLY: Output your updated position. ONLY change your answer if you find "
    "compelling new logical evidence.\n"
    "   Changing your position due to social pressure or because others disagree is NOT acceptable.\n"
    '   If you change your answer, explicitly state: "I changed my answer because: [specific logical reason]"\n\n'
    'Output JSON: {{"answer": "...", "reasoning": "...", "confidence": 0.0-1.0, '
    '"changed": true/false, "change_justification": "..."}}'
)


class Debater:
    """A single participant in a multi-agent debate.

    Args:
        index: Zero-based index identifying this debater.
        config: Model configuration for this participant.
    """

    def __init__(self, index: int, config: DebaterConfig) -> None:
        self._index = index
        self._config = config

    @property
    def index(self) -> int:
        """Zero-based position of this debater in the panel."""
        return self._index

    @property
    def model_id(self) -> str:
        """Model identifier string for this debater."""
        return self._config.model_id

    async def respond(
        self,
        question: str,
        round_number: int,
        previous_round: DebateRound | None,
        cost_tracker: Any | None,
        tracer: Any | None,
    ) -> DebaterPosition:
        """Generate this debater's position for the given round.

        Round 1: independent reasoning (no peer positions shown).
        Round 2+: sycophancy-resistant prompt showing peers' prior answers.
        """
        messages = (
            self._build_round1_messages(question)
            if round_number == 1 or previous_round is None
            else self._build_round2_messages(question, previous_round)
        )

        response = await self._call_model(messages)
        parsed = _parse_json(response.get("content") or "")

        changed = bool(parsed.get("changed", False)) if round_number > 1 else False
        position = DebaterPosition(
            debater_index=self._index,
            model_id=self._config.model_id,
            answer=str(parsed.get("answer") or response.get("content") or ""),
            reasoning=str(parsed.get("reasoning", "")),
            confidence=float(parsed.get("confidence", 0.5)),
            changed_from_previous=changed,
            change_justification=str(parsed.get("change_justification", "")),
            token_usage=response.get("token_usage"),
        )
        _log.debug(
            "debater.respond",
            index=self._index,
            round=round_number,
            confidence=position.confidence,
            changed=position.changed_from_previous,
        )
        return position

    def _build_round1_messages(self, question: str) -> list[Message]:
        hint = f"\n{self._config.role_hint}" if self._config.role_hint else ""
        system = _ROUND1_SYSTEM.format(role_hint=hint)
        return [
            Message(role=Role.SYSTEM, content=system),
            Message(role=Role.USER, content=question),
        ]

    def _build_round2_messages(self, question: str, previous_round: DebateRound) -> list[Message]:
        hint = f"\n{self._config.role_hint}" if self._config.role_hint else ""
        system = _ROUND2_SYSTEM.format(role_hint=hint)
        peer_lines = [
            f"Agent {p.debater_index}: {p.answer}\nReasoning: {p.reasoning}"
            for p in previous_round.positions
            if p.debater_index != self._index
        ]
        peer_text = "\n\n".join(peer_lines) if peer_lines else "(no peer positions)"
        return [
            Message(role=Role.SYSTEM, content=system),
            Message(role=Role.USER, content=question),
            Message(role=Role.USER, content=f"Peer positions:\n{peer_text}"),
        ]

    async def _call_model(self, messages: list[Message]) -> dict[str, Any]:
        """Call the model client, handling temperature injection robustly."""
        kwargs: dict[str, Any] = {"temperature": self._config.temperature}
        try:
            response = await self._config.model_client.complete(
                messages=messages,
                model=self._config.model_id,
                **kwargs,
            )
        except TypeError:
            response = await self._config.model_client.complete(
                messages=messages,
                model=self._config.model_id,
            )
        usage: TokenUsage | None = getattr(response, "token_usage", None)
        return {"content": getattr(response, "content", None), "token_usage": usage}


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
