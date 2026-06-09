"""CausalRelationExtractor: LLM labels causal relations from agent reasoning text.

The LLM's job is only to identify and name causal relationships.
All causal inference is performed by SimpleCausalInference in inference.py.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from nexus.causal.types import CausalRelation
from nexus.core.logging import get_logger

_log = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a causal analyst. Given a passage of agent reasoning or tool output, "
    "extract explicit causal claims. "
    "Reply with only valid JSON: "
    '{"relations": [{"cause": "...", "effect": "...", '
    '"type": "caused|enabled|prevented|triggered", "confidence": 0.0}]}. '
    "Only extract clearly stated relationships. Maximum 3 per passage. "
    'If none, reply {"relations": []}.'
)

_USER_TEMPLATE = "Passage:\n{text}\n\nExtract causal relations:"


class CausalRelationExtractor:
    """Extracts causal claims from agent reasoning text using a single LLM call.

    The LLM labels; code stores and reasons. Never ask the LLM to infer
    causal effects — only to identify stated or strongly implied causation.

    Args:
        model_client: ModelClient for LLM calls.
    """

    def __init__(self, model_client: Any) -> None:
        self._model = model_client

    async def extract(
        self,
        text: str,
        *,
        session_id: str,
        agent_id: str,
    ) -> list[CausalRelation]:
        """Extract causal relations from text.

        Returns [] on LLM error, JSON parse failure, or empty text. Never raises.
        """
        if not text or not text.strip():
            return []
        try:
            return await self._call_and_parse(text, session_id=session_id, agent_id=agent_id)
        except Exception:
            _log.warning("causal_extract_failed", session_id=session_id)
            return []

    async def _call_and_parse(
        self, text: str, *, session_id: str, agent_id: str
    ) -> list[CausalRelation]:
        from nexus.core.types import Message, Role  # noqa: PLC0415

        messages = [
            Message(role=Role.SYSTEM, content=_SYSTEM_PROMPT),
            Message(role=Role.USER, content=_USER_TEMPLATE.format(text=text[:1500])),
        ]
        response = await self._model.complete(messages, temperature=0.1, max_tokens=300)
        raw = (response.content or "").strip()
        data = json.loads(raw)
        relations: list[CausalRelation] = []
        for item in data.get("relations", []):
            confidence = float(item.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            relations.append(
                CausalRelation(
                    relation_id=str(uuid.uuid4()),
                    cause_description=str(item.get("cause", "")),
                    effect_description=str(item.get("effect", "")),
                    relation_type=str(item.get("type", "caused")),
                    confidence=confidence,
                    evidence_text=text[:200],
                    session_id=session_id,
                    agent_id=agent_id,
                )
            )
        return relations
