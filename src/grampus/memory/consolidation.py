"""Consolidation pipeline: extract semantic facts from episodic records via LLM."""

from __future__ import annotations

import json
import uuid
from typing import Any

from pydantic import BaseModel

from grampus.core.logging import get_logger
from grampus.core.types import Message, Role
from grampus.memory.types import SemanticFact

_log = get_logger(__name__)

_EXTRACTION_PROMPT = """\
Extract factual knowledge from the conversation records below as a JSON array.
Each fact must have: subject, predicate, object_value, confidence (0.0–1.0).
Focus on durable facts: preferences, skills, relationships, attributes.
Return ONLY a valid JSON array. If no facts can be extracted, return [].

Records:
{records}

Response format: [{{"subject": "...", "predicate": "...", "object_value": "...", "confidence": 0.9}}]"""


class ConsolidationResult(BaseModel):
    """Summary of a single consolidation run."""

    facts_extracted: int
    facts_merged: int
    episodes_processed: int


class ConsolidationPipeline:
    """Scans recent episodic records, extracts facts via LLM, updates semantic memory.

    Episodes are processed in batches of ``batch_size`` (newest-first).
    Each processed episode is marked with ``metadata["consolidated"] = True``
    so subsequent runs skip it.

    Args:
        episodic_memory: Source of episodic records.
        semantic_memory: Destination for extracted facts.
        model_client: LLM client for fact extraction.
        agent_id: Scopes processing to this agent.
        batch_size: Maximum number of episodes to process per run.
    """

    def __init__(
        self,
        episodic_memory: Any,
        semantic_memory: Any,
        model_client: Any,
        *,
        agent_id: str,
        batch_size: int = 10,
    ) -> None:
        self._episodic = episodic_memory
        self._semantic = semantic_memory
        self._client = model_client
        self._agent_id = agent_id
        self._batch_size = batch_size

    async def run(self) -> ConsolidationResult:
        """Run one consolidation pass and return a summary."""
        all_episodes = await self._episodic.list_all()
        pending = [ep for ep in all_episodes if not ep.metadata.get("consolidated")]
        batch = pending[: self._batch_size]

        if not batch:
            _log.debug("consolidation_no_episodes", agent=self._agent_id)
            return ConsolidationResult(facts_extracted=0, facts_merged=0, episodes_processed=0)

        facts_extracted = 0
        for episode in batch:
            raw_facts = await self._extract_facts(episode.content)
            for raw in raw_facts:
                fact = SemanticFact(
                    id=str(uuid.uuid4()),
                    subject=raw.get("subject", ""),
                    predicate=raw.get("predicate", ""),
                    object_value=raw.get("object_value", ""),
                    confidence=float(raw.get("confidence", 1.0)),
                    source_episode_ids=[episode.id],
                )
                await self._semantic.store(fact)
                facts_extracted += 1

            await self._episodic.update_metadata(episode.id, {"consolidated": True})

        _log.debug(
            "consolidation_complete",
            agent=self._agent_id,
            episodes=len(batch),
            facts=facts_extracted,
        )
        return ConsolidationResult(
            facts_extracted=facts_extracted,
            facts_merged=0,
            episodes_processed=len(batch),
        )

    async def _extract_facts(self, content: str) -> list[dict[str, Any]]:
        prompt = _EXTRACTION_PROMPT.format(records=content)
        messages = [Message(role=Role.USER, content=prompt)]
        try:
            response = await self._client.complete(
                messages=messages,
                model="claude-haiku-4-5-20251001",
                temperature=0.0,
            )
            raw = (response.content or "").strip()
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                return []
            return [item for item in parsed if isinstance(item, dict)]
        except (json.JSONDecodeError, Exception) as exc:
            _log.warning("consolidation_extraction_failed", error=str(exc))
            return []
