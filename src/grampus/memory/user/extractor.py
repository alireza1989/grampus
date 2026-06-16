"""FactExtractor — inductive agent: session → UserFacts (Bi-Mem arXiv 2601.06490)."""

from __future__ import annotations

import json
import math
import uuid
from typing import TYPE_CHECKING, Any

from grampus.core.logging import get_logger
from grampus.memory.user.types import (
    FactExtractionResult,
    UserFact,
    UserFactCategory,
)

if TYPE_CHECKING:
    from grampus.memory.episodic import EpisodicMemory
    from grampus.memory.user.store import UserMemoryStore

_log = get_logger(__name__)

_EXTRACT_SYSTEM_PROMPT = (
    "You are a user modeling system. "
    "Analyze the following conversation and extract factual statements about the USER. "
    "Focus on: expertise level and domains, communication preferences, decisions made, "
    "current goals or projects, and any stated constraints. "
    "Each fact must be a single clear statement. Extract only facts explicitly stated "
    "or strongly implied — do not infer speculatively. "
    'Reply with only valid JSON: {"facts": [{"content": "...", "category": "...", '
    '"confidence": 0.0-1.0}]}. '
    "category must be one of: expertise, preference, decision, context, constraint. "
    'If no facts found, reply with {"facts": []}.'
)

_EXTRACT_USER_TEMPLATE = (
    "Conversation (session_id={session_id}):\n\n{conversation}\n\nExtract facts about the user:"
)

_CONTRADICTION_SYSTEM_PROMPT = (
    "You are checking for logical contradictions between statements. "
    "For each pair, reply with a JSON array of booleans: [true/false, ...] "
    "where true means the new statement CONTRADICTS the existing statement. "
    "A contradiction means both cannot be true simultaneously. "
    "Temporal changes are NOT contradictions."
)

_SIMILARITY_DEDUP_THRESHOLD = 0.90
_CONTRADICTION_MIN_SIMILARITY = 0.50


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


class FactExtractor:
    """Inductive agent: extracts UserFacts from a completed session's EpisodicMemory records.

    Deduplication logic:
    - If a new fact has cosine similarity > threshold to an existing valid fact
      of the same category: update existing fact's confidence (EMA, alpha=0.3)
      rather than creating a duplicate.
    - If a new fact explicitly contradicts an existing fact (LLM contradiction check):
      expire the old fact (set valid_until = now) and store the new one.

    Args:
        store: UserMemoryStore for persisting extracted facts.
        episodic_memory: Source of raw session records.
        embedding_service: For embedding new facts.
        max_conversation_tokens: Truncate session content to this length
            before sending to LLM. Default 2000 chars.
    """

    def __init__(
        self,
        store: UserMemoryStore,
        episodic_memory: EpisodicMemory,
        embedding_service: Any,
        *,
        max_conversation_tokens: int = 2000,
    ) -> None:
        self._store = store
        self._episodic = episodic_memory
        self._embeddings = embedding_service
        self._max_chars = max_conversation_tokens * 4

    async def extract_from_session(
        self,
        user_id: str,
        session_id: str,
        model_client: Any,
    ) -> FactExtractionResult:
        """Extract and persist UserFacts from a completed session.

        Steps:
        1. Load all EpisodicRecords for this session with matching user_id
        2. Concatenate content into conversation string (truncate to max chars)
        3. LLM call to extract facts
        4. For each extracted fact: dedup, contradiction check, or store new
        5. Return FactExtractionResult

        Never raises — returns FactExtractionResult(facts_extracted=0, ...) on error.
        """
        try:
            return await self._do_extract(user_id, session_id, model_client)
        except Exception:
            _log.warning("fact_extraction_failed", user_id=user_id, session_id=session_id)
            return FactExtractionResult(
                user_id=user_id,
                session_id=session_id,
                facts_extracted=0,
                facts_updated=0,
                facts_expired=0,
            )

    async def _do_extract(
        self,
        user_id: str,
        session_id: str,
        model_client: Any,
    ) -> FactExtractionResult:
        all_records = await self._episodic.list_all()
        session_records = [
            r for r in all_records if r.user_id == user_id and r.session_id == session_id
        ]

        conversation = self._build_conversation_string(session_records, self._max_chars)
        if not conversation:
            return FactExtractionResult(
                user_id=user_id,
                session_id=session_id,
                facts_extracted=0,
                facts_updated=0,
                facts_expired=0,
            )

        from grampus.core.types import Message, Role  # noqa: PLC0415

        response = await model_client.complete(
            messages=[
                Message(role=Role.SYSTEM, content=_EXTRACT_SYSTEM_PROMPT),
                Message(
                    role=Role.USER,
                    content=_EXTRACT_USER_TEMPLATE.format(
                        session_id=session_id, conversation=conversation
                    ),
                ),
            ],
            model="claude-haiku-4-5-20251001",
            temperature=0.2,
        )

        raw_facts = self._parse_extraction_response(response.content or "")
        if not raw_facts:
            return FactExtractionResult(
                user_id=user_id,
                session_id=session_id,
                facts_extracted=0,
                facts_updated=0,
                facts_expired=0,
            )

        facts_extracted = 0
        facts_updated = 0
        facts_expired = 0
        new_fact_ids: list[str] = []

        existing_facts = await self._store.get_valid_facts(user_id)

        for raw in raw_facts:
            content = raw.get("content", "").strip()
            category_str = raw.get("category", "context")
            confidence = float(raw.get("confidence", 1.0))
            if not content:
                continue

            try:
                category = UserFactCategory(category_str)
            except ValueError:
                category = UserFactCategory.CONTEXT

            embedding: list[float] | None = None
            try:
                embedding = await self._embeddings.embed(content)
            except Exception:
                _log.warning("fact_embedding_failed", user_id=user_id)

            same_cat = [f for f in existing_facts if f.category == category]
            top_similar: list[tuple[UserFact, float]] = []

            if embedding is not None:
                for ef in same_cat:
                    if ef.embedding is not None:
                        sim = _cosine(embedding, ef.embedding)
                        top_similar.append((ef, sim))
                top_similar.sort(key=lambda x: x[1], reverse=True)

            if top_similar and top_similar[0][1] >= _SIMILARITY_DEDUP_THRESHOLD:
                best_fact, _ = top_similar[0]
                new_conf = 0.3 * confidence + 0.7 * best_fact.confidence
                updated = best_fact.model_copy(update={"confidence": new_conf})
                await self._store.update_fact(updated)
                facts_updated += 1
                continue

            maybe_contradicted: list[UserFact] = [
                f
                for f, sim in top_similar
                if _CONTRADICTION_MIN_SIMILARITY <= sim < _SIMILARITY_DEDUP_THRESHOLD
            ]

            if maybe_contradicted:
                contradicted_ids = await self._check_contradiction(
                    content, maybe_contradicted[:5], model_client
                )
                for fid in contradicted_ids:
                    await self._store.expire_fact(user_id, fid)
                    facts_expired += 1

            new_fact = UserFact(
                id=str(uuid.uuid4()),
                user_id=user_id,
                content=content,
                category=category,
                confidence=min(max(confidence, 0.0), 1.0),
                source_session_id=session_id,
                embedding=embedding,
            )
            stored = await self._store.store_fact(new_fact)
            new_fact_ids.append(stored.id)
            facts_extracted += 1

        return FactExtractionResult(
            user_id=user_id,
            session_id=session_id,
            facts_extracted=facts_extracted,
            facts_updated=facts_updated,
            facts_expired=facts_expired,
            new_fact_ids=new_fact_ids,
        )

    async def _check_contradiction(
        self,
        new_content: str,
        existing_facts: list[UserFact],
        model_client: Any,
    ) -> list[str]:
        """Return fact_ids that the new fact contradicts.

        On any error, returns empty list (prefer false negatives over false
        positives for expiry).
        """
        if not existing_facts:
            return []
        try:
            pairs = "\n".join(
                f"{i + 1}. New: '{new_content}' vs Existing: '{f.content}'"
                for i, f in enumerate(existing_facts)
            )
            from grampus.core.types import Message, Role  # noqa: PLC0415

            response = await model_client.complete(
                messages=[
                    Message(role=Role.SYSTEM, content=_CONTRADICTION_SYSTEM_PROMPT),
                    Message(role=Role.USER, content=f"Check contradictions:\n{pairs}"),
                ],
                model="claude-haiku-4-5-20251001",
                temperature=0.0,
            )
            text = (response.content or "").strip()
            start = text.find("[")
            end = text.rfind("]")
            if start == -1 or end == -1:
                return []
            flags: list[bool] = json.loads(text[start : end + 1])
            return [
                existing_facts[i].id
                for i, flag in enumerate(flags)
                if i < len(existing_facts) and flag
            ]
        except Exception:
            return []

    def _build_conversation_string(
        self,
        records: list[Any],
        max_chars: int,
    ) -> str:
        """Concatenate and truncate episodic record content."""
        parts = [r.content for r in records if r.content]
        combined = "\n".join(parts)
        return combined[:max_chars]

    def _parse_extraction_response(self, response_text: str) -> list[dict[str, Any]]:
        """Parse JSON from LLM response. Returns empty list on parse failure."""
        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start == -1 or end == 0:
                return []
            data = json.loads(response_text[start:end])
            return data.get("facts", [])  # type: ignore[no-any-return]
        except Exception:
            return []
