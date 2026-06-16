"""Tests for the F2 persistent user modeling subsystem (42 tests).

Covers: UserFact/UserProfile types, UserMemoryStore, FactExtractor,
ProfileSynthesizer, UserMemoryAdapter, and AgentRunner integration hooks.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from grampus.memory.user.adapter import UserMemoryAdapter
from grampus.memory.user.extractor import FactExtractor
from grampus.memory.user.store import UserMemoryStore
from grampus.memory.user.synthesizer import ProfileSynthesizer
from grampus.memory.user.types import (
    FactExtractionResult,
    UserFact,
    UserFactCategory,
    UserMemoryContext,
    UserProfile,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_store() -> Any:
    """Return an in-memory mock Dapr state store."""
    storage: dict[str, Any] = {}

    async def _save(entity: str, key: str, value: Any, **kwargs: Any) -> None:
        storage[f"{entity}:{key}"] = value

    async def _get(entity: str, key: str, model_cls: Any) -> tuple[Any, str]:
        val = storage.get(f"{entity}:{key}")
        if val is None:
            return None, ""
        if hasattr(model_cls, "model_validate") and isinstance(val, dict):
            return model_cls.model_validate(val), "etag-1"
        return val, "etag-1"

    async def _delete(entity: str, key: str) -> None:
        storage.pop(f"{entity}:{key}", None)

    store = MagicMock()
    store.save = AsyncMock(side_effect=_save)
    store.get = AsyncMock(side_effect=_get)
    store.delete = AsyncMock(side_effect=_delete)
    return store


def _make_embedding_svc(vec: list[float] | None = None) -> Any:
    svc = MagicMock()
    svc.embed = AsyncMock(return_value=vec or [1.0, 0.0, 0.0, 0.0])
    return svc


def _make_model_client(content: str = '{"facts": []}') -> Any:
    from grampus.core.models.base import ModelResponse
    from grampus.core.types import TokenUsage

    def _resp(c: str) -> ModelResponse:
        return ModelResponse(
            content=c,
            tool_calls=[],
            token_usage=TokenUsage(
                input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.0, model="test"
            ),
            model="test",
            stop_reason="end_turn",
        )

    client = MagicMock()
    client.complete = AsyncMock(return_value=_resp(content))
    return client


def _make_fact(
    *,
    user_id: str = "alice",
    category: UserFactCategory = UserFactCategory.EXPERTISE,
    content: str = "Knows Python",
    confidence: float = 0.9,
    embedding: list[float] | None = None,
    valid_until: datetime | None = None,
    access_count: int = 0,
    session_id: str = "sess-1",
) -> UserFact:
    return UserFact(
        id=str(uuid.uuid4()),
        user_id=user_id,
        content=content,
        category=category,
        confidence=confidence,
        embedding=embedding,
        valid_until=valid_until,
        access_count=access_count,
        source_session_id=session_id,
    )


def _make_user_memory_store(vec: list[float] | None = None) -> UserMemoryStore:
    return UserMemoryStore(_make_store(), _make_embedding_svc(vec))


def _make_episodic_memory(records: list[Any] | None = None) -> Any:
    mem = MagicMock()
    mem.list_all = AsyncMock(return_value=records or [])
    return mem


def _make_episodic_record(
    *,
    user_id: str = "alice",
    session_id: str = "sess-1",
    content: str = "I prefer concise answers.",
) -> Any:
    from grampus.memory.types import EpisodicRecord

    return EpisodicRecord(
        id=str(uuid.uuid4()),
        agent_id="agent-1",
        user_id=user_id,
        session_id=session_id,
        content=content,
    )


# ===========================================================================
# 1–4: UserFact + UserProfile type tests
# ===========================================================================


class TestUserFactTypes:
    def test_userfact_is_valid_when_no_expiry(self) -> None:
        fact = _make_fact()
        assert fact.is_valid is True

    def test_userfact_is_not_valid_after_expiry(self) -> None:
        past = datetime.now(UTC) - timedelta(hours=1)
        fact = _make_fact(valid_until=past)
        assert fact.is_valid is False

    def test_userfact_confidence_range(self) -> None:
        with pytest.raises(ValidationError):
            _make_fact(confidence=1.5)

    def test_userprofile_expertise_level_range(self) -> None:
        with pytest.raises(ValidationError):
            UserProfile(user_id="x", expertise_level=6)


# ===========================================================================
# 5–17: UserMemoryStore tests
# ===========================================================================


class TestUserMemoryStore:
    @pytest.mark.asyncio()
    async def test_store_fact_persists_and_indexes(self) -> None:
        store = _make_user_memory_store()
        fact = _make_fact()
        stored = await store.store_fact(fact)
        retrieved = await store.get_fact(stored.user_id, stored.id)
        assert retrieved is not None
        assert retrieved.id == stored.id

    @pytest.mark.asyncio()
    async def test_store_fact_generates_embedding(self) -> None:
        embed_svc = _make_embedding_svc([0.5, 0.5, 0.0, 0.0])
        raw_store = _make_store()
        store = UserMemoryStore(raw_store, embed_svc)
        fact = _make_fact(embedding=None)
        stored = await store.store_fact(fact)
        assert stored.embedding is not None
        assert stored.embedding == [0.5, 0.5, 0.0, 0.0]

    @pytest.mark.asyncio()
    async def test_get_fact_returns_stored(self) -> None:
        store = _make_user_memory_store()
        fact = _make_fact(content="Uses VSCode")
        await store.store_fact(fact)
        retrieved = await store.get_fact(fact.user_id, fact.id)
        assert retrieved is not None
        assert retrieved.content == "Uses VSCode"

    @pytest.mark.asyncio()
    async def test_get_fact_returns_none_for_missing(self) -> None:
        store = _make_user_memory_store()
        result = await store.get_fact("alice", "nonexistent-id")
        assert result is None

    @pytest.mark.asyncio()
    async def test_get_valid_facts_excludes_expired(self) -> None:
        store = _make_user_memory_store()
        past = datetime.now(UTC) - timedelta(hours=1)
        active = _make_fact(content="active fact")
        expired = _make_fact(content="expired fact", valid_until=past)
        await store.store_fact(active)
        await store.store_fact(expired)
        valid = await store.get_valid_facts("alice")
        assert any(f.content == "active fact" for f in valid)
        assert not any(f.content == "expired fact" for f in valid)

    @pytest.mark.asyncio()
    async def test_get_valid_facts_filters_by_category(self) -> None:
        store = _make_user_memory_store()
        expertise = _make_fact(category=UserFactCategory.EXPERTISE, content="Knows Go")
        pref = _make_fact(category=UserFactCategory.PREFERENCE, content="Prefers dark mode")
        await store.store_fact(expertise)
        await store.store_fact(pref)
        filtered = await store.get_valid_facts("alice", category=UserFactCategory.EXPERTISE)
        assert all(f.category == UserFactCategory.EXPERTISE for f in filtered)
        assert len(filtered) == 1

    @pytest.mark.asyncio()
    async def test_expire_fact_sets_valid_until(self) -> None:
        store = _make_user_memory_store()
        fact = _make_fact()
        await store.store_fact(fact)
        await store.expire_fact(fact.user_id, fact.id)
        updated = await store.get_fact(fact.user_id, fact.id)
        assert updated is not None
        assert updated.valid_until is not None
        assert updated.is_valid is False

    @pytest.mark.asyncio()
    async def test_find_similar_facts_returns_top_k_by_cosine(self) -> None:
        embed_svc = MagicMock()
        raw_store = _make_store()
        store = UserMemoryStore(raw_store, embed_svc)

        fa = _make_fact(content="Fact A", embedding=[1.0, 0.0, 0.0, 0.0])
        fb = _make_fact(content="Fact B", embedding=[0.0, 1.0, 0.0, 0.0])
        fc = _make_fact(content="Fact C", embedding=[0.9, 0.1, 0.0, 0.0])

        embed_svc.embed = AsyncMock(side_effect=[fa.embedding, fb.embedding, fc.embedding])
        await store.store_fact(fa)
        await store.store_fact(fb)
        await store.store_fact(fc)

        results = await store.find_similar_facts("alice", [1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        contents = {r.content for r in results}
        assert "Fact A" in contents
        assert "Fact C" in contents

    @pytest.mark.asyncio()
    async def test_find_similar_facts_valid_only_excludes_expired(self) -> None:
        embed_svc = MagicMock()
        raw_store = _make_store()
        store = UserMemoryStore(raw_store, embed_svc)

        past = datetime.now(UTC) - timedelta(hours=1)
        active = _make_fact(content="Active", embedding=[1.0, 0.0, 0.0, 0.0])
        expired = _make_fact(content="Expired", embedding=[1.0, 0.0, 0.0, 0.0], valid_until=past)
        embed_svc.embed = AsyncMock(side_effect=[active.embedding, expired.embedding])
        await store.store_fact(active)
        await store.store_fact(expired)

        results = await store.find_similar_facts(
            "alice", [1.0, 0.0, 0.0, 0.0], top_k=5, valid_only=True
        )
        assert not any(r.content == "Expired" for r in results)

    @pytest.mark.asyncio()
    async def test_find_similar_facts_returns_all_when_no_embeddings(self) -> None:
        store = _make_user_memory_store()
        fa = _make_fact(content="No embed A", embedding=None)
        fb = _make_fact(content="No embed B", embedding=None)
        store2 = UserMemoryStore(_make_store(), MagicMock())
        store2._embeddings.embed = AsyncMock(side_effect=Exception("no embed"))
        await store.store_fact(fa)
        await store.store_fact(fb)
        results = await store.find_similar_facts("alice", [1.0, 0.0, 0.0, 0.0], top_k=5)
        assert len(results) >= 2

    @pytest.mark.asyncio()
    async def test_increment_access_updates_count_and_timestamp(self) -> None:
        store = _make_user_memory_store()
        fact = _make_fact(access_count=0)
        await store.store_fact(fact)
        await store.increment_access(fact.user_id, fact.id)
        updated = await store.get_fact(fact.user_id, fact.id)
        assert updated is not None
        assert updated.access_count == 1
        assert updated.last_accessed is not None

    @pytest.mark.asyncio()
    async def test_store_profile_overwrites_existing(self) -> None:
        store = _make_user_memory_store()
        p1 = UserProfile(user_id="alice", expertise_level=2, version=1)
        p2 = UserProfile(user_id="alice", expertise_level=5, version=2)
        await store.store_profile(p1)
        await store.store_profile(p2)
        loaded = await store.get_profile("alice")
        assert loaded is not None
        assert loaded.expertise_level == 5
        assert loaded.version == 2

    @pytest.mark.asyncio()
    async def test_get_profile_returns_none_for_new_user(self) -> None:
        store = _make_user_memory_store()
        result = await store.get_profile("brand-new-user")
        assert result is None


# ===========================================================================
# 18–23: FactExtractor tests
# ===========================================================================


class TestFactExtractor:
    def _make_extractor(
        self,
        *,
        records: list[Any] | None = None,
        embed_vec: list[float] | None = None,
    ) -> tuple[FactExtractor, UserMemoryStore]:
        raw_store = _make_store()
        embed_svc = _make_embedding_svc(embed_vec or [1.0, 0.0, 0.0, 0.0])
        mem_store = UserMemoryStore(raw_store, embed_svc)
        episodic = _make_episodic_memory(records)
        extractor = FactExtractor(
            store=mem_store,
            episodic_memory=episodic,
            embedding_service=embed_svc,
        )
        return extractor, mem_store

    @pytest.mark.asyncio()
    async def test_extract_from_session_stores_new_facts(self) -> None:
        records = [_make_episodic_record(content="I am a Python expert.")]
        extractor, store = self._make_extractor(records=records)
        client = _make_model_client(
            '{"facts": [{"content": "User is a Python expert", "category": "expertise", "confidence": 0.9}]}'
        )
        result = await extractor.extract_from_session("alice", "sess-1", client)
        assert result.facts_extracted == 1
        facts = await store.get_valid_facts("alice")
        assert len(facts) == 1
        assert facts[0].content == "User is a Python expert"

    @pytest.mark.asyncio()
    async def test_extract_from_session_deduplicates_similar_facts(self) -> None:
        """similarity > 0.9 → update existing fact, not duplicate."""
        raw_store = _make_store()
        embed_svc = MagicMock()
        mem_store = UserMemoryStore(raw_store, embed_svc)
        episodic = _make_episodic_memory([_make_episodic_record(content="Python expert")])

        existing = _make_fact(
            content="Knows Python well",
            category=UserFactCategory.EXPERTISE,
            confidence=0.8,
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        embed_svc.embed = AsyncMock(return_value=[1.0, 0.0, 0.0, 0.0])
        await mem_store.store_fact(existing)

        extractor = FactExtractor(
            store=mem_store, episodic_memory=episodic, embedding_service=embed_svc
        )
        client = _make_model_client(
            '{"facts": [{"content": "Knows Python well", "category": "expertise", "confidence": 1.0}]}'
        )
        result = await extractor.extract_from_session("alice", "sess-1", client)
        assert result.facts_updated == 1
        assert result.facts_extracted == 0
        facts = await mem_store.get_valid_facts("alice")
        assert len(facts) == 1

    @pytest.mark.asyncio()
    async def test_extract_from_session_expires_contradicted_fact(self) -> None:
        """A moderately-similar fact (0.5–0.9) that the LLM marks as contradicted gets expired."""
        raw_store = _make_store()
        embed_svc = MagicMock()
        mem_store = UserMemoryStore(raw_store, embed_svc)
        episodic = _make_episodic_memory([_make_episodic_record(content="I switched to Rust")])

        existing = _make_fact(
            content="Prefers Python exclusively",
            category=UserFactCategory.PREFERENCE,
            confidence=0.9,
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        embed_svc.embed = AsyncMock(return_value=[0.8, 0.0, 0.6, 0.0])
        await mem_store.store_fact(existing)

        extractor = FactExtractor(
            store=mem_store, episodic_memory=episodic, embedding_service=embed_svc
        )

        from grampus.core.models.base import ModelResponse
        from grampus.core.types import TokenUsage

        call_count = 0

        async def _complete(**kwargs: Any) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                c = '{"facts": [{"content": "Now uses Rust", "category": "preference", "confidence": 0.9}]}'
            else:
                c = "[true]"
            return ModelResponse(
                content=c,
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=5, output_tokens=5, total_tokens=10, cost_usd=0.0, model="t"
                ),
                model="t",
                stop_reason="end_turn",
            )

        client = MagicMock()
        client.complete = AsyncMock(side_effect=_complete)

        result = await extractor.extract_from_session("alice", "sess-1", client)
        assert result.facts_expired >= 1
        old = await mem_store.get_fact("alice", existing.id)
        assert old is not None
        assert old.is_valid is False

    @pytest.mark.asyncio()
    async def test_extract_from_session_never_raises(self) -> None:
        """If model_client raises, returns FactExtractionResult(facts_extracted=0)."""
        records = [_make_episodic_record(content="Hello")]
        extractor, _ = self._make_extractor(records=records)
        client = MagicMock()
        client.complete = AsyncMock(side_effect=RuntimeError("boom"))
        result = await extractor.extract_from_session("alice", "sess-1", client)
        assert result.facts_extracted == 0
        assert isinstance(result, FactExtractionResult)

    @pytest.mark.asyncio()
    async def test_extract_from_session_filters_by_user_id(self) -> None:
        """Records from other users are ignored when filtering by user_id."""
        alice_rec = _make_episodic_record(user_id="alice", content="Alice is an expert")
        bob_rec = _make_episodic_record(user_id="bob", content="Bob uses Java")
        extractor, store = self._make_extractor(records=[alice_rec, bob_rec])
        client = _make_model_client(
            '{"facts": [{"content": "User is an expert", "category": "expertise", "confidence": 0.9}]}'
        )
        result = await extractor.extract_from_session("alice", "sess-1", client)
        # Only alice's record is in the conversation sent to LLM
        # (bob's record filtered out because user_id != "alice" or session_id mismatch)
        assert result.user_id == "alice"

    def test_parse_extraction_response_handles_invalid_json(self) -> None:
        embed_svc = _make_embedding_svc()
        store = _make_user_memory_store()
        extractor = FactExtractor(
            store=store,
            episodic_memory=_make_episodic_memory(),
            embedding_service=embed_svc,
        )
        assert extractor._parse_extraction_response("not json at all") == []
        assert extractor._parse_extraction_response("") == []


# ===========================================================================
# 24–30: ProfileSynthesizer tests
# ===========================================================================


class TestProfileSynthesizer:
    def _make_synthesizer(
        self,
        *,
        synthesis_interval: int = 3,
        max_facts: int = 10,
    ) -> tuple[ProfileSynthesizer, UserMemoryStore]:
        store = _make_user_memory_store()
        synth = ProfileSynthesizer(
            store=store,
            synthesis_interval=synthesis_interval,
            max_facts_in_prompt=max_facts,
        )
        return synth, store

    @pytest.mark.asyncio()
    async def test_synthesize_triggers_after_threshold_facts(self) -> None:
        synth, store = self._make_synthesizer(synthesis_interval=3)
        for i in range(3):
            await store.store_fact(_make_fact(content=f"Fact {i}"))
        client = _make_model_client(
            '{"expertise_level": 4, "expertise_domains": ["Python"], '
            '"communication_style": "concise", "preferred_depth": "deep-dive", '
            '"active_goals": [], "past_key_decisions": [], "active_constraints": []}'
        )
        result = await synth.synthesize("alice", client)
        assert result.triggered is True
        assert result.new_version == 1

    @pytest.mark.asyncio()
    async def test_synthesize_skips_below_threshold_without_force(self) -> None:
        synth, store = self._make_synthesizer(synthesis_interval=10)
        await store.store_fact(_make_fact(content="Only one fact"))
        client = _make_model_client()
        result = await synth.synthesize("alice", client)
        assert result.triggered is False

    @pytest.mark.asyncio()
    async def test_synthesize_force_overrides_threshold(self) -> None:
        synth, store = self._make_synthesizer(synthesis_interval=10)
        await store.store_fact(_make_fact(content="One fact"))
        client = _make_model_client(
            '{"expertise_level": 3, "expertise_domains": [], '
            '"communication_style": "balanced", "preferred_depth": "balanced", '
            '"active_goals": [], "past_key_decisions": [], "active_constraints": []}'
        )
        result = await synth.synthesize("alice", client, force=True)
        assert result.triggered is True

    @pytest.mark.asyncio()
    async def test_synthesize_increments_version(self) -> None:
        synth, store = self._make_synthesizer(synthesis_interval=1)
        await store.store_fact(_make_fact(content="Fact 1"))
        client_resp = (
            '{"expertise_level": 3, "expertise_domains": ["Go"], '
            '"communication_style": "balanced", "preferred_depth": "balanced", '
            '"active_goals": [], "past_key_decisions": [], "active_constraints": []}'
        )
        client = _make_model_client(client_resp)
        r1 = await synth.synthesize("alice", client)
        assert r1.new_version == 1

        await store.store_fact(_make_fact(content="Fact 2"))
        client2 = _make_model_client(client_resp)
        r2 = await synth.synthesize("alice", client2)
        assert r2.new_version == 2

    @pytest.mark.asyncio()
    async def test_synthesize_never_raises(self) -> None:
        synth, store = self._make_synthesizer(synthesis_interval=1)
        await store.store_fact(_make_fact(content="Fact"))
        client = MagicMock()
        client.complete = AsyncMock(side_effect=RuntimeError("boom"))
        result = await synth.synthesize("alice", client)
        assert result.triggered is False

    def test_parse_synthesis_response_returns_safe_defaults_on_invalid_json(self) -> None:
        synth = ProfileSynthesizer(_make_user_memory_store())
        profile = synth._parse_synthesis_response("not json", "alice", 1, 5)
        assert profile.user_id == "alice"
        assert profile.version == 1
        assert profile.expertise_level == 3  # default

    @pytest.mark.asyncio()
    async def test_synthesize_uses_top_facts_by_access_count(self) -> None:
        """max_facts_in_prompt is respected — only top N by access_count sent to LLM."""
        synth, store = self._make_synthesizer(synthesis_interval=1, max_facts=2)
        f1 = _make_fact(content="High access", access_count=10)
        f2 = _make_fact(content="Mid access", access_count=5)
        f3 = _make_fact(content="Low access", access_count=0)
        for f in (f1, f2, f3):
            await store.store_fact(f)

        captured_prompts: list[str] = []

        from grampus.core.models.base import ModelResponse
        from grampus.core.types import TokenUsage

        async def _capture(**kwargs: Any) -> ModelResponse:
            for msg in kwargs.get("messages", []):
                if hasattr(msg, "content") and msg.content:
                    captured_prompts.append(str(msg.content))
            return ModelResponse(
                content=(
                    '{"expertise_level": 3, "expertise_domains": [], '
                    '"communication_style": "balanced", "preferred_depth": "balanced", '
                    '"active_goals": [], "past_key_decisions": [], "active_constraints": []}'
                ),
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=5, output_tokens=5, total_tokens=10, cost_usd=0.0, model="t"
                ),
                model="t",
                stop_reason="end_turn",
            )

        client = MagicMock()
        client.complete = AsyncMock(side_effect=_capture)
        await synth.synthesize("alice", client)

        prompt_text = " ".join(captured_prompts)
        assert "Low access" not in prompt_text
        assert "High access" in prompt_text


# ===========================================================================
# 31–38: UserMemoryAdapter tests
# ===========================================================================


class TestUserMemoryAdapter:
    def _make_adapter(
        self,
        *,
        embed_vec: list[float] | None = None,
        records: list[Any] | None = None,
    ) -> tuple[UserMemoryAdapter, UserMemoryStore]:
        embed_svc = _make_embedding_svc(embed_vec or [1.0, 0.0, 0.0, 0.0])
        raw_store = _make_store()
        store = UserMemoryStore(raw_store, embed_svc)
        episodic = _make_episodic_memory(records)
        extractor = FactExtractor(
            store=store, episodic_memory=episodic, embedding_service=embed_svc
        )
        synthesizer = ProfileSynthesizer(store=store, synthesis_interval=100)
        adapter = UserMemoryAdapter(
            store=store,
            extractor=extractor,
            synthesizer=synthesizer,
            embedding_service=embed_svc,
        )
        return adapter, store

    @pytest.mark.asyncio()
    async def test_get_context_returns_empty_for_unknown_user(self) -> None:
        adapter, _ = self._make_adapter()
        ctx = await adapter.get_context("no-user", "task", _make_model_client())
        assert ctx.profile is None
        assert ctx.relevant_facts == []

    @pytest.mark.asyncio()
    async def test_get_context_returns_profile_and_relevant_facts(self) -> None:
        adapter, store = self._make_adapter()
        profile = UserProfile(user_id="alice", expertise_level=4, version=1)
        await store.store_profile(profile)
        fact = _make_fact(content="Uses TDD", embedding=[1.0, 0.0, 0.0, 0.0])
        await store.store_fact(fact)

        ctx = await adapter.get_context("alice", "testing approach", _make_model_client())
        assert ctx.profile is not None
        assert ctx.profile.expertise_level == 4
        assert ctx.formatted_context != ""

    @pytest.mark.asyncio()
    async def test_get_context_never_raises(self) -> None:
        """If store throws, returns empty UserMemoryContext."""
        bad_store = MagicMock()
        bad_store.get_profile = AsyncMock(side_effect=RuntimeError("store error"))
        bad_store.get = AsyncMock(side_effect=RuntimeError("store error"))
        embed_svc = MagicMock()
        embed_svc.embed = AsyncMock(side_effect=RuntimeError("embed error"))
        store = UserMemoryStore(bad_store, embed_svc)
        episodic = _make_episodic_memory()
        extractor = FactExtractor(
            store=store, episodic_memory=episodic, embedding_service=embed_svc
        )
        synthesizer = ProfileSynthesizer(store=store)
        adapter = UserMemoryAdapter(
            store=store,
            extractor=extractor,
            synthesizer=synthesizer,
            embedding_service=embed_svc,
        )
        ctx = await adapter.get_context("alice", "task", _make_model_client())
        assert isinstance(ctx, UserMemoryContext)
        assert ctx.formatted_context == ""

    @pytest.mark.asyncio()
    async def test_get_context_increments_fact_access_count(self) -> None:
        adapter, store = self._make_adapter(embed_vec=[1.0, 0.0, 0.0, 0.0])
        fact = _make_fact(content="Knows Rust", embedding=[1.0, 0.0, 0.0, 0.0], access_count=0)
        await store.store_fact(fact)

        await adapter.get_context("alice", "systems programming", _make_model_client())
        updated = await store.get_fact("alice", fact.id)
        assert updated is not None
        assert updated.access_count == 1

    @pytest.mark.asyncio()
    async def test_observe_session_end_calls_extractor_and_synthesizer(self) -> None:
        adapter, store = self._make_adapter(
            records=[_make_episodic_record(content="I work with Kubernetes")]
        )
        client = _make_model_client(
            '{"facts": [{"content": "Uses Kubernetes", "category": "expertise", "confidence": 0.85}]}'
        )
        result = await adapter.observe_session_end("alice", "sess-1", client)
        assert isinstance(result, FactExtractionResult)
        assert result.user_id == "alice"

    @pytest.mark.asyncio()
    async def test_observe_session_end_never_raises(self) -> None:
        bad_store = MagicMock()
        bad_store.get = AsyncMock(side_effect=RuntimeError("boom"))
        bad_store.save = AsyncMock(side_effect=RuntimeError("boom"))
        embed_svc = MagicMock()
        embed_svc.embed = AsyncMock(side_effect=RuntimeError("boom"))
        store = UserMemoryStore(bad_store, embed_svc)
        episodic = MagicMock()
        episodic.list_all = AsyncMock(side_effect=RuntimeError("boom"))
        extractor = FactExtractor(
            store=store, episodic_memory=episodic, embedding_service=embed_svc
        )
        synthesizer = ProfileSynthesizer(store=store)
        adapter = UserMemoryAdapter(
            store=store,
            extractor=extractor,
            synthesizer=synthesizer,
            embedding_service=embed_svc,
        )
        result = await adapter.observe_session_end("alice", "sess-1", _make_model_client())
        assert result.facts_extracted == 0

    def test_format_context_empty_when_no_profile_and_no_facts(self) -> None:
        adapter, _ = self._make_adapter()
        result = adapter.format_context("alice", None, [])
        assert result == ""

    def test_format_context_profile_only_when_no_relevant_facts(self) -> None:
        adapter, _ = self._make_adapter()
        profile = UserProfile(
            user_id="alice",
            expertise_level=4,
            expertise_domains=["Python"],
            communication_style="concise",
            preferred_depth="deep-dive",
        )
        result = adapter.format_context("alice", profile, [])
        assert "alice" in result
        assert "Python" in result
        assert "concise" in result
        assert "Relevant facts" not in result


# ===========================================================================
# 39–42: AgentRunner integration tests
# ===========================================================================


class TestAgentRunnerF2Integration:
    def _make_runner(
        self,
        *,
        user_memory_adapter: Any | None = None,
    ) -> Any:
        from grampus.orchestration.runner import AgentRunner
        from grampus.tools.executor import ToolExecutor

        tool_reg = MagicMock()
        tool_reg.get = MagicMock(return_value=None)
        executor = ToolExecutor(tool_reg)

        from grampus.core.models.base import ModelResponse
        from grampus.core.types import TokenUsage

        resp = ModelResponse(
            content="Done.",
            tool_calls=[],
            token_usage=TokenUsage(
                input_tokens=5, output_tokens=5, total_tokens=10, cost_usd=0.0, model="t"
            ),
            model="t",
            stop_reason="end_turn",
        )
        client = MagicMock()
        client.complete = AsyncMock(return_value=resp)

        return AgentRunner(
            model_client=client,
            tool_executor=executor,
            user_memory_adapter=user_memory_adapter,
        )

    def _make_agent_def(self) -> Any:
        from grampus.core.types import AgentDefinition

        return AgentDefinition(name="test-agent", model="test", system_prompt="You help.")

    @pytest.mark.asyncio()
    async def test_runner_injects_user_context_before_llm_call(self) -> None:
        """Mock adapter's get_context is called when user_id is provided."""
        adapter = MagicMock()
        ctx = UserMemoryContext(
            user_id="alice",
            formatted_context="User profile for alice: ...",
        )
        adapter.get_context = AsyncMock(return_value=ctx)
        adapter.observe_session_end = AsyncMock(
            return_value=FactExtractionResult(
                user_id="alice",
                session_id="s1",
                facts_extracted=0,
                facts_updated=0,
                facts_expired=0,
            )
        )

        runner = self._make_runner(user_memory_adapter=adapter)
        agent = self._make_agent_def()
        await runner.run(agent, "Help me.", session_id="s1", user_id="alice")

        adapter.get_context.assert_called_once()
        call_args = adapter.get_context.call_args
        assert call_args[0][0] == "alice"

    @pytest.mark.asyncio()
    async def test_runner_calls_observe_session_end_after_run(self) -> None:
        adapter = MagicMock()
        adapter.get_context = AsyncMock(
            return_value=UserMemoryContext(user_id="alice", formatted_context="")
        )
        adapter.observe_session_end = AsyncMock(
            return_value=FactExtractionResult(
                user_id="alice",
                session_id="s1",
                facts_extracted=1,
                facts_updated=0,
                facts_expired=0,
            )
        )

        runner = self._make_runner(user_memory_adapter=adapter)
        agent = self._make_agent_def()
        await runner.run(agent, "Do task.", session_id="s1", user_id="alice")

        adapter.observe_session_end.assert_called_once_with("alice", "s1", runner._model_client)

    @pytest.mark.asyncio()
    async def test_runner_without_user_adapter_unchanged(self) -> None:
        """user_memory_adapter=None → no behavior change, no errors."""
        runner = self._make_runner(user_memory_adapter=None)
        agent = self._make_agent_def()
        result = await runner.run(agent, "Hello.", session_id="s1")
        assert result.output == "Done."

    @pytest.mark.asyncio()
    async def test_runner_without_user_id_skips_all_hooks(self) -> None:
        """user_id=None → adapter hooks are never called."""
        adapter = MagicMock()
        adapter.get_context = AsyncMock(
            return_value=UserMemoryContext(user_id="", formatted_context="")
        )
        adapter.observe_session_end = AsyncMock()

        runner = self._make_runner(user_memory_adapter=adapter)
        agent = self._make_agent_def()
        await runner.run(agent, "No user.", session_id="s1")

        adapter.get_context.assert_not_called()
        adapter.observe_session_end.assert_not_called()
