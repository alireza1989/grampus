"""Unit tests for MemoryManager.list_records()."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.memory.manager import MemoryManager
from nexus.memory.types import EpisodicRecord, SemanticFact


def _make_manager(**overrides: object) -> MemoryManager:
    defaults: dict[str, object] = {
        "working_memory": MagicMock(),
        "episodic_memory": MagicMock(),
        "semantic_memory": MagicMock(),
        "procedural_memory": MagicMock(),
        "episodic_retriever": MagicMock(),
        "semantic_retriever": MagicMock(),
        "consolidation_pipeline": MagicMock(),
    }
    defaults.update(overrides)
    return MemoryManager(
        defaults["working_memory"],  # type: ignore[arg-type]
        defaults["episodic_memory"],  # type: ignore[arg-type]
        defaults["semantic_memory"],  # type: ignore[arg-type]
        defaults["procedural_memory"],  # type: ignore[arg-type]
        defaults["episodic_retriever"],  # type: ignore[arg-type]
        defaults["semantic_retriever"],  # type: ignore[arg-type]
        defaults["consolidation_pipeline"],  # type: ignore[arg-type]
        agent_id="test-agent",
    )


def _ep_record(
    record_id: str = "ep-1",
    agent_id: str = "test-agent",
    trust_score: float = 0.9,
    content: str = "episodic content",
    created_at: datetime | None = None,
) -> EpisodicRecord:
    return EpisodicRecord(
        id=record_id,
        agent_id=agent_id,
        session_id="s1",
        content=content,
        trust_score=trust_score,
        timestamp=created_at or datetime(2026, 6, 9, tzinfo=UTC),
    )


def _sem_fact(
    fact_id: str = "sf-1",
    subject: str = "test",
    confidence: float = 0.8,
) -> SemanticFact:
    return SemanticFact(
        id=fact_id,
        subject=subject,
        predicate="knows",
        object_value="something",
        confidence=confidence,
        created_at=datetime(2026, 6, 8, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_list_records_empty_when_no_backends() -> None:
    ep = MagicMock()
    ep.list_all = AsyncMock(return_value=[])
    sm = MagicMock()
    sm.list_all = AsyncMock(return_value=[])
    pr = MagicMock()
    pr.list_all = AsyncMock(return_value=[])
    wm = MagicMock()
    wm.get_messages = AsyncMock(return_value=[])
    mgr = _make_manager(
        working_memory=wm, episodic_memory=ep, semantic_memory=sm, procedural_memory=pr
    )
    result = await mgr.list_records()
    assert result == []


@pytest.mark.asyncio
async def test_list_records_delegates_to_episodic() -> None:
    rec = _ep_record()
    ep = MagicMock()
    ep.list_all = AsyncMock(return_value=[rec])
    sm = MagicMock()
    sm.list_all = AsyncMock(return_value=[])
    pr = MagicMock()
    pr.list_all = AsyncMock(return_value=[])
    wm = MagicMock()
    wm.get_messages = AsyncMock(return_value=[])
    mgr = _make_manager(
        working_memory=wm, episodic_memory=ep, semantic_memory=sm, procedural_memory=pr
    )
    result = await mgr.list_records(memory_type="episodic")
    assert len(result) == 1
    assert result[0]["id"] == "ep-1"
    assert result[0]["memory_type"] == "episodic"


@pytest.mark.asyncio
async def test_list_records_delegates_to_semantic() -> None:
    fact = _sem_fact()
    ep = MagicMock()
    ep.list_all = AsyncMock(return_value=[])
    sm = MagicMock()
    sm.list_all = AsyncMock(return_value=[fact])
    pr = MagicMock()
    pr.list_all = AsyncMock(return_value=[])
    wm = MagicMock()
    wm.get_messages = AsyncMock(return_value=[])
    mgr = _make_manager(
        working_memory=wm, episodic_memory=ep, semantic_memory=sm, procedural_memory=pr
    )
    result = await mgr.list_records(memory_type="semantic")
    assert len(result) == 1
    assert result[0]["id"] == "sf-1"
    assert result[0]["memory_type"] == "semantic"


@pytest.mark.asyncio
async def test_list_records_all_types_combined_and_sorted() -> None:
    older = _ep_record("ep-old", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    newer = _ep_record("ep-new", created_at=datetime(2026, 6, 1, tzinfo=UTC))
    ep = MagicMock()
    ep.list_all = AsyncMock(return_value=[older, newer])
    sm = MagicMock()
    sm.list_all = AsyncMock(return_value=[])
    pr = MagicMock()
    pr.list_all = AsyncMock(return_value=[])
    wm = MagicMock()
    wm.get_messages = AsyncMock(return_value=[])
    mgr = _make_manager(
        working_memory=wm, episodic_memory=ep, semantic_memory=sm, procedural_memory=pr
    )
    result = await mgr.list_records()
    ids = [r["id"] for r in result if r["memory_type"] == "episodic"]
    assert ids[0] == "ep-new"


@pytest.mark.asyncio
async def test_list_records_filter_by_agent_id() -> None:
    r1 = _ep_record("ep-1", agent_id="agent-a")
    r2 = _ep_record("ep-2", agent_id="agent-b")
    ep = MagicMock()
    ep.list_all = AsyncMock(return_value=[r1, r2])
    sm = MagicMock()
    sm.list_all = AsyncMock(return_value=[])
    pr = MagicMock()
    pr.list_all = AsyncMock(return_value=[])
    wm = MagicMock()
    wm.get_messages = AsyncMock(return_value=[])
    mgr = _make_manager(
        working_memory=wm, episodic_memory=ep, semantic_memory=sm, procedural_memory=pr
    )
    result = await mgr.list_records(agent_id="agent-a")
    assert all(r["agent_id"] == "agent-a" for r in result)
    assert len(result) == 1


@pytest.mark.asyncio
async def test_list_records_filter_by_min_trust() -> None:
    low = _ep_record("ep-low", trust_score=0.3)
    high = _ep_record("ep-high", trust_score=0.9)
    ep = MagicMock()
    ep.list_all = AsyncMock(return_value=[low, high])
    sm = MagicMock()
    sm.list_all = AsyncMock(return_value=[])
    pr = MagicMock()
    pr.list_all = AsyncMock(return_value=[])
    wm = MagicMock()
    wm.get_messages = AsyncMock(return_value=[])
    mgr = _make_manager(
        working_memory=wm, episodic_memory=ep, semantic_memory=sm, procedural_memory=pr
    )
    result = await mgr.list_records(min_trust=0.5)
    assert len(result) == 1
    assert result[0]["id"] == "ep-high"


@pytest.mark.asyncio
async def test_list_records_limit_and_offset() -> None:
    records = [
        _ep_record(f"ep-{i}", created_at=datetime(2026, 6, i + 1, tzinfo=UTC)) for i in range(5)
    ]
    ep = MagicMock()
    ep.list_all = AsyncMock(return_value=records)
    sm = MagicMock()
    sm.list_all = AsyncMock(return_value=[])
    pr = MagicMock()
    pr.list_all = AsyncMock(return_value=[])
    wm = MagicMock()
    wm.get_messages = AsyncMock(return_value=[])
    mgr = _make_manager(
        working_memory=wm, episodic_memory=ep, semantic_memory=sm, procedural_memory=pr
    )
    result = await mgr.list_records(limit=2, offset=1)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_list_records_backend_exception_returns_partial() -> None:
    rec = _ep_record()
    ep = MagicMock()
    ep.list_all = AsyncMock(return_value=[rec])
    sm = MagicMock()
    sm.list_all = AsyncMock(side_effect=RuntimeError("backend down"))
    pr = MagicMock()
    pr.list_all = AsyncMock(return_value=[])
    wm = MagicMock()
    wm.get_messages = AsyncMock(return_value=[])
    mgr = _make_manager(
        working_memory=wm, episodic_memory=ep, semantic_memory=sm, procedural_memory=pr
    )
    result = await mgr.list_records()
    assert any(r["memory_type"] == "episodic" for r in result)
    assert not any(r["memory_type"] == "semantic" for r in result)


@pytest.mark.asyncio
async def test_list_records_returns_normalized_dict_keys() -> None:
    rec = _ep_record()
    ep = MagicMock()
    ep.list_all = AsyncMock(return_value=[rec])
    sm = MagicMock()
    sm.list_all = AsyncMock(return_value=[])
    pr = MagicMock()
    pr.list_all = AsyncMock(return_value=[])
    wm = MagicMock()
    wm.get_messages = AsyncMock(return_value=[])
    mgr = _make_manager(
        working_memory=wm, episodic_memory=ep, semantic_memory=sm, procedural_memory=pr
    )
    result = await mgr.list_records()
    assert len(result) == 1
    required = {
        "id",
        "agent_id",
        "memory_type",
        "content",
        "trust_score",
        "created_at",
        "last_accessed",
        "metadata",
        "provenance",
    }
    assert required.issubset(result[0].keys())
