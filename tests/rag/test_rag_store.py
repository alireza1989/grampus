"""Tests for demos.rag.rag_store — pure-Python helpers (no DB needed)."""

from __future__ import annotations

from demos.rag.rag_store import (
    ChunkRecord,
    RetrievedChunk,
    _reorder_lost_in_middle,
    make_document_id,
)


def _make_chunk(rrf_score: float, idx: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"id-{idx}",
        content=f"content {idx}",
        context_header="",
        source_path="/doc.md",
        chunk_index=idx,
        metadata={},
        rrf_score=rrf_score,
    )


def test_make_document_id_deterministic() -> None:
    assert make_document_id("/docs/file.md") == make_document_id("/docs/file.md")


def test_make_document_id_differs() -> None:
    assert make_document_id("/docs/a.md") != make_document_id("/docs/b.md")


def test_reorder_single() -> None:
    chunks = [_make_chunk(0.9, 0)]
    assert _reorder_lost_in_middle(chunks) == chunks


def test_reorder_two() -> None:
    chunks = [_make_chunk(0.9, 0), _make_chunk(0.5, 1)]
    assert _reorder_lost_in_middle(chunks) == chunks


def test_reorder_six() -> None:
    chunks = [_make_chunk(1.0 - i * 0.1, i) for i in range(6)]
    result = _reorder_lost_in_middle(chunks)
    assert result[0] is chunks[0]
    assert result[5] is chunks[1]
    assert result[2] is chunks[4]


def test_chunk_record_roundtrip() -> None:
    record = ChunkRecord(
        id="abc123",
        namespace="default",
        document_id="doc42",
        source_path="/a/b.md",
        chunk_index=0,
        content="hello",
        context_header="Title",
        metadata={"k": "v"},
        embedding=[0.1, 0.2, 0.3],
    )
    d = record.model_dump()
    restored = ChunkRecord(**d)
    assert restored == record


def test_retrieved_chunk_roundtrip() -> None:
    chunk = RetrievedChunk(
        id="xyz",
        content="world",
        context_header="Sec",
        source_path="/x/y.pdf",
        chunk_index=2,
        metadata={"page": 5},
        rrf_score=0.42,
    )
    d = chunk.model_dump()
    restored = RetrievedChunk(**d)
    assert restored == chunk
