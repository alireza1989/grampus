"""RAG store lifecycle tests against real PostgreSQL + pgvector."""

from __future__ import annotations

import uuid

import pytest

from demos.rag.rag_store import ChunkRecord, RAGStore
from grampus.core.errors import RAGError

pytestmark = pytest.mark.integration

DIM = 4  # tiny dimension — no embedding API needed


def _ns() -> str:
    """Return a unique namespace for a test to prevent cross-test pollution."""
    return f"test-{uuid.uuid4().hex[:8]}"


def _vec(seed: float, dim: int = DIM) -> list[float]:
    """Return a deterministic unit-ish vector."""
    return [seed + i * 0.01 for i in range(dim)]


def _chunk(doc_id: str, idx: int, content: str, ns: str) -> ChunkRecord:
    return ChunkRecord(
        id=f"{doc_id}-{idx}",
        namespace=ns,
        document_id=doc_id,
        source_path=f"/docs/{doc_id}.txt",
        chunk_index=idx,
        content=content,
        context_header="Test > Section",
        metadata={},
        embedding=_vec(0.1 * (idx + 1)),
    )


class TestSetupCreatesSchema:
    async def test_setup_creates_schema(self, asyncpg_pool):
        ns = _ns()
        store = RAGStore(asyncpg_pool)
        await store.setup(DIM)

        async with asyncpg_pool.acquire() as conn:
            table_oid = await conn.fetchval("SELECT to_regclass('public.rag_chunks')")
            index_oid = await conn.fetchval("SELECT to_regclass('public.rag_chunks_embedding_idx')")

        assert table_oid is not None, f"Table not created (ns={ns})"
        assert index_oid is not None, "HNSW index not created"


class TestUpsertAndRetrieve:
    async def test_upsert_and_retrieve_returns_results(self, asyncpg_pool):
        ns = _ns()
        store = RAGStore(asyncpg_pool)
        await store.setup(DIM)

        doc_id = "doc-a"
        chunks = [_chunk(doc_id, i, f"chunk {i} content", ns) for i in range(5)]
        await store.upsert_chunks(chunks)

        results = await store.retrieve(
            query_embedding=_vec(0.1),
            query_text="chunk",
            namespace=ns,
            top_k=5,
            limit=5,
        )

        assert len(results) >= 1
        assert all(r.rrf_score > 0.0 for r in results)


class TestHybridSearchFTS:
    async def test_hybrid_search_fts_component(self, asyncpg_pool):
        ns = _ns()
        store = RAGStore(asyncpg_pool)
        await store.setup(DIM)

        doc_id = "doc-fts"
        texts = ["dapr state management", "redis cache store", "pubsub broker"]
        chunks = [_chunk(doc_id, i, t, ns) for i, t in enumerate(texts)]
        await store.upsert_chunks(chunks)

        results = await store.retrieve(
            query_embedding=_vec(0.1),
            query_text="dapr",
            namespace=ns,
            top_k=5,
            limit=5,
        )

        assert len(results) >= 1
        contents = [r.content for r in results]
        assert any("dapr" in c.lower() for c in contents), (
            f"Expected 'dapr' in results but got: {contents}"
        )


class TestDimensionMismatch:
    async def test_dimension_mismatch_raises(self, asyncpg_pool):
        store1 = RAGStore(asyncpg_pool)
        await store1.setup(DIM)  # creates table with DIM=4

        store2 = RAGStore(asyncpg_pool)
        with pytest.raises(RAGError) as exc_info:
            await store2.setup(dimensions=8)

        assert exc_info.value.code == "DIMENSION_MISMATCH"


class TestDeleteDocument:
    async def test_delete_document_removes_chunks(self, asyncpg_pool):
        ns = _ns()
        store = RAGStore(asyncpg_pool)
        await store.setup(DIM)

        doc_id = "doc-del"
        chunks = [_chunk(doc_id, i, f"content {i}", ns) for i in range(3)]
        await store.upsert_chunks(chunks)

        stats_before = await store.get_stats(ns)
        assert stats_before["chunk_count"] == 3

        await store.delete_document(ns, doc_id)

        stats_after = await store.get_stats(ns)
        assert stats_after["chunk_count"] == 0


class TestUpsertIdempotency:
    async def test_upsert_is_idempotent(self, asyncpg_pool):
        ns = _ns()
        store = RAGStore(asyncpg_pool)
        await store.setup(DIM)

        chunk = _chunk("doc-idem", 0, "idempotent content", ns)
        await store.upsert_chunks([chunk])
        await store.upsert_chunks([chunk])  # same chunk again

        stats = await store.get_stats(ns)
        assert stats["chunk_count"] == 1, (
            f"Expected 1 chunk after idempotent upsert, got {stats['chunk_count']}"
        )


class TestNamespaceIsolation:
    async def test_namespace_isolation(self, asyncpg_pool):
        ns_a = _ns()
        ns_b = _ns()
        store = RAGStore(asyncpg_pool)
        await store.setup(DIM)

        chunks_a = [_chunk("doc-a", i, f"ns-a content {i}", ns_a) for i in range(2)]
        chunks_b = [_chunk("doc-b", i, f"ns-b content {i}", ns_b) for i in range(2)]
        await store.upsert_chunks(chunks_a)
        await store.upsert_chunks(chunks_b)

        results = await store.retrieve(
            query_embedding=_vec(0.1),
            query_text="content",
            namespace=ns_a,
            top_k=10,
            limit=10,
        )

        # All returned ids start with the doc-a prefix
        for r in results:
            assert r.id.startswith("doc-a"), (
                f"Expected namespace isolation but got chunk from another ns: {r.id}"
            )


class TestGetStats:
    async def test_get_stats_correct(self, asyncpg_pool):
        ns = _ns()
        store = RAGStore(asyncpg_pool)
        await store.setup(DIM)

        chunks = [
            _chunk("doc-s1", 0, "stats doc1 chunk0", ns),
            _chunk("doc-s1", 1, "stats doc1 chunk1", ns),
            _chunk("doc-s2", 0, "stats doc2 chunk0", ns),
            _chunk("doc-s2", 1, "stats doc2 chunk1", ns),
        ]
        await store.upsert_chunks(chunks)

        stats = await store.get_stats(ns)
        assert stats["chunk_count"] == 4, f"Expected 4 chunks, got {stats['chunk_count']}"
        assert stats["document_count"] == 2, f"Expected 2 documents, got {stats['document_count']}"
