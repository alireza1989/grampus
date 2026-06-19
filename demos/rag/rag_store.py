"""asyncpg-backed pgvector store for RAG document chunks."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import asyncpg
from pydantic import BaseModel

from grampus.core.errors import RAGError
from grampus.core.logging import get_logger

_log = get_logger(__name__)

_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS rag_chunks (
    id          TEXT PRIMARY KEY,
    namespace   TEXT NOT NULL,
    document_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    context_header TEXT NOT NULL DEFAULT '',
    metadata    JSONB NOT NULL DEFAULT '{{}}',
    embedding   vector({dimensions}),
    ts_vector   tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_CHECK_DIMENSIONS = """
SELECT a.atttypmod AS dim
FROM pg_attribute a
JOIN pg_class c ON a.attrelid = c.oid
WHERE c.relname = 'rag_chunks' AND a.attname = 'embedding'
"""

_CREATE_HNSW = """
CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
ON rag_chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
"""

_CREATE_FTS_IDX = (
    "CREATE INDEX IF NOT EXISTS rag_chunks_fts_idx ON rag_chunks USING GIN (ts_vector)"
)
_CREATE_NS_IDX = (
    "CREATE INDEX IF NOT EXISTS rag_chunks_ns_idx ON rag_chunks (namespace, document_id)"
)

_UPSERT_CHUNK = """
INSERT INTO rag_chunks
    (id, namespace, document_id, source_path, chunk_index, content, context_header, metadata, embedding)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector)
ON CONFLICT (id) DO UPDATE SET
    content        = EXCLUDED.content,
    context_header = EXCLUDED.context_header,
    metadata       = EXCLUDED.metadata,
    embedding      = EXCLUDED.embedding
"""

_HYBRID_SEARCH = """
WITH vector_results AS (
    SELECT id,
           ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rank
    FROM rag_chunks
    WHERE namespace = $2
    ORDER BY embedding <=> $1::vector
    LIMIT $3
),
fts_results AS (
    SELECT id,
           ROW_NUMBER() OVER (ORDER BY ts_rank(ts_vector, query) DESC) AS rank
    FROM rag_chunks, plainto_tsquery('english', $4) query
    WHERE namespace = $2 AND ts_vector @@ query
    ORDER BY ts_rank(ts_vector, query) DESC
    LIMIT $3
),
rrf AS (
    SELECT
        COALESCE(v.id, f.id) AS id,
        (COALESCE(1.0 / ($5 + v.rank), 0.0) + COALESCE(1.0 / ($5 + f.rank), 0.0)) AS rrf_score
    FROM vector_results v
    FULL OUTER JOIN fts_results f ON v.id = f.id
)
SELECT c.id, c.content, c.context_header, c.source_path,
       c.chunk_index, c.metadata, r.rrf_score
FROM rrf r
JOIN rag_chunks c ON r.id = c.id
ORDER BY r.rrf_score DESC
LIMIT $6
"""

_DELETE_DOCUMENT = "DELETE FROM rag_chunks WHERE namespace = $1 AND document_id = $2"
_GET_STATS = """
SELECT COUNT(*) AS chunk_count, COUNT(DISTINCT document_id) AS document_count
FROM rag_chunks WHERE namespace = $1
"""


class ChunkRecord(BaseModel):
    id: str
    namespace: str
    document_id: str
    source_path: str
    chunk_index: int
    content: str
    context_header: str = ""
    metadata: dict[str, Any] = {}
    embedding: list[float]


class RetrievedChunk(BaseModel):
    id: str
    content: str
    context_header: str
    source_path: str
    chunk_index: int
    metadata: dict[str, Any]
    rrf_score: float


class RAGStore:
    """asyncpg-backed pgvector store for RAG document chunks.

    All operations are scoped by namespace for multi-tenant isolation.
    Uses HNSW index (not IVFFlat) for zero-training incremental inserts.
    Hybrid BM25 + vector search with Reciprocal Rank Fusion.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:  # type: ignore[type-arg]
        self._pool = pool

    @classmethod
    async def create(cls, db_url: str, *, dimensions: int) -> RAGStore:
        """Connect to PostgreSQL, create schema if needed, return ready store.

        Raises:
            RAGError: If pgvector extension is unavailable or dimension mismatch detected.
        """
        try:
            pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
        except Exception as exc:
            raise RAGError(
                f"Cannot connect to PostgreSQL: {exc}\nCheck RAG_DB_URL or --db-url.",
                code="DB_CONNECTION_FAILED",
            ) from exc
        store = cls(pool)
        await store.setup(dimensions)
        return store

    async def setup(self, dimensions: int) -> None:
        """Idempotent: enable pgvector, create table and indexes.

        Raises:
            RAGError: If existing table has different embedding dimensions.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_EXTENSION)
            row = await conn.fetchrow(_CHECK_DIMENSIONS)
            if row is not None and row["dim"] != dimensions:
                raise RAGError(
                    f"Existing rag_chunks table has dimension {row['dim']} "
                    f"but embedding service produces {dimensions}. "
                    "Either use the same embedding model or drop the table: "
                    "DROP TABLE rag_chunks;",
                    code="DIMENSION_MISMATCH",
                )
            await conn.execute(_CREATE_TABLE.format(dimensions=dimensions))
            await conn.execute(_CREATE_HNSW)
            await conn.execute(_CREATE_FTS_IDX)
            await conn.execute(_CREATE_NS_IDX)
        _log.info("rag_store_ready", dimensions=dimensions)

    async def upsert_chunks(self, chunks: list[ChunkRecord]) -> int:
        """Upsert chunks in batches of 100. Returns total upserted count."""
        if not chunks:
            return 0
        total = 0
        async with self._pool.acquire() as conn:
            for i in range(0, len(chunks), 100):
                batch = chunks[i : i + 100]
                rows = [
                    (
                        c.id,
                        c.namespace,
                        c.document_id,
                        c.source_path,
                        c.chunk_index,
                        c.content,
                        c.context_header,
                        json.dumps(c.metadata),
                        "[" + ",".join(str(x) for x in c.embedding) + "]",
                    )
                    for c in batch
                ]
                await conn.executemany(_UPSERT_CHUNK, rows)
                total += len(batch)
        return total

    async def delete_document(self, namespace: str, document_id: str) -> int:
        """Delete all chunks for a document. Returns deleted row count."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(_DELETE_DOCUMENT, namespace, document_id)
        return int(result.split()[-1])

    async def retrieve(
        self,
        query_embedding: list[float],
        query_text: str,
        *,
        namespace: str,
        top_k: int = 10,
        rrf_k: int = 60,
        limit: int = 6,
    ) -> list[RetrievedChunk]:
        """Hybrid BM25 + vector search with RRF fusion and lost-in-the-middle reordering."""
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                _HYBRID_SEARCH,
                embedding_str,
                namespace,
                top_k,
                query_text,
                float(rrf_k),
                limit,
            )
        chunks = [
            RetrievedChunk(
                id=row["id"],
                content=row["content"],
                context_header=row["context_header"],
                source_path=row["source_path"],
                chunk_index=row["chunk_index"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                rrf_score=float(row["rrf_score"]),
            )
            for row in rows
        ]
        return _reorder_lost_in_middle(chunks)

    async def get_stats(self, namespace: str) -> dict[str, Any]:
        """Return chunk_count and document_count for the namespace."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_GET_STATS, namespace)
        return {
            "chunk_count": row["chunk_count"] if row else 0,
            "document_count": row["document_count"] if row else 0,
        }

    async def close(self) -> None:
        await self._pool.close()


def _reorder_lost_in_middle(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Reorder so highest-scoring chunks appear at context start and end.

    LLMs recall context placed at the beginning and end of a window better than
    the middle. Interleaving puts rank-1 first, rank-2 last, rank-3 second, etc.
    """
    if len(chunks) <= 2:
        return chunks
    result: list[RetrievedChunk | None] = [None] * len(chunks)
    left, right = 0, len(chunks) - 1
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            result[left] = chunk
            left += 1
        else:
            result[right] = chunk
            right -= 1
    return [c for c in result if c is not None]


def make_document_id(source_path: str) -> str:
    """Stable 16-char document ID derived from source path."""
    return hashlib.sha256(source_path.encode()).hexdigest()[:16]
