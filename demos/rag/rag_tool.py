"""RAG retrieval tool factory for Nexus agents."""

from __future__ import annotations

from typing import Any

from demos.rag.config import RAGConfig
from demos.rag.rag_store import RAGStore, RetrievedChunk
from nexus.core.logging import get_logger
from nexus.core.types import ToolParameter
from nexus.memory.embeddings import EmbeddingService
from nexus.tools.registry import ToolRegistry

_log = get_logger(__name__)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant context found in the knowledge base."
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = f" ({chunk.context_header})" if chunk.context_header else ""
        source = f"{chunk.source_path}{header}"
        parts.append(f"[{i}] Source: {source}\n{chunk.content}")
    return "\n\n".join(parts)


def make_retrieve_tool(
    store: RAGStore,
    embedding_service: EmbeddingService,
    config: RAGConfig,
) -> tuple[ToolRegistry, Any]:
    """Create a retrieve_context tool bound to the given store and embedding service.

    Returns:
        (registry, tool_fn) — register registry tools with AgentRunner,
        or use tool_fn directly.
    """

    async def retrieve_context(query: str, top_k: int = 5) -> dict[str, Any]:
        """Search the knowledge base for document chunks relevant to the query.

        Always call this tool before answering questions that may require
        specific facts, figures, or document content.

        Args:
            query: Natural language search query.
            top_k: Number of chunks to retrieve (1-10).
        """
        try:
            top_k = max(1, min(10, top_k))
            query_embedding = await embedding_service.embed(query, input_type="search_query")
            chunks = await store.retrieve(
                query_embedding,
                query,
                namespace=config.namespace,
                top_k=config.top_k,
                rrf_k=config.rrf_k,
                limit=top_k,
            )
            context = _format_context(chunks)
            _log.debug("rag_retrieve", query=query, chunks_returned=len(chunks))
            return {
                "ok": True,
                "context": context,
                "chunk_count": len(chunks),
                "sources": [c.source_path for c in chunks],
            }
        except Exception as exc:
            _log.warning("rag_retrieve_error", error=str(exc))
            return {"ok": False, "error": f"Retrieval failed: {exc}", "code": "RETRIEVAL_ERROR"}

    registry = ToolRegistry()
    registry.register(
        retrieve_context,
        name="retrieve_context",
        description=(
            "Search the document knowledge base for relevant information. "
            "Returns numbered context chunks with source citations. "
            "Use [1], [2] notation in your answer to cite specific chunks."
        ),
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="Natural language search query.",
                required=True,
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="Number of chunks to retrieve (1-10, default 5).",
                required=False,
                default=5,
            ),
        ],
    )
    return registry, retrieve_context
