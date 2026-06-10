"""Embedding service with provider abstraction and Dapr-backed cache."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from nexus.core.logging import get_logger
from nexus.memory.embedding_providers import EmbeddingProvider

_log = get_logger(__name__)

_CACHE_ENTITY = "embedding_cache"


class EmbeddingService:
    """Unified embedding facade backed by any EmbeddingProvider.

    Preserved interface: .embed(text), .embed_batch(texts).
    All call sites in the memory layer are unchanged.
    New: .dimensions property for pgvector column validation.
    New: input_type param on embed/embed_batch (used by Cohere; ignored by others).

    Args:
        provider: Any EmbeddingProvider implementation.
        cache_store: Dapr state store (or duck-typed equivalent) for caching.
        openai_client: Deprecated — pass provider= instead. Accepted for
            backwards compatibility; wraps the client in OpenAIEmbeddingProvider.
        model: Model name; only used with the deprecated openai_client path.
    """

    def __init__(
        self,
        provider: EmbeddingProvider | None = None,
        cache_store: Any = None,
        *,
        openai_client: Any = None,
        model: str = "text-embedding-3-small",
    ) -> None:
        if provider is None and openai_client is not None:
            from nexus.memory.embedding_providers import OpenAIEmbeddingProvider

            provider = OpenAIEmbeddingProvider(openai_client, model)
        if provider is None:
            raise TypeError("provider is required (or pass openai_client= for backwards compat)")
        if cache_store is None:
            raise TypeError("cache_store is required")
        self._provider: EmbeddingProvider = provider
        self._cache: Any = cache_store

    @property
    def dimensions(self) -> int:
        """Number of dimensions in vectors produced by this service."""
        return self._provider.dimensions

    @property
    def model(self) -> str:
        """Model identifier forwarded from the provider (for logging)."""
        m = getattr(self._provider, "_model", None)
        if isinstance(m, str):
            return m
        return self._provider.provider_name

    async def embed(self, text: str, *, input_type: str = "search_document") -> list[float]:
        """Return the embedding vector for *text*, using the cache when possible."""
        key = _cache_key(self._provider.provider_name, self.model, text)
        cached = await self._load_from_cache(key)
        if cached is not None:
            return cached
        vectors = await self._provider.embed_batch([text], input_type=input_type)
        embedding = vectors[0]
        await self._save_to_cache(key, embedding)
        return embedding

    async def embed_batch(
        self, texts: list[str], *, input_type: str = "search_document"
    ) -> list[list[float]]:
        """Return embeddings for all texts. Cache hits are served without API calls."""
        import asyncio

        async def _one(t: str) -> list[float]:
            return await self.embed(t, input_type=input_type)

        return list(await asyncio.gather(*[_one(t) for t in texts]))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _load_from_cache(self, key: str) -> list[float] | None:
        result, _ = await self._cache.get(_CACHE_ENTITY, key, _CacheEntry)
        if result is None:
            return None
        try:
            return json.loads(result.data)  # type: ignore[no-any-return]
        except Exception:
            return None

    async def _save_to_cache(self, key: str, embedding: list[float]) -> None:
        data = json.dumps(embedding).encode()
        await self._cache.save(_CACHE_ENTITY, key, data)


class _CacheEntry:
    """Minimal class used as a type hint for the cache store get() call."""

    data: bytes


def _cache_key(provider: str, model: str, text: str) -> str:
    return hashlib.sha256(f"{provider}:{model}:{text}".encode()).hexdigest()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two vectors.

    Returns 0.0 if either vector is the zero vector.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
