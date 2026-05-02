"""Embedding service with Redis cache backed by Dapr."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from nexus.core.logging import get_logger

_log = get_logger(__name__)

_DEFAULT_MODEL = "text-embedding-3-small"
_CACHE_ENTITY = "embedding_cache"


class EmbeddingService:
    """Compute text embeddings via OpenAI API with Dapr-backed caching.

    Cache keys are SHA-256(model + ":" + text) so switching models never
    returns stale embeddings.

    Args:
        openai_client: Async OpenAI client with ``embeddings.create``.
        cache_store: A DaprStateStore (or duck-typed equivalent) for caching.
        model: Embedding model name.
    """

    def __init__(
        self,
        openai_client: Any,
        cache_store: Any,
        *,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._client = openai_client
        self._cache = cache_store
        self._model = model

    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text*, using the cache when possible."""
        cache_key = _cache_key(self._model, text)
        cached = await self._load_from_cache(cache_key)
        if cached is not None:
            return cached
        embedding = await self._call_api(text)
        await self._save_to_cache(cache_key, embedding)
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for all texts, parallelising API calls for cache misses."""
        import asyncio

        return list(await asyncio.gather(*[self.embed(t) for t in texts]))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_api(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            input=text,
            model=self._model,
        )
        embedding: list[float] = response.data[0].embedding
        _log.debug("embedding_computed", model=self._model, text_len=len(text))
        return embedding

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


def _cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}:{text}".encode()).hexdigest()


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
