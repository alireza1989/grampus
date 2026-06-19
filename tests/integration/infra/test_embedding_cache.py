"""Embedding cache tests against real Redis.

Uses a _RedisCacheStore adapter that implements the DaprStateStore duck-type
required by EmbeddingService (get/save with entity+key).
"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio
import redis.asyncio as aioredis

from grampus.memory.embedding_providers import EmbeddingProvider
from grampus.memory.embeddings import EmbeddingService

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Minimal Redis-backed cache store
# ---------------------------------------------------------------------------


class _RedisCacheStore:
    """DaprStateStore-compatible adapter backed by real Redis."""

    def __init__(self, client: aioredis.Redis) -> None:  # type: ignore[type-arg]
        self._r = client

    async def get(self, entity: str, key: str, model_class: Any) -> tuple[Any, str]:
        raw: bytes | None = await self._r.get(f"{entity}:{key}")
        if raw is None:
            return None, ""

        class _Wrap:
            data = raw

        return _Wrap(), "etag-1"

    async def save(self, entity: str, key: str, data: bytes, **_: Any) -> None:
        await self._r.set(f"{entity}:{key}", data)


# ---------------------------------------------------------------------------
# Fake embedding provider
# ---------------------------------------------------------------------------


class _FakeProvider(EmbeddingProvider):
    """Provider returning deterministic text-dependent vectors for testing."""

    def __init__(self, dim: int = 4, name: str = "fake") -> None:
        self._dim = dim
        self._name = name
        self.call_count = 0

    @property
    def dimensions(self) -> int:
        return self._dim

    @property
    def provider_name(self) -> str:
        return self._name

    async def embed_batch(
        self, texts: list[str], *, input_type: str = "search_document"
    ) -> list[list[float]]:
        self.call_count += 1
        result: list[list[float]] = []
        for t in texts:
            # Deterministic, text-dependent: different texts → different vectors
            padded = (t + "\0" * self._dim)[: self._dim]
            vec = [float(ord(c)) / 256.0 for c in padded]
            result.append(vec)
        return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def redis_client(redis_container: Any):  # type: ignore[misc]
    """Real async Redis client with a flushed DB for a clean test slate."""
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    client: aioredis.Redis = aioredis.from_url(f"redis://{host}:{port}")  # type: ignore[type-arg]
    await client.flushdb()
    yield client
    await client.aclose()


@pytest_asyncio.fixture()
async def cache_store(redis_client: aioredis.Redis) -> _RedisCacheStore:  # type: ignore[type-arg]
    return _RedisCacheStore(redis_client)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCacheMissThenHit:
    async def test_cache_miss_then_hit(self, cache_store: _RedisCacheStore) -> None:
        provider = _FakeProvider()
        svc = EmbeddingService(provider=provider, cache_store=cache_store)

        vec1 = await svc.embed("hello world")
        vec2 = await svc.embed("hello world")  # should hit cache

        assert provider.call_count == 1, (
            f"Provider should be called exactly once for two identical embed() calls, "
            f"got call_count={provider.call_count}"
        )
        assert vec1 == vec2, "Both calls should return the same vector"


class TestDifferentTextsDifferentCacheKeys:
    async def test_different_texts_different_cache_keys(
        self, cache_store: _RedisCacheStore
    ) -> None:
        provider = _FakeProvider()
        svc = EmbeddingService(provider=provider, cache_store=cache_store)

        # Use texts that differ in their first character so DIM=4 vectors differ
        vec_a = await svc.embed("apple")
        vec_b = await svc.embed("banana")

        assert provider.call_count == 2, (
            f"Provider should be called once per unique text, got {provider.call_count}"
        )
        assert vec_a != vec_b, (
            "Different texts must produce different cache entries (not collision)"
        )


class TestCacheSurvivesNewServiceInstance:
    async def test_cache_survives_new_service_instance(self, cache_store: _RedisCacheStore) -> None:
        provider1 = _FakeProvider(name="fake")
        svc1 = EmbeddingService(provider=provider1, cache_store=cache_store)
        await svc1.embed("cached text")
        assert provider1.call_count == 1

        # New service instance pointing at the same cache store
        provider2 = _FakeProvider(name="fake")
        svc2 = EmbeddingService(provider=provider2, cache_store=cache_store)
        await svc2.embed("cached text")

        assert provider2.call_count == 0, (
            "Second EmbeddingService with same provider name should hit Redis cache "
            f"and NOT call the provider; got call_count={provider2.call_count}"
        )
