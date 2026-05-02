"""Tests for nexus.memory.embeddings — EmbeddingService with cache."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nexus.memory.embeddings import EmbeddingService, cosine_similarity

FAKE_EMBEDDING = [0.1, 0.2, 0.3, 0.4, 0.5]
ZERO_EMBEDDING = [0.0, 0.0, 0.0, 0.0, 0.0]


def make_openai_response(embedding: list[float]) -> MagicMock:
    resp = MagicMock()
    resp.data = [MagicMock(embedding=embedding)]
    return resp


@pytest.fixture()
def mock_openai() -> AsyncMock:
    client = AsyncMock()
    client.embeddings = AsyncMock()
    client.embeddings.create = AsyncMock(return_value=make_openai_response(FAKE_EMBEDDING))
    return client


@pytest.fixture()
def mock_cache() -> AsyncMock:
    cache = AsyncMock()
    cache.save = AsyncMock(return_value=None)
    cache.get = AsyncMock(return_value=(None, ""))
    return cache


@pytest.fixture()
def service(mock_openai: AsyncMock, mock_cache: AsyncMock) -> EmbeddingService:
    return EmbeddingService(openai_client=mock_openai, cache_store=mock_cache)


class TestCosineSimilarity:
    def test_identical_vectors_are_one(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert math.isclose(cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors_are_zero(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert math.isclose(cosine_similarity(a, b), 0.0, abs_tol=1e-9)

    def test_opposite_vectors_are_minus_one(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert math.isclose(cosine_similarity(a, b), -1.0)

    def test_is_commutative(self) -> None:
        a = [0.3, 0.7, 0.1]
        b = [0.9, 0.2, 0.5]
        assert math.isclose(cosine_similarity(a, b), cosine_similarity(b, a))

    def test_known_value(self) -> None:
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        expected = 1.0 / math.sqrt(2)
        assert math.isclose(cosine_similarity(a, b), expected, rel_tol=1e-6)

    def test_zero_vector_returns_zero(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 0.0]
        result = cosine_similarity(a, b)
        assert result == 0.0

    @given(
        st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=2, max_size=10),
        st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=2, max_size=10),
    )
    @settings(max_examples=50)
    def test_result_in_valid_range(self, a: list[float], b: list[float]) -> None:
        if len(a) != len(b):
            return
        result = cosine_similarity(a, b)
        assert -1.0 - 1e-6 <= result <= 1.0 + 1e-6


class TestEmbeddingServiceEmbed:
    async def test_cache_miss_calls_api(
        self, service: EmbeddingService, mock_openai: AsyncMock, mock_cache: AsyncMock
    ) -> None:
        mock_cache.get.return_value = (None, "")
        result = await service.embed("hello world")
        mock_openai.embeddings.create.assert_called_once()
        assert result == FAKE_EMBEDDING

    async def test_cache_hit_skips_api(
        self, service: EmbeddingService, mock_openai: AsyncMock, mock_cache: AsyncMock
    ) -> None:
        import json

        cached = json.dumps(FAKE_EMBEDDING).encode()
        mock_cache.get.return_value = (MagicMock(data=cached), "etag1")
        result = await service.embed("hello world")
        mock_openai.embeddings.create.assert_not_called()
        assert result == FAKE_EMBEDDING

    async def test_cache_miss_stores_result(
        self, service: EmbeddingService, mock_cache: AsyncMock
    ) -> None:
        mock_cache.get.return_value = (None, "")
        await service.embed("hello")
        mock_cache.save.assert_called_once()

    async def test_same_text_called_twice_api_once(
        self, mock_openai: AsyncMock, mock_cache: AsyncMock
    ) -> None:

        cached_data: bytes | None = None

        async def fake_get(entity_type: str, key: str, cls: type) -> tuple:  # type: ignore
            nonlocal cached_data
            if cached_data is not None:
                m = MagicMock()
                m.data = cached_data
                return m, "e1"
            return None, ""

        async def fake_save(entity_type: str, key: str, data: bytes) -> None:
            nonlocal cached_data
            cached_data = data

        mock_cache.get = AsyncMock(side_effect=fake_get)
        mock_cache.save = AsyncMock(side_effect=fake_save)

        service = EmbeddingService(openai_client=mock_openai, cache_store=mock_cache)
        await service.embed("test text")
        await service.embed("test text")
        assert mock_openai.embeddings.create.call_count == 1

    async def test_empty_text_still_calls_api(
        self, service: EmbeddingService, mock_openai: AsyncMock
    ) -> None:
        await service.embed("")
        mock_openai.embeddings.create.assert_called_once()

    async def test_returns_list_of_floats(self, service: EmbeddingService) -> None:
        result = await service.embed("hello")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)


class TestEmbeddingServiceEmbedBatch:
    async def test_batch_returns_one_embedding_per_text(
        self, service: EmbeddingService, mock_openai: AsyncMock
    ) -> None:
        texts = ["a", "b", "c"]
        results = await service.embed_batch(texts)
        assert len(results) == 3

    async def test_empty_batch_returns_empty(self, service: EmbeddingService) -> None:
        results = await service.embed_batch([])
        assert results == []

    async def test_batch_all_cached_calls_api_zero_times(
        self, service: EmbeddingService, mock_openai: AsyncMock, mock_cache: AsyncMock
    ) -> None:
        import json

        cached = json.dumps(FAKE_EMBEDDING).encode()
        mock_cache.get.return_value = (MagicMock(data=cached), "etag1")
        await service.embed_batch(["x", "y", "z"])
        mock_openai.embeddings.create.assert_not_called()

    async def test_cache_key_includes_model(
        self, mock_openai: AsyncMock, mock_cache: AsyncMock
    ) -> None:
        s1 = EmbeddingService(
            openai_client=mock_openai,
            cache_store=mock_cache,
            model="text-embedding-3-small",
        )
        s2 = EmbeddingService(
            openai_client=mock_openai,
            cache_store=mock_cache,
            model="text-embedding-3-large",
        )
        mock_cache.get.return_value = (None, "")
        await s1.embed("hello")
        await s2.embed("hello")
        keys = [str(call) for call in mock_cache.get.call_args_list]
        # The two calls should use different keys
        assert keys[0] != keys[1]
