"""Tests for EmbeddingProvider implementations and refactored EmbeddingService."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from nexus.core.errors import EmbeddingError
from nexus.memory.embedding_providers import (
    CohereEmbeddingProvider,
    EmbeddingProvider,
    EmbeddingRouter,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from nexus.memory.embeddings import EmbeddingService, _cache_key

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.1, 0.2, 0.3]


class MockProvider(EmbeddingProvider):
    """Minimal concrete provider that records calls for assertions."""

    def __init__(self) -> None:
        self._calls: list[tuple[list[str], str]] = []

    @property
    def dimensions(self) -> int:
        return 5

    @property
    def provider_name(self) -> str:
        return "mock"

    async def embed_batch(
        self, texts: list[str], *, input_type: str = "search_document"
    ) -> list[list[float]]:
        self._calls.append((texts, input_type))
        return [[0.1] * 5 for _ in texts]


def _make_openai_response(embedding: list[float]) -> MagicMock:
    resp = MagicMock()
    resp.data = [MagicMock(embedding=embedding)]
    return resp


def _make_cohere_response(embeddings: list[list[float]]) -> MagicMock:
    resp = MagicMock()
    resp.embeddings = MagicMock()
    resp.embeddings.float_ = embeddings
    return resp


def _make_httpx_response(embeddings: list[list[float]]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"embeddings": embeddings})
    return resp


@pytest.fixture()
def mock_cache() -> AsyncMock:
    cache = AsyncMock()
    cache.save = AsyncMock(return_value=None)
    cache.get = AsyncMock(return_value=(None, ""))
    return cache


# ---------------------------------------------------------------------------
# OpenAIEmbeddingProvider
# ---------------------------------------------------------------------------


class TestOpenAIEmbeddingProvider:
    @pytest.fixture()
    def mock_openai(self) -> AsyncMock:
        client = AsyncMock()
        client.embeddings = AsyncMock()
        client.embeddings.create = AsyncMock(
            return_value=_make_openai_response(FAKE_EMBEDDING)
        )
        return client

    def test_openai_dimensions_small(self, mock_openai: AsyncMock) -> None:
        p = OpenAIEmbeddingProvider(mock_openai, model="text-embedding-3-small")
        assert p.dimensions == 1536

    def test_openai_dimensions_large(self, mock_openai: AsyncMock) -> None:
        p = OpenAIEmbeddingProvider(mock_openai, model="text-embedding-3-large")
        assert p.dimensions == 3072

    def test_openai_dimensions_unknown_model_defaults_1536(
        self, mock_openai: AsyncMock
    ) -> None:
        p = OpenAIEmbeddingProvider(mock_openai, model="some-future-model")
        assert p.dimensions == 1536

    def test_openai_provider_name(self, mock_openai: AsyncMock) -> None:
        p = OpenAIEmbeddingProvider(mock_openai)
        assert p.provider_name == "openai"

    async def test_openai_embed_batch_calls_create(self, mock_openai: AsyncMock) -> None:
        p = OpenAIEmbeddingProvider(mock_openai, model="text-embedding-3-small")
        result = await p.embed_batch(["hello"])
        mock_openai.embeddings.create.assert_called_once_with(
            input=["hello"], model="text-embedding-3-small"
        )
        assert result == [FAKE_EMBEDDING]

    async def test_openai_input_type_ignored(self, mock_openai: AsyncMock) -> None:
        p = OpenAIEmbeddingProvider(mock_openai)
        await p.embed_batch(["hello"], input_type="search_query")
        call_kwargs = mock_openai.embeddings.create.call_args.kwargs
        assert "input_type" not in call_kwargs

    async def test_openai_api_error_raises_embedding_error(
        self, mock_openai: AsyncMock
    ) -> None:
        mock_openai.embeddings.create.side_effect = RuntimeError("API down")
        p = OpenAIEmbeddingProvider(mock_openai)
        with pytest.raises(EmbeddingError) as exc_info:
            await p.embed_batch(["hello"])
        assert exc_info.value.code == "EMBEDDING_API_ERROR"


# ---------------------------------------------------------------------------
# CohereEmbeddingProvider
# ---------------------------------------------------------------------------


class TestCohereEmbeddingProvider:
    @pytest.fixture()
    def mock_cohere(self) -> AsyncMock:
        client = AsyncMock()
        client.embed = AsyncMock(return_value=_make_cohere_response([FAKE_EMBEDDING]))
        return client

    def test_cohere_dimensions_english_v3(self, mock_cohere: AsyncMock) -> None:
        p = CohereEmbeddingProvider(mock_cohere, model="embed-english-v3.0")
        assert p.dimensions == 1024

    def test_cohere_dimensions_light(self, mock_cohere: AsyncMock) -> None:
        p = CohereEmbeddingProvider(mock_cohere, model="embed-english-light-v3.0")
        assert p.dimensions == 384

    def test_cohere_provider_name(self, mock_cohere: AsyncMock) -> None:
        p = CohereEmbeddingProvider(mock_cohere)
        assert p.provider_name == "cohere"

    async def test_cohere_embed_batch_passes_input_type(
        self, mock_cohere: AsyncMock
    ) -> None:
        p = CohereEmbeddingProvider(mock_cohere)
        await p.embed_batch(["hello"], input_type="search_query")
        call_kwargs = mock_cohere.embed.call_args.kwargs
        assert call_kwargs["input_type"] == "search_query"

    async def test_cohere_embed_batch_default_input_type_is_search_document(
        self, mock_cohere: AsyncMock
    ) -> None:
        p = CohereEmbeddingProvider(mock_cohere)
        await p.embed_batch(["hello"])
        call_kwargs = mock_cohere.embed.call_args.kwargs
        assert call_kwargs["input_type"] == "search_document"

    async def test_cohere_embed_batch_returns_float_embeddings(
        self, mock_cohere: AsyncMock
    ) -> None:
        mock_cohere.embed.return_value = _make_cohere_response([[0.1, 0.2]])
        p = CohereEmbeddingProvider(mock_cohere)
        result = await p.embed_batch(["hello"])
        assert result == [[0.1, 0.2]]

    async def test_cohere_api_error_raises_embedding_error(
        self, mock_cohere: AsyncMock
    ) -> None:
        mock_cohere.embed.side_effect = RuntimeError("quota exceeded")
        p = CohereEmbeddingProvider(mock_cohere)
        with pytest.raises(EmbeddingError) as exc_info:
            await p.embed_batch(["hello"])
        assert exc_info.value.code == "EMBEDDING_API_ERROR"


# ---------------------------------------------------------------------------
# OllamaEmbeddingProvider
# ---------------------------------------------------------------------------


class TestOllamaEmbeddingProvider:
    @pytest.fixture()
    def mock_http(self) -> AsyncMock:
        client = AsyncMock()
        client.post = AsyncMock(return_value=_make_httpx_response([FAKE_EMBEDDING]))
        client.aclose = AsyncMock()
        return client

    def test_ollama_dimensions_nomic(self) -> None:
        p = OllamaEmbeddingProvider(model="nomic-embed-text")
        assert p.dimensions == 768

    def test_ollama_dimensions_mxbai(self) -> None:
        p = OllamaEmbeddingProvider(model="mxbai-embed-large")
        assert p.dimensions == 1024

    def test_ollama_dimensions_unknown_model_defaults_768(self) -> None:
        p = OllamaEmbeddingProvider(model="some-custom-model")
        assert p.dimensions == 768

    def test_ollama_provider_name(self) -> None:
        p = OllamaEmbeddingProvider()
        assert p.provider_name == "ollama"

    async def test_ollama_embed_batch_posts_to_api_embed(
        self, mock_http: AsyncMock
    ) -> None:
        p = OllamaEmbeddingProvider(model="nomic-embed-text", http_client=mock_http)
        await p.embed_batch(["hello", "world"])
        mock_http.post.assert_called_once()
        call = mock_http.post.call_args
        assert call.args[0].endswith("/api/embed")
        posted_json = call.kwargs["json"]
        assert posted_json["model"] == "nomic-embed-text"
        assert posted_json["input"] == ["hello", "world"]

    async def test_ollama_input_type_ignored(self, mock_http: AsyncMock) -> None:
        p = OllamaEmbeddingProvider(http_client=mock_http)
        await p.embed_batch(["hello"], input_type="search_query")
        posted_json = mock_http.post.call_args.kwargs["json"]
        assert "input_type" not in posted_json

    async def test_ollama_parses_embeddings_from_response(
        self, mock_http: AsyncMock
    ) -> None:
        mock_http.post.return_value = _make_httpx_response([[0.1, 0.2]])
        p = OllamaEmbeddingProvider(http_client=mock_http)
        result = await p.embed_batch(["hello"])
        assert result == [[0.1, 0.2]]

    async def test_ollama_http_error_raises_embedding_error(
        self, mock_http: AsyncMock
    ) -> None:
        mock_http.post.side_effect = httpx.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(spec=httpx.Request),
            response=MagicMock(spec=httpx.Response),
        )
        p = OllamaEmbeddingProvider(http_client=mock_http)
        with pytest.raises(EmbeddingError) as exc_info:
            await p.embed_batch(["hello"])
        assert exc_info.value.code == "EMBEDDING_API_ERROR"
        assert "ollama serve" in exc_info.value.hint

    async def test_ollama_connection_error_raises_embedding_error(
        self, mock_http: AsyncMock
    ) -> None:
        mock_http.post.side_effect = httpx.ConnectError("Connection refused")
        p = OllamaEmbeddingProvider(http_client=mock_http)
        with pytest.raises(EmbeddingError) as exc_info:
            await p.embed_batch(["hello"])
        assert exc_info.value.code == "EMBEDDING_API_ERROR"


# ---------------------------------------------------------------------------
# EmbeddingService with EmbeddingProvider
# ---------------------------------------------------------------------------


class TestEmbeddingServiceWithProvider:
    @pytest.fixture()
    def provider(self) -> MockProvider:
        return MockProvider()

    @pytest.fixture()
    def service(self, provider: MockProvider, mock_cache: AsyncMock) -> EmbeddingService:
        return EmbeddingService(provider, mock_cache)

    def test_service_dimensions_forwarded_from_provider(
        self, provider: MockProvider, mock_cache: AsyncMock
    ) -> None:
        svc = EmbeddingService(provider, mock_cache)
        assert svc.dimensions == 5

    async def test_service_embed_calls_provider_embed_batch(
        self, service: EmbeddingService, provider: MockProvider
    ) -> None:
        await service.embed("hello")
        assert len(provider._calls) == 1
        assert provider._calls[0][0] == ["hello"]

    async def test_service_embed_passes_input_type_to_provider(
        self, service: EmbeddingService, provider: MockProvider
    ) -> None:
        await service.embed("hello", input_type="search_query")
        assert provider._calls[0][1] == "search_query"

    async def test_service_embed_caches_result(
        self, provider: MockProvider
    ) -> None:
        cached_data: bytes | None = None

        async def fake_get(entity_type: str, key: str, cls: type) -> tuple[Any, str]:
            nonlocal cached_data
            if cached_data is not None:
                m = MagicMock()
                m.data = cached_data
                return m, "e1"
            return None, ""

        async def fake_save(entity_type: str, key: str, data: bytes) -> None:
            nonlocal cached_data
            cached_data = data

        cache = AsyncMock()
        cache.get = AsyncMock(side_effect=fake_get)
        cache.save = AsyncMock(side_effect=fake_save)

        svc = EmbeddingService(provider, cache)
        await svc.embed("test text")
        await svc.embed("test text")
        assert len(provider._calls) == 1

    async def test_service_embed_batch_gathers_all(
        self, service: EmbeddingService, provider: MockProvider
    ) -> None:
        result = await service.embed_batch(["a", "b"])
        assert len(result) == 2
        assert result == [[0.1] * 5, [0.1] * 5]

    def test_cache_key_includes_provider_name(self) -> None:
        key1 = _cache_key("openai", "text-embedding-3-small", "hello")
        key2 = _cache_key("cohere", "text-embedding-3-small", "hello")
        assert key1 != key2

    def test_cache_key_includes_model(self) -> None:
        key1 = _cache_key("openai", "text-embedding-3-small", "hello")
        key2 = _cache_key("openai", "text-embedding-3-large", "hello")
        assert key1 != key2


# ---------------------------------------------------------------------------
# EmbeddingRouter
# ---------------------------------------------------------------------------


class TestEmbeddingRouter:
    @pytest.fixture()
    def default_provider(self) -> MockProvider:
        return MockProvider()

    @pytest.fixture()
    def semantic_provider(self) -> MockProvider:
        return MockProvider()

    @pytest.fixture()
    def default_svc(
        self, default_provider: MockProvider, mock_cache: AsyncMock
    ) -> EmbeddingService:
        return EmbeddingService(default_provider, mock_cache)

    @pytest.fixture()
    def semantic_svc(
        self, semantic_provider: MockProvider, mock_cache: AsyncMock
    ) -> EmbeddingService:
        return EmbeddingService(semantic_provider, mock_cache)

    def test_router_requires_default_route(
        self, semantic_svc: EmbeddingService
    ) -> None:
        with pytest.raises(ValueError, match="default"):
            EmbeddingRouter({"semantic": semantic_svc})

    def test_router_returns_default_service_for_unknown_purpose(
        self, default_svc: EmbeddingService
    ) -> None:
        router = EmbeddingRouter({"default": default_svc})
        assert router.service_for("unknown_purpose") is default_svc

    def test_router_returns_registered_service_for_known_purpose(
        self, default_svc: EmbeddingService, semantic_svc: EmbeddingService
    ) -> None:
        router = EmbeddingRouter({"default": default_svc, "semantic": semantic_svc})
        assert router.service_for("semantic") is semantic_svc

    def test_router_dimensions_from_default_service(
        self, default_svc: EmbeddingService
    ) -> None:
        router = EmbeddingRouter({"default": default_svc})
        assert router.dimensions == 5

    async def test_router_embed_routes_to_correct_service(
        self,
        default_provider: MockProvider,
        semantic_provider: MockProvider,
        default_svc: EmbeddingService,
        semantic_svc: EmbeddingService,
    ) -> None:
        router = EmbeddingRouter({"default": default_svc, "semantic": semantic_svc})
        await router.embed("hello", purpose="semantic")
        assert len(semantic_provider._calls) == 1
        assert len(default_provider._calls) == 0

    async def test_router_embed_uses_default_when_purpose_unregistered(
        self, default_provider: MockProvider, default_svc: EmbeddingService
    ) -> None:
        router = EmbeddingRouter({"default": default_svc})
        await router.embed("hello", purpose="nonexistent")
        assert len(default_provider._calls) == 1

    async def test_router_embed_batch_passes_input_type(
        self, default_provider: MockProvider, default_svc: EmbeddingService
    ) -> None:
        router = EmbeddingRouter({"default": default_svc})
        await router.embed_batch(["hello"], input_type="search_query")
        assert default_provider._calls[0][1] == "search_query"
