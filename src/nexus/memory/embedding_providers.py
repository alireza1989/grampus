"""Embedding provider implementations: OpenAI, Cohere, and Ollama."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from nexus.memory.embeddings import EmbeddingService

import httpx

from nexus.core.errors import EmbeddingError
from nexus.core.logging import get_logger

_log = get_logger(__name__)

_OPENAI_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_COHERE_DIMENSIONS: dict[str, int] = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
}

_OLLAMA_DIMENSIONS: dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "qwen3-embedding": 2048,
}


class EmbeddingProvider(ABC):
    """Abstract base for all embedding backends."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Number of dimensions in the output vectors."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier used in cache keys, e.g. 'openai', 'cohere', 'ollama'."""

    @abstractmethod
    async def embed_batch(
        self,
        texts: list[str],
        *,
        input_type: str = "search_document",
    ) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: Strings to embed.
            input_type: Cohere Embed v3+ distinguishes 'search_document' (stored
                content) from 'search_query' (retrieval queries). Other providers
                ignore this parameter.
        """


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by the OpenAI embeddings API."""

    def __init__(self, client: Any, model: str = "text-embedding-3-small") -> None:
        self._client = client
        self._model = model

    @property
    def dimensions(self) -> int:
        return _OPENAI_DIMENSIONS.get(self._model, 1536)

    @property
    def provider_name(self) -> str:
        return "openai"

    async def embed_batch(
        self, texts: list[str], *, input_type: str = "search_document"
    ) -> list[list[float]]:
        try:
            response = await self._client.embeddings.create(input=texts, model=self._model)
            _log.debug("openai_embed_batch", model=self._model, count=len(texts))
            return cast(list[list[float]], [item.embedding for item in response.data])
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"OpenAI embedding error: {exc}",
                code="EMBEDDING_API_ERROR",
                details={"provider": "openai", "model": self._model},
            ) from exc


class CohereEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by the Cohere Embed v3 API (AsyncClientV2)."""

    def __init__(self, client: Any, model: str = "embed-english-v3.0") -> None:
        self._client = client
        self._model = model

    @property
    def dimensions(self) -> int:
        return _COHERE_DIMENSIONS.get(self._model, 1024)

    @property
    def provider_name(self) -> str:
        return "cohere"

    async def embed_batch(
        self, texts: list[str], *, input_type: str = "search_document"
    ) -> list[list[float]]:
        # Cohere REQUIRES input_type for v3+ models — never omit it.
        try:
            response = await self._client.embed(
                texts=texts,
                model=self._model,
                input_type=input_type,
                embedding_types=["float"],
            )
            _log.debug("cohere_embed_batch", model=self._model, count=len(texts))
            return cast(list[list[float]], list(response.embeddings.float_))
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"Cohere embedding error: {exc}",
                code="EMBEDDING_API_ERROR",
                details={"provider": "cohere", "model": self._model},
            ) from exc


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by Ollama's /api/embed endpoint via httpx."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        *,
        http_client: Any = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._http = http_client

    @property
    def dimensions(self) -> int:
        return _OLLAMA_DIMENSIONS.get(self._model, 768)

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def embed_batch(
        self, texts: list[str], *, input_type: str = "search_document"
    ) -> list[list[float]]:
        owned = self._http is None
        client: httpx.AsyncClient = httpx.AsyncClient(timeout=30.0) if owned else self._http
        try:
            response = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": texts},
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            _log.debug("ollama_embed_batch", model=self._model, count=len(texts))
            return cast(list[list[float]], data["embeddings"])
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"Ollama embed error: {exc}",
                code="EMBEDDING_API_ERROR",
                details={"provider": "ollama", "model": self._model},
                hint=f"Is Ollama running at {self._base_url}? Run: ollama serve",
            ) from exc
        finally:
            if owned:
                await client.aclose()


class EmbeddingRouter:
    """Routes embed() calls to different EmbeddingService instances by purpose.

    Purpose strings map to memory types: "episodic", "semantic", "working",
    "procedural", "default". Any unmapped purpose falls back to the "default"
    service. Duck-type compatible with EmbeddingService for .embed() /
    .embed_batch() / .dimensions — existing call sites accept a router with no
    code changes.

    Args:
        routes: Mapping of purpose string → EmbeddingService.
                Must include a "default" key.

    Example::

        router = EmbeddingRouter({
            "default":  EmbeddingService(OpenAIEmbeddingProvider(client, "text-embedding-3-small"), cache),
            "semantic": EmbeddingService(OpenAIEmbeddingProvider(client, "text-embedding-3-large"), cache),
            "working":  EmbeddingService(OllamaEmbeddingProvider("nomic-embed-text"), cache),
        })
        vector = await router.embed(text, purpose="semantic")
    """

    def __init__(self, routes: dict[str, EmbeddingService]) -> None:
        if "default" not in routes:
            raise ValueError("EmbeddingRouter requires a 'default' route.")
        self._routes = routes

    def service_for(self, purpose: str) -> EmbeddingService:
        """Return the EmbeddingService registered for *purpose*, or the default."""
        return self._routes.get(purpose, self._routes["default"])

    @property
    def dimensions(self) -> int:
        """Dimensions of the default provider (used when no purpose is specified)."""
        return self._routes["default"].dimensions

    async def embed(
        self,
        text: str,
        *,
        purpose: str = "default",
        input_type: str = "search_document",
    ) -> list[float]:
        """Embed *text* using the service registered for *purpose*."""
        return await self.service_for(purpose).embed(text, input_type=input_type)

    async def embed_batch(
        self,
        texts: list[str],
        *,
        purpose: str = "default",
        input_type: str = "search_document",
    ) -> list[list[float]]:
        """Embed a batch using the service registered for *purpose*."""
        return await self.service_for(purpose).embed_batch(texts, input_type=input_type)
