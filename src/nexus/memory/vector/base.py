"""VectorStore ABC and shared data models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class VectorStoreType(StrEnum):
    """Supported vector store backends."""

    PGVECTOR = "pgvector"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"


class VectorRecord(BaseModel):
    """A single vector record to be stored in a vector DB."""

    id: str
    vector: list[float]
    payload: dict[str, Any] = Field(default_factory=dict)


class VectorSearchResult(BaseModel):
    """A search result returned by a vector store."""

    id: str
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)


class VectorStore(ABC):
    """Abstract base class for all vector store adapters."""

    @abstractmethod
    async def ensure_collection(self, dimension: int) -> None:
        """Create the index/collection if it does not exist. Idempotent."""

    @abstractmethod
    async def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or overwrite records by ID."""

    @abstractmethod
    async def search(
        self,
        vector: list[float],
        top_k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Return top_k nearest neighbours."""

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete records by ID. Silent if ID not found."""

    async def close(self) -> None:  # noqa: B027
        """Release connections. Default: no-op."""
