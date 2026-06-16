"""Vector store adapters for the Nexus memory layer."""

from grampus.memory.vector.base import (
    VectorRecord,
    VectorSearchResult,
    VectorStore,
    VectorStoreType,
)
from grampus.memory.vector.factory import create_vector_store

__all__ = [
    "VectorRecord",
    "VectorSearchResult",
    "VectorStore",
    "VectorStoreType",
    "create_vector_store",
]
