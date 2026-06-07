"""Vector store adapters for the Nexus memory layer."""

from nexus.memory.vector.base import VectorRecord, VectorSearchResult, VectorStore, VectorStoreType
from nexus.memory.vector.factory import create_vector_store

__all__ = [
    "VectorRecord",
    "VectorSearchResult",
    "VectorStore",
    "VectorStoreType",
    "create_vector_store",
]
