"""Factory for creating VectorStore instances from config."""

from __future__ import annotations

from typing import Any

from nexus.core.errors import ConfigError
from nexus.memory.vector.base import VectorStore, VectorStoreType


def create_vector_store(config: Any, state_store: Any = None) -> VectorStore:
    """Instantiate the correct VectorStore from config.

    Args:
        config: A VectorStoreConfig instance.
        state_store: Only required for the PGVECTOR backend.

    Raises:
        ConfigError: When required credentials are missing, with the env var
            name in the message so the user knows exactly what to set.
    """
    from nexus.memory.vector.pgvector import PgVectorStore  # noqa: PLC0415
    from nexus.memory.vector.pinecone import PineconeVectorStore  # noqa: PLC0415
    from nexus.memory.vector.qdrant import QdrantVectorStore  # noqa: PLC0415
    from nexus.memory.vector.weaviate import WeaviateVectorStore  # noqa: PLC0415

    match config.type:
        case VectorStoreType.PGVECTOR:
            return PgVectorStore(state_store=state_store)

        case VectorStoreType.PINECONE:
            if not config.pinecone_api_key:
                raise ConfigError(
                    "Pinecone API key is required. "
                    "Set NEXUS_MEMORY__VECTOR_STORE__PINECONE_API_KEY.",
                    code="config_missing_pinecone_api_key",
                )
            if not config.pinecone_index_host:
                raise ConfigError(
                    "Pinecone index host is required. "
                    "Set NEXUS_MEMORY__VECTOR_STORE__PINECONE_INDEX_HOST.",
                    code="config_missing_pinecone_index_host",
                )
            return PineconeVectorStore(
                api_key=config.pinecone_api_key.get_secret_value(),
                index_host=config.pinecone_index_host,
                namespace=config.pinecone_namespace,
                cloud=config.pinecone_cloud,
                region=config.pinecone_region,
            )

        case VectorStoreType.WEAVIATE:
            return WeaviateVectorStore(
                host=config.weaviate_host,
                port=config.weaviate_port,
                api_key=(
                    config.weaviate_api_key.get_secret_value() if config.weaviate_api_key else None
                ),
                collection_name=config.weaviate_collection,
            )

        case VectorStoreType.QDRANT:
            return QdrantVectorStore(
                url=config.qdrant_url,
                api_key=(
                    config.qdrant_api_key.get_secret_value() if config.qdrant_api_key else None
                ),
                collection_name=config.qdrant_collection,
            )

    raise ConfigError(
        f"Unknown vector store type: {config.type!r}",
        code="config_unknown_vector_store_type",
    )
