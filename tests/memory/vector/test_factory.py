"""Tests for grampus.memory.vector.factory — create_vector_store()."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from grampus.core.errors import ConfigError
from grampus.memory.vector.base import VectorStoreType
from grampus.memory.vector.factory import create_vector_store
from grampus.memory.vector.pgvector import PgVectorStore
from grampus.memory.vector.pinecone import PineconeVectorStore
from grampus.memory.vector.qdrant import QdrantVectorStore
from grampus.memory.vector.weaviate import WeaviateVectorStore


def _cfg(**overrides):  # type: ignore[no-untyped-def]
    """Build a minimal VectorStoreConfig dict and instantiate it."""
    from grampus.core.config import VectorStoreConfig

    return VectorStoreConfig(**overrides)


def test_factory_pgvector_returns_pg_store() -> None:
    cfg = _cfg(type=VectorStoreType.PGVECTOR)
    store = create_vector_store(cfg, state_store=MagicMock())
    assert isinstance(store, PgVectorStore)


def test_factory_pinecone_missing_api_key_raises_config_error() -> None:
    cfg = _cfg(
        type=VectorStoreType.PINECONE,
        pinecone_index_host="https://idx.example.io",
    )
    with pytest.raises(ConfigError, match="GRAMPUS_MEMORY__VECTOR_STORE__PINECONE_API_KEY"):
        create_vector_store(cfg)


def test_factory_pinecone_missing_index_host_raises_config_error() -> None:
    cfg = _cfg(
        type=VectorStoreType.PINECONE,
        pinecone_api_key=SecretStr("pk-test"),
    )
    with pytest.raises(ConfigError, match="GRAMPUS_MEMORY__VECTOR_STORE__PINECONE_INDEX_HOST"):
        create_vector_store(cfg)


def test_factory_pinecone_returns_pinecone_store() -> None:
    cfg = _cfg(
        type=VectorStoreType.PINECONE,
        pinecone_api_key=SecretStr("pk-test"),
        pinecone_index_host="https://idx.example.io",
    )
    store = create_vector_store(cfg)
    assert isinstance(store, PineconeVectorStore)


def test_factory_weaviate_returns_weaviate_store() -> None:
    cfg = _cfg(type=VectorStoreType.WEAVIATE)
    store = create_vector_store(cfg)
    assert isinstance(store, WeaviateVectorStore)


def test_factory_weaviate_with_api_key() -> None:
    cfg = _cfg(type=VectorStoreType.WEAVIATE, weaviate_api_key=SecretStr("wv-key"))
    store = create_vector_store(cfg)
    assert isinstance(store, WeaviateVectorStore)


def test_factory_qdrant_returns_qdrant_store() -> None:
    cfg = _cfg(type=VectorStoreType.QDRANT)
    store = create_vector_store(cfg)
    assert isinstance(store, QdrantVectorStore)


def test_factory_qdrant_with_api_key() -> None:
    cfg = _cfg(type=VectorStoreType.QDRANT, qdrant_api_key=SecretStr("qd-key"))
    store = create_vector_store(cfg)
    assert isinstance(store, QdrantVectorStore)
