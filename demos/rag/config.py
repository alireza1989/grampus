"""RAG pipeline configuration — loaded from env vars, JSON, or YAML."""

from __future__ import annotations

import json
import os
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from nexus.core.errors import RAGError


class EmbeddingProviderName(StrEnum):
    openai = "openai"
    cohere = "cohere"
    ollama = "ollama"


class RAGConfig(BaseModel):
    db_url: str = "postgresql://localhost/nexus"
    namespace: str = "default"
    embedding_provider: EmbeddingProviderName = EmbeddingProviderName.openai
    embedding_model: str = "text-embedding-3-small"
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str = ""
    cohere_api_key: str = ""
    chunk_size: int = 512
    chunk_strategy: str = "recursive"
    enrich_chunks: bool = False
    top_k: int = 10
    rrf_k: int = 60
    max_context_chunks: int = 6
    model_id: str = "claude-haiku-4-5-20251001"
    anthropic_api_key: str = ""
    max_tokens: int = 1024
    temperature: float = 0.1

    @classmethod
    def from_file(cls, path: str) -> RAGConfig:
        """Load from a JSON or YAML file."""
        with open(path) as f:
            if path.endswith((".yaml", ".yml")):
                try:
                    import yaml

                    data: dict[str, Any] = yaml.safe_load(f)
                except ImportError as exc:
                    raise RAGError(
                        "PyYAML required for YAML config files: pip install pyyaml",
                        code="MISSING_DEPENDENCY",
                    ) from exc
            else:
                data = json.load(f)
        return cls(**data)

    @classmethod
    def from_env(cls) -> RAGConfig:
        """Load from RAG_* and provider API key environment variables."""
        mapping: dict[str, Any] = {
            "db_url": os.environ.get("RAG_DB_URL"),
            "namespace": os.environ.get("RAG_NAMESPACE"),
            "embedding_provider": os.environ.get("RAG_EMBEDDING_PROVIDER"),
            "embedding_model": os.environ.get("RAG_EMBEDDING_MODEL"),
            "ollama_base_url": os.environ.get("RAG_OLLAMA_BASE_URL"),
            "openai_api_key": os.environ.get("OPENAI_API_KEY"),
            "cohere_api_key": os.environ.get("COHERE_API_KEY"),
            "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY"),
            "model_id": os.environ.get("RAG_MODEL_ID"),
            "chunk_size": os.environ.get("RAG_CHUNK_SIZE"),
            "top_k": os.environ.get("RAG_TOP_K"),
        }
        filtered = {k: v for k, v in mapping.items() if v is not None}
        return cls(**filtered)
