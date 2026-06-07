"""NexusConfig — hierarchical configuration loaded from env vars and YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, YamlConfigSettingsSource

from nexus.memory.vector.base import VectorStoreType


class VectorStoreConfig(BaseModel):
    """Vector store backend selection and credentials."""

    type: VectorStoreType = VectorStoreType.PGVECTOR

    # Pinecone (cloud)
    pinecone_api_key: SecretStr | None = None
    pinecone_index_host: str | None = None
    pinecone_namespace: str = "nexus"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Weaviate
    weaviate_host: str = "localhost"
    weaviate_port: int = 8080
    weaviate_api_key: SecretStr | None = None
    weaviate_collection: str = "NexusMemory"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "nexus_memory"


class ModelConfig(BaseSettings):
    """LLM provider settings."""

    default_model: str = "claude-3-5-haiku-20241022"
    temperature: float = 0.0
    max_tokens: int = 4096
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    ollama_host: str = "http://localhost:11434"

    model_config = {"env_prefix": "NEXUS_MODEL__", "extra": "ignore"}


class MemoryConfig(BaseSettings):
    """Memory subsystem settings."""

    working_memory_token_limit: int = 100_000
    episodic_top_k: int = 5
    decay_rate: float = 0.01
    summarization_strategy: str = "hybrid"
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)

    model_config = {"env_prefix": "NEXUS_MEMORY__", "extra": "ignore"}


class SafetyConfig(BaseSettings):
    """Safety guardrail settings."""

    injection_detection_level: str = "balanced"
    pii_detection_enabled: bool = True
    action_rate_limit_per_minute: int = 60

    model_config = {"env_prefix": "NEXUS_SAFETY__", "extra": "ignore"}


class DaprConfig(BaseSettings):
    """Dapr sidecar connection settings."""

    host: str = "localhost"
    port: int = 3500
    grpc_port: int = 50001
    state_store_name: str = "statestore"
    pubsub_name: str = "pubsub"
    cache_store_name: str = "cache"

    model_config = {"env_prefix": "NEXUS_DAPR__", "extra": "ignore"}

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class ObservabilityConfig(BaseSettings):
    """Observability / telemetry settings."""

    otel_enabled: bool = True
    otel_endpoint: str = "http://localhost:4317"
    log_level: str = "INFO"
    metrics_enabled: bool = True

    model_config = {"env_prefix": "NEXUS_OBSERVABILITY__", "extra": "ignore"}


def _yaml_path() -> str | None:
    """Resolve the YAML config file path from env or default locations."""
    env_path = os.environ.get("NEXUS_CONFIG_FILE")
    if env_path:
        return env_path
    for candidate in ("nexus.yaml", "nexus.yml"):
        if Path(candidate).exists():
            return candidate
    return None


class NexusConfig(BaseSettings):
    """Top-level application configuration.

    Loads from (in priority order, highest first):
      1. Environment variables (NEXUS_ prefix)
      2. YAML file at NEXUS_CONFIG_FILE / nexus.yaml / _config_file kwarg
      3. Coded defaults
    """

    _active_yaml_path: ClassVar[str | None] = None

    model: ModelConfig = ModelConfig()
    memory: MemoryConfig = MemoryConfig()
    safety: SafetyConfig = SafetyConfig()
    dapr: DaprConfig = DaprConfig()
    observability: ObservabilityConfig = ObservabilityConfig()

    model_config = {"env_prefix": "NEXUS_", "env_nested_delimiter": "__", "extra": "ignore"}

    def __init__(self, _config_file: str | None = None, **data: Any) -> None:
        if _config_file is not None:
            NexusConfig._active_yaml_path = _config_file
        try:
            super().__init__(**data)
        finally:
            NexusConfig._active_yaml_path = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_path = cls._active_yaml_path or _yaml_path()
        if yaml_path and Path(yaml_path).exists():
            return (
                init_settings,
                env_settings,
                YamlConfigSettingsSource(settings_cls, yaml_file=Path(yaml_path)),
            )
        return (init_settings, env_settings)
