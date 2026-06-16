"""Tests for grampus.core.config — GrampusConfig and sub-configs."""

from pathlib import Path

import pytest

from grampus.core.config import (
    DaprConfig,
    GrampusConfig,
    MemoryConfig,
    ModelConfig,
    ObservabilityConfig,
    SafetyConfig,
)

# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_defaults(self) -> None:
        cfg = ModelConfig()
        assert cfg.default_model is not None
        assert cfg.temperature >= 0.0
        assert cfg.max_tokens > 0

    def test_api_key_is_secret(self) -> None:
        cfg = ModelConfig(anthropic_api_key="sk-secret")
        assert "sk-secret" not in repr(cfg)

    def test_set_values(self) -> None:
        cfg = ModelConfig(default_model="gpt-4o", temperature=0.5, max_tokens=2048)
        assert cfg.default_model == "gpt-4o"
        assert cfg.temperature == 0.5
        assert cfg.max_tokens == 2048


# ---------------------------------------------------------------------------
# MemoryConfig
# ---------------------------------------------------------------------------


class TestMemoryConfig:
    def test_defaults(self) -> None:
        cfg = MemoryConfig()
        assert cfg.working_memory_token_limit > 0
        assert cfg.episodic_top_k > 0
        assert 0.0 <= cfg.decay_rate <= 1.0

    def test_custom_values(self) -> None:
        cfg = MemoryConfig(working_memory_token_limit=8000, episodic_top_k=10, decay_rate=0.05)
        assert cfg.working_memory_token_limit == 8000
        assert cfg.episodic_top_k == 10


# ---------------------------------------------------------------------------
# SafetyConfig
# ---------------------------------------------------------------------------


class TestSafetyConfig:
    def test_defaults(self) -> None:
        cfg = SafetyConfig()
        assert cfg.injection_detection_level in ("strict", "balanced", "permissive")
        assert isinstance(cfg.pii_detection_enabled, bool)

    def test_custom_values(self) -> None:
        cfg = SafetyConfig(injection_detection_level="strict", pii_detection_enabled=True)
        assert cfg.injection_detection_level == "strict"


# ---------------------------------------------------------------------------
# DaprConfig
# ---------------------------------------------------------------------------


class TestDaprConfig:
    def test_defaults(self) -> None:
        cfg = DaprConfig()
        assert cfg.host is not None
        assert cfg.port > 0
        assert cfg.grpc_port > 0
        assert cfg.state_store_name is not None
        assert cfg.pubsub_name is not None
        assert cfg.cache_store_name is not None

    def test_base_url(self) -> None:
        cfg = DaprConfig(host="localhost", port=3500)
        assert "localhost" in cfg.base_url
        assert "3500" in cfg.base_url


# ---------------------------------------------------------------------------
# ObservabilityConfig
# ---------------------------------------------------------------------------


class TestObservabilityConfig:
    def test_defaults(self) -> None:
        cfg = ObservabilityConfig()
        assert isinstance(cfg.otel_enabled, bool)
        assert cfg.log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

    def test_custom(self) -> None:
        cfg = ObservabilityConfig(otel_enabled=True, log_level="DEBUG")
        assert cfg.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# GrampusConfig (top-level)
# ---------------------------------------------------------------------------


class TestGrampusConfig:
    def test_creates_with_defaults(self) -> None:
        cfg = GrampusConfig()
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.memory, MemoryConfig)
        assert isinstance(cfg.safety, SafetyConfig)
        assert isinstance(cfg.dapr, DaprConfig)
        assert isinstance(cfg.observability, ObservabilityConfig)

    def test_env_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRAMPUS_DAPR__PORT", "4500")
        cfg = GrampusConfig()
        assert cfg.dapr.port == 4500

    def test_model_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRAMPUS_MODEL__DEFAULT_MODEL", "claude-3-opus")
        cfg = GrampusConfig()
        assert cfg.model.default_model == "claude-3-opus"

    def test_api_key_masked_in_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRAMPUS_MODEL__ANTHROPIC_API_KEY", "sk-super-secret")
        cfg = GrampusConfig()
        assert "sk-super-secret" not in repr(cfg)

    def test_load_from_yaml_file(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "grampus.yaml"
        yaml_file.write_text("model:\n  default_model: gpt-4o-mini\n")
        cfg = GrampusConfig(_config_file=str(yaml_file))  # type: ignore[call-arg]
        assert cfg.model.default_model == "gpt-4o-mini"

    def test_yaml_path_overridable_via_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        yaml_file = tmp_path / "custom.yaml"
        yaml_file.write_text("memory:\n  episodic_top_k: 99\n")
        monkeypatch.setenv("GRAMPUS_CONFIG_FILE", str(yaml_file))
        cfg = GrampusConfig()
        assert cfg.memory.episodic_top_k == 99

    def test_env_overrides_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        yaml_file = tmp_path / "grampus.yaml"
        yaml_file.write_text("dapr:\n  port: 3501\n")
        monkeypatch.setenv("GRAMPUS_CONFIG_FILE", str(yaml_file))
        monkeypatch.setenv("GRAMPUS_DAPR__PORT", "9999")
        cfg = GrampusConfig()
        assert cfg.dapr.port == 9999

    def test_missing_yaml_file_uses_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRAMPUS_CONFIG_FILE", "/nonexistent/grampus.yaml")
        cfg = GrampusConfig()
        assert isinstance(cfg.model, ModelConfig)
