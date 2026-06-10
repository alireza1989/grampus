"""Tests for demos.rag.config — RAGConfig construction and loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from demos.rag.config import RAGConfig


def test_rag_config_defaults() -> None:
    cfg = RAGConfig()
    assert cfg.namespace == "default"
    assert cfg.top_k == 10
    assert cfg.chunk_size == 512


def test_rag_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_DB_URL", "postgresql://test/db")
    monkeypatch.setenv("RAG_NAMESPACE", "myns")
    cfg = RAGConfig.from_env()
    assert cfg.db_url == "postgresql://test/db"
    assert cfg.namespace == "myns"


def test_rag_config_from_json_file(tmp_path: Path) -> None:
    config_file = tmp_path / "rag.json"
    config_file.write_text(json.dumps({"namespace": "test", "top_k": 20}))
    cfg = RAGConfig.from_file(str(config_file))
    assert cfg.namespace == "test"
    assert cfg.top_k == 20
