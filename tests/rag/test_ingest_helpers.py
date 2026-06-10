"""Tests for demos.rag.ingest helpers — no live DB or embedding API needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from demos.rag.ingest import _collect_files, _NullCacheStore


def test_collect_files_single_file(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("# Hello")
    result = _collect_files(str(f))
    assert result == [f]


def test_collect_files_directory(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("md")
    (tmp_path / "b.pdf").write_text("pdf")
    (tmp_path / "c.txt").write_text("txt")
    (tmp_path / "d.py").write_text("py")
    result = _collect_files(str(tmp_path))
    suffixes = {p.suffix for p in result}
    assert ".py" not in suffixes
    assert len(result) == 3


def test_collect_files_recursive(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "top.md").write_text("top")
    (sub / "nested.txt").write_text("nested")
    result = _collect_files(str(tmp_path))
    names = {p.name for p in result}
    assert "top.md" in names
    assert "nested.txt" in names


@pytest.mark.asyncio
async def test_null_cache_store_get_returns_none() -> None:
    store = _NullCacheStore()
    result = await store.get("entity", "key", object)
    assert result == (None, None)


@pytest.mark.asyncio
async def test_null_cache_store_save_is_noop() -> None:
    store = _NullCacheStore()
    await store.save("entity", "key", b"data")
