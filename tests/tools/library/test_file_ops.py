"""Tests for nexus.tools.library.file_read and file_write."""

from __future__ import annotations

from pathlib import Path

from nexus.tools.library.file_read import file_read
from nexus.tools.library.file_write import file_write


class TestFileRead:
    async def test_file_read_returns_content(self, tmp_path: Path) -> None:
        p = tmp_path / "hello.txt"
        p.write_text("hello world")
        result = await file_read(path=str(p), allowed_base_dir=str(tmp_path))
        assert result["ok"] is True
        assert result["content"] == "hello world"

    async def test_file_read_includes_path_and_bytes_read(self, tmp_path: Path) -> None:
        p = tmp_path / "data.txt"
        p.write_text("abc")
        result = await file_read(path=str(p), allowed_base_dir=str(tmp_path))
        assert result["ok"] is True
        assert result["bytes_read"] == 3

    async def test_file_read_path_traversal_blocked(self, tmp_path: Path) -> None:
        jail = tmp_path / "jail"
        jail.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("secret")
        result = await file_read(path=str(secret), allowed_base_dir=str(jail))
        assert result["ok"] is False
        assert (
            "PATH_TRAVERSAL" in result.get("code", "")
            or "traversal" in result.get("error", "").lower()
        )

    async def test_file_read_missing_file_returns_err(self, tmp_path: Path) -> None:
        result = await file_read(path=str(tmp_path / "no_such.txt"), allowed_base_dir=str(tmp_path))
        assert result["ok"] is False

    async def test_file_read_truncates_large_file(self, tmp_path: Path) -> None:
        p = tmp_path / "big.txt"
        p.write_bytes(b"A" * 200_000)
        result = await file_read(path=str(p), allowed_base_dir=str(tmp_path), max_bytes=1000)
        assert result["ok"] is True
        assert result["truncated"] is True
        assert len(result["content"]) <= 1000

    async def test_file_read_no_truncation_flag_when_small(self, tmp_path: Path) -> None:
        p = tmp_path / "small.txt"
        p.write_text("hi")
        result = await file_read(path=str(p), allowed_base_dir=str(tmp_path))
        assert result.get("truncated") is not True


class TestFileWrite:
    async def test_file_write_creates_file(self, tmp_path: Path) -> None:
        p = tmp_path / "out.txt"
        result = await file_write(path=str(p), content="hello", allowed_base_dir=str(tmp_path))
        assert result["ok"] is True
        assert p.read_text() == "hello"

    async def test_file_write_returns_bytes_written(self, tmp_path: Path) -> None:
        p = tmp_path / "out.txt"
        result = await file_write(path=str(p), content="hello", allowed_base_dir=str(tmp_path))
        assert result["ok"] is True
        assert result["bytes_written"] == 5

    async def test_file_write_path_traversal_blocked(self, tmp_path: Path) -> None:
        jail = tmp_path / "jail"
        jail.mkdir()
        outside = tmp_path / "outside.txt"
        result = await file_write(path=str(outside), content="bad", allowed_base_dir=str(jail))
        assert result["ok"] is False

    async def test_file_write_overwrite_false_blocked(self, tmp_path: Path) -> None:
        p = tmp_path / "existing.txt"
        p.write_text("original")
        result = await file_write(
            path=str(p),
            content="new",
            allowed_base_dir=str(tmp_path),
            overwrite=False,
        )
        assert result["ok"] is False
        assert p.read_text() == "original"

    async def test_file_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "a" / "b" / "c.txt"
        result = await file_write(
            path=str(p),
            content="deep",
            allowed_base_dir=str(tmp_path),
            create_dirs=True,
        )
        assert result["ok"] is True
        assert p.read_text() == "deep"

    async def test_file_write_overwrite_true_replaces(self, tmp_path: Path) -> None:
        p = tmp_path / "f.txt"
        p.write_text("old")
        result = await file_write(
            path=str(p), content="new", allowed_base_dir=str(tmp_path), overwrite=True
        )
        assert result["ok"] is True
        assert p.read_text() == "new"
