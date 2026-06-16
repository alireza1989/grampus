"""Tests for grampus.tools.library.sql_query."""

from __future__ import annotations

from pathlib import Path

import pytest

from grampus.tools.library.sql_query import sql_query


def _sqlite_url(tmp_path: Path) -> str:
    return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"


async def _setup_db(url: str) -> None:
    try:
        import sqlalchemy as sa
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(url)
        async with engine.begin() as conn:
            await conn.execute(sa.text("CREATE TABLE IF NOT EXISTS users (id INTEGER, name TEXT)"))
            await conn.execute(sa.text("DELETE FROM users"))
            await conn.execute(
                sa.text("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')")
            )
        await engine.dispose()
    except ImportError:
        pytest.skip("sqlalchemy/aiosqlite not installed")


class TestSqlQuery:
    async def test_select_returns_rows(self, tmp_path: Path) -> None:
        url = _sqlite_url(tmp_path)
        await _setup_db(url)
        result = await sql_query(query="SELECT * FROM users", connection_string=url)
        assert result["ok"] is True
        assert result["row_count"] == 3
        assert len(result["rows"]) == 3

    async def test_column_names_in_result(self, tmp_path: Path) -> None:
        url = _sqlite_url(tmp_path)
        await _setup_db(url)
        result = await sql_query(query="SELECT * FROM users", connection_string=url)
        assert result["ok"] is True
        assert "id" in result["columns"]
        assert "name" in result["columns"]

    async def test_rows_are_dicts(self, tmp_path: Path) -> None:
        url = _sqlite_url(tmp_path)
        await _setup_db(url)
        result = await sql_query(query="SELECT * FROM users", connection_string=url)
        assert result["ok"] is True
        for row in result["rows"]:
            assert isinstance(row, dict)

    async def test_non_select_rejected(self, tmp_path: Path) -> None:
        url = _sqlite_url(tmp_path)
        result = await sql_query(query="DROP TABLE users", connection_string=url)
        assert result["ok"] is False
        assert "SELECT" in result.get("error", "")

    async def test_insert_rejected(self, tmp_path: Path) -> None:
        url = _sqlite_url(tmp_path)
        result = await sql_query(
            query="INSERT INTO users VALUES (99, 'Evil')", connection_string=url
        )
        assert result["ok"] is False

    async def test_max_rows_truncated(self, tmp_path: Path) -> None:
        url = _sqlite_url(tmp_path)
        await _setup_db(url)
        result = await sql_query(
            query="SELECT * FROM users",
            connection_string=url,
            max_rows=2,
        )
        assert result["ok"] is True
        assert len(result["rows"]) == 2
        assert result.get("truncated") is True

    async def test_invalid_connection_string_returns_err(self) -> None:
        result = await sql_query(query="SELECT 1", connection_string="invalid://not:a:real:db")
        assert result["ok"] is False

    async def test_missing_optional_deps_returns_err(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name in ("sqlalchemy", "sqlalchemy.ext.asyncio"):
                raise ImportError("no module")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = await sql_query(query="SELECT 1", connection_string=_sqlite_url(tmp_path))
        assert result["ok"] is False
        assert "pip install" in result.get("error", "")
