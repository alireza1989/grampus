"""Read-only SQL query tool — SELECT only, sqlalchemy optional dep."""

from __future__ import annotations

import re
from typing import Any

from grampus.tools.library._base import err

_SELECT_RE = re.compile(r"^\s*(/\*.*?\*/\s*|--[^\n]*\n\s*)*select\b", re.IGNORECASE | re.DOTALL)


async def sql_query(
    query: str,
    connection_string: str,
    max_rows: int = 100,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Execute a read-only SELECT query and return results.

    Args:
        query: SQL SELECT statement.
        connection_string: SQLAlchemy async connection string (e.g. "sqlite+aiosqlite:///db.sqlite3").
        max_rows: Maximum number of rows to return.
        timeout_seconds: Query execution timeout.

    Returns:
        ``{"ok": True, "rows": list[dict], "columns": list[str], "row_count": int}`` or error dict.
    """
    if not _SELECT_RE.match(query):
        return err("Only SELECT queries are permitted", code="NON_SELECT_QUERY")

    try:
        import sqlalchemy as sa
        from sqlalchemy.ext.asyncio import create_async_engine
    except ImportError:
        return err(
            "sql_query requires: pip install grampus-ai[sql]",
            code="MISSING_DEPS",
        )

    try:
        engine = create_async_engine(connection_string, future=True)
    except Exception as exc:
        return err(f"Failed to create engine: {exc}", code="CONNECTION_ERROR")

    try:
        async with engine.connect() as conn:
            result = await conn.execute(sa.text(query))
            columns = list(result.keys())
            all_rows = result.fetchmany(max_rows + 1)
            truncated = len(all_rows) > max_rows
            rows = [dict(zip(columns, row, strict=True)) for row in all_rows[:max_rows]]
    except Exception as exc:
        return err(f"Query failed: {exc}", code="QUERY_ERROR")
    finally:
        await engine.dispose()

    out: dict[str, Any] = {
        "ok": True,
        "rows": rows,
        "columns": columns,
        "row_count": len(rows),
    }
    if truncated:
        out["truncated"] = True
    return out
