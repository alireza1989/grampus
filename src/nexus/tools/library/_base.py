"""Shared helpers for all library tools."""

from __future__ import annotations

from typing import Any


def ok(output: Any, **extra: Any) -> dict[str, Any]:
    """Wrap a successful result."""
    result: dict[str, Any] = {"ok": True, "result": output}
    result.update(extra)
    return result


def err(message: str, *, code: str = "TOOL_ERROR") -> dict[str, Any]:
    """Wrap an error — never raises, returns error payload."""
    return {"ok": False, "error": message, "code": code}
