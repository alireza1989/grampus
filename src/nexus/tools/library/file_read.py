"""Safe file reader tool with path sandboxing."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from nexus.tools.library._base import err

_DEFAULT_MAX_BYTES = 102_400  # 100 KB


def _read_file(path: Path, max_bytes: int) -> tuple[str, int, bool]:
    raw = path.read_bytes()
    truncated = len(raw) > max_bytes
    chunk = raw[:max_bytes]
    content = chunk.decode("utf-8")
    return content, len(chunk), truncated


async def file_read(
    path: str,
    allowed_base_dir: str = ".",
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    """Read a file within *allowed_base_dir*.

    Args:
        path: File path to read.
        allowed_base_dir: Restrict reads to this directory tree.
        max_bytes: Maximum bytes to read; file is truncated beyond this.

    Returns:
        ``{"ok": True, "path": str, "content": str, "bytes_read": int}`` or error dict.
    """
    target = Path(path).resolve()
    base = Path(allowed_base_dir).resolve()

    try:
        target.relative_to(base)
    except ValueError:
        return err(
            f"Path traversal blocked: {path!r} is outside {allowed_base_dir!r}",
            code="PATH_TRAVERSAL_BLOCKED",
        )

    if not target.exists():
        return err(f"File not found: {path!r}", code="FILE_NOT_FOUND")
    if not target.is_file():
        return err(f"Not a file: {path!r}", code="NOT_A_FILE")

    try:
        content, bytes_read, truncated = await asyncio.to_thread(_read_file, target, max_bytes)
    except PermissionError:
        return err(f"Permission denied: {path!r}", code="PERMISSION_DENIED")
    except UnicodeDecodeError:
        return err(f"Cannot read binary file as text: {path!r}", code="BINARY_FILE")

    result: dict[str, Any] = {
        "ok": True,
        "path": str(target),
        "content": content,
        "bytes_read": bytes_read,
    }
    if truncated:
        result["truncated"] = True
    return result
