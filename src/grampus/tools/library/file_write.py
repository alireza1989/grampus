"""Safe file writer tool with path sandboxing."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from grampus.tools.library._base import err


def _write_file(target: Path, content: str, create_dirs: bool, overwrite: bool) -> int:
    if not overwrite and target.exists():
        raise FileExistsError(f"File already exists: {target}")
    if create_dirs:
        target.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    target.write_bytes(encoded)
    return len(encoded)


async def file_write(
    path: str,
    content: str,
    allowed_base_dir: str = ".",
    create_dirs: bool = True,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Write *content* to *path* within *allowed_base_dir*.

    Args:
        path: Destination file path.
        content: Text content to write.
        allowed_base_dir: Restrict writes to this directory tree.
        create_dirs: Create parent directories if they don't exist.
        overwrite: Whether to overwrite an existing file.

    Returns:
        ``{"ok": True, "path": str, "bytes_written": int}`` or error dict.
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

    try:
        bytes_written = await asyncio.to_thread(
            _write_file, target, content, create_dirs, overwrite
        )
    except FileExistsError as exc:
        return err(str(exc), code="FILE_EXISTS")
    except PermissionError:
        return err(f"Permission denied: {path!r}", code="PERMISSION_DENIED")

    return {"ok": True, "path": str(target), "bytes_written": bytes_written}
