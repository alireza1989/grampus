"""Registered tool functions for PDF, DOCX, and Excel document parsing (H44)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from grampus.core.errors import ToolError
from grampus.tools.library._base import err
from grampus.tools.library.document_chunker import DocumentChunker
from grampus.tools.library.document_reader import read_docx, read_excel, read_pdf
from grampus.tools.library.document_types import ChunkStrategy

_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB


def _check_file(
    path_str: str, expected_suffix: str
) -> tuple[Path, dict[str, Any]] | tuple[None, dict[str, Any]]:
    """Validate path, extension, and size. Returns (Path, {}) or (None, err_dict)."""
    target = Path(path_str)
    if not target.exists():
        return None, err(f"File not found: {path_str!r}", code="FILE_NOT_FOUND")
    if target.suffix.lower() != expected_suffix:
        return None, err(
            f"Expected {expected_suffix} file, got {target.suffix!r}",
            code="UNSUPPORTED_FORMAT",
        )
    if target.stat().st_size > _MAX_FILE_BYTES:
        mb = _MAX_FILE_BYTES // (1024 * 1024)
        return None, err(f"File exceeds {mb} MB size limit: {path_str!r}", code="FILE_TOO_LARGE")
    return target, {}


def _parse_strategy(
    chunk_strategy: str,
) -> tuple[ChunkStrategy, dict[str, Any]] | tuple[None, dict[str, Any]]:
    try:
        return ChunkStrategy(chunk_strategy), {}
    except ValueError:
        return None, err(
            f"Invalid chunk_strategy {chunk_strategy!r}; must be 'recursive' or 'fixed'",
            code="INVALID_STRATEGY",
        )


def _format_result(doc: Any) -> dict[str, Any]:
    return {
        "ok": True,
        "chunks": [c.model_dump() for c in doc.chunks],
        "metadata": doc.metadata.model_dump(),
        "total_chunks": len(doc.chunks),
    }


async def read_pdf_tool(
    path: str,
    chunk_size: int = 512,
    chunk_strategy: str = "recursive",
) -> dict[str, Any]:
    """Parse a PDF into structured text chunks ready for RAG or memory ingestion.

    Args:
        path: Path to the .pdf file.
        chunk_size: Target token count per chunk (default 512).
        chunk_strategy: 'recursive' (default) or 'fixed'.

    Returns:
        ``{"ok": True, "chunks": [...], "metadata": {...}, "total_chunks": N}`` or error dict.
    """
    target, failure = _check_file(path, ".pdf")
    if target is None:
        return failure
    strategy, strat_err = _parse_strategy(chunk_strategy)
    if strategy is None:
        return strat_err
    chunker = DocumentChunker(strategy=strategy, chunk_size=chunk_size)
    try:
        doc = await read_pdf(target, chunker)
    except ToolError as exc:
        return err(str(exc), code=exc.code)
    except Exception as exc:
        return err(f"Failed to parse PDF: {exc}", code="PARSE_ERROR")
    return _format_result(doc)


async def read_docx_tool(
    path: str,
    chunk_size: int = 512,
    chunk_strategy: str = "recursive",
) -> dict[str, Any]:
    """Parse a Word (.docx) file into structured text chunks.

    Args:
        path: Path to the .docx file.
        chunk_size: Target token count per chunk (default 512).
        chunk_strategy: 'recursive' (default) or 'fixed'.

    Returns:
        ``{"ok": True, "chunks": [...], "metadata": {...}, "total_chunks": N}`` or error dict.
    """
    target, failure = _check_file(path, ".docx")
    if target is None:
        return failure
    strategy, strat_err = _parse_strategy(chunk_strategy)
    if strategy is None:
        return strat_err
    chunker = DocumentChunker(strategy=strategy, chunk_size=chunk_size)
    try:
        doc = await read_docx(target, chunker)
    except ToolError as exc:
        return err(str(exc), code=exc.code)
    except Exception as exc:
        return err(f"Failed to parse DOCX: {exc}", code="PARSE_ERROR")
    return _format_result(doc)


async def read_excel_tool(
    path: str,
    chunk_size: int = 512,
    chunk_strategy: str = "recursive",
) -> dict[str, Any]:
    """Parse an Excel (.xlsx) file into structured text chunks.

    Args:
        path: Path to the .xlsx file.
        chunk_size: Target token count per chunk (default 512).
        chunk_strategy: 'recursive' (default) or 'fixed'.

    Returns:
        ``{"ok": True, "chunks": [...], "metadata": {...}, "total_chunks": N}`` or error dict.
    """
    target, failure = _check_file(path, ".xlsx")
    if target is None:
        return failure
    strategy, strat_err = _parse_strategy(chunk_strategy)
    if strategy is None:
        return strat_err
    chunker = DocumentChunker(strategy=strategy, chunk_size=chunk_size)
    try:
        doc = await read_excel(target, chunker)
    except ToolError as exc:
        return err(str(exc), code=exc.code)
    except Exception as exc:
        return err(f"Failed to parse Excel: {exc}", code="PARSE_ERROR")
    return _format_result(doc)
