"""Recursive and fixed-window text chunker for document processing (H44)."""

from __future__ import annotations

import math
import uuid
from pathlib import Path

from grampus.tools.library.document_types import (
    ChunkStrategy,
    DocumentChunk,
    DocumentMetadata,
)

_TOKEN_FACTOR = 1.3


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: word count × 1.3, rounded up."""
    return math.ceil(len(text.split()) * _TOKEN_FACTOR)


def _context_header(section_path: list[str], metadata: DocumentMetadata) -> str:
    if section_path:
        return " > ".join(section_path)
    if metadata.title:
        return metadata.title
    return Path(metadata.source).name


def _make_chunk(
    content: str,
    index: int,
    total: int,
    metadata: DocumentMetadata,
    section_path: list[str],
    page: int | None,
    sheet: str | None,
) -> DocumentChunk:
    return DocumentChunk(
        id=str(uuid.uuid4()),
        content=content,
        context_header=_context_header(section_path, metadata),
        chunk_index=index,
        total_chunks=total,
        token_estimate=_estimate_tokens(content),
        page=page,
        sheet=sheet,
        section_path=list(section_path),
        metadata=metadata,
    )


def _merge_parts(parts: list[str], sep: str, max_tokens: int) -> list[str]:
    """Accumulate split parts into chunks that each fit within max_tokens."""
    chunks: list[str] = []
    current: list[str] = []
    for part in parts:
        candidate = sep.join(current + [part])
        if _estimate_tokens(candidate) > max_tokens and current:
            chunks.extend(_split_recursive(sep.join(current), max_tokens))
            current = [part]
        else:
            current.append(part)
    if current:
        chunks.extend(_split_recursive(sep.join(current), max_tokens))
    return chunks


def _split_recursive(text: str, max_tokens: int) -> list[str]:
    """Split text at paragraph → line → sentence → word boundaries until ≤ max_tokens."""
    if _estimate_tokens(text) <= max_tokens:
        return [text]
    for sep in ["\n\n", "\n", ". ", " "]:
        parts = text.split(sep)
        if len(parts) > 1:
            return _merge_parts(parts, sep, max_tokens)
    words = text.split()
    return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]


class DocumentChunker:
    """Splits text into DocumentChunk records using recursive or fixed strategies.

    Args:
        strategy: RECURSIVE (default) or FIXED.
        chunk_size: Target token count per chunk.
        overlap_ratio: Fraction of chunk_size that overlaps between adjacent FIXED chunks.
    """

    def __init__(
        self,
        strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        chunk_size: int = 512,
        overlap_ratio: float = 0.10,
    ) -> None:
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio

    def chunk(
        self,
        text: str,
        metadata: DocumentMetadata,
        section_path: list[str] | None = None,
        page: int | None = None,
        sheet: str | None = None,
    ) -> list[DocumentChunk]:
        """Split *text* into a list of DocumentChunk records.

        Args:
            text: Raw text to split.
            metadata: Source document metadata attached to every chunk.
            section_path: Heading breadcrumb above this text block.
            page: Page number (PDF only).
            sheet: Sheet name (Excel only).
        """
        effective_path = section_path or []
        if not text.strip():
            return []
        raw = (
            self._fixed_chunks(text)
            if self.strategy == ChunkStrategy.FIXED
            else _split_recursive(text, self.chunk_size)
        )
        raw = [c for c in raw if c.strip()]
        total = len(raw)
        return [
            _make_chunk(c, i, total, metadata, effective_path, page, sheet)
            for i, c in enumerate(raw)
        ]

    def _fixed_chunks(self, text: str) -> list[str]:
        """Produce fixed word-count windows with configurable overlap."""
        words = text.split()
        overlap = int(self.chunk_size * self.overlap_ratio)
        step = max(1, self.chunk_size - overlap)
        result: list[str] = []
        i = 0
        while i < len(words):
            result.append(" ".join(words[i : i + self.chunk_size]))
            i += step
        return result
