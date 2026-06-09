"""Pydantic models for document processing (H44)."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ChunkStrategy(StrEnum):
    """Text chunking strategy."""

    RECURSIVE = "recursive"
    FIXED = "fixed"


class DocumentMetadata(BaseModel):
    """Metadata extracted from a source document."""

    source: str
    format: str
    total_pages: int | None = None
    total_sheets: int | None = None
    title: str | None = None
    author: str | None = None
    created_at: datetime | None = None
    parser: str


class DocumentChunk(BaseModel):
    """A single text chunk extracted from a document, ready for RAG ingestion."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    context_header: str
    chunk_index: int
    total_chunks: int
    token_estimate: int
    page: int | None = None
    sheet: str | None = None
    section_path: list[str] = Field(default_factory=list)
    metadata: DocumentMetadata


class ParsedDocument(BaseModel):
    """A fully parsed document with all chunks and metadata."""

    chunks: list[DocumentChunk]
    metadata: DocumentMetadata
    raw_text: str
