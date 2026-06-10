"""Tests for demos.rag.rag_tool — pure unit tests with mocks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from demos.rag.config import RAGConfig
from demos.rag.rag_store import RetrievedChunk
from demos.rag.rag_tool import _format_context, make_retrieve_tool


def _chunk(
    idx: int, content: str = "body text", source: str = "/doc.md", header: str = ""
) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"id-{idx}",
        content=content,
        context_header=header,
        source_path=source,
        chunk_index=idx,
        metadata={},
        rrf_score=0.9 - idx * 0.1,
    )


def test_format_context_numbered() -> None:
    chunks = [_chunk(0, "alpha text"), _chunk(1, "beta text")]
    result = _format_context(chunks)
    assert "[1]" in result
    assert "[2]" in result
    assert "alpha text" in result
    assert "beta text" in result


def test_format_context_empty() -> None:
    result = _format_context([])
    assert "No relevant context" in result


def test_format_context_includes_source() -> None:
    chunks = [_chunk(0, "content", source="/docs/guide.md")]
    result = _format_context(chunks)
    assert "/docs/guide.md" in result


def test_make_retrieve_fn_returns_callable() -> None:
    mock_store = MagicMock()
    mock_svc = MagicMock()
    config = RAGConfig()
    registry, fn = make_retrieve_tool(mock_store, mock_svc, config)
    import asyncio

    assert asyncio.iscoroutinefunction(fn)


@pytest.mark.asyncio
async def test_retrieve_tool_calls_embed_with_search_query() -> None:
    mock_store = MagicMock()
    mock_store.retrieve = AsyncMock(return_value=[])
    mock_svc = MagicMock()
    mock_svc.embed = AsyncMock(return_value=[0.1] * 1536)
    config = RAGConfig()
    _, fn = make_retrieve_tool(mock_store, mock_svc, config)
    await fn("what is Nexus?")
    mock_svc.embed.assert_called_once_with("what is Nexus?", input_type="search_query")


@pytest.mark.asyncio
async def test_retrieve_tool_returns_ok_dict() -> None:
    mock_store = MagicMock()
    mock_store.retrieve = AsyncMock(return_value=[_chunk(0, "some result")])
    mock_svc = MagicMock()
    mock_svc.embed = AsyncMock(return_value=[0.1] * 1536)
    config = RAGConfig()
    _, fn = make_retrieve_tool(mock_store, mock_svc, config)
    result = await fn("test query")
    assert result["ok"] is True
    assert "context" in result
