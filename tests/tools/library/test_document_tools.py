"""Tests for document processing tools (H44) — PDF, DOCX, Excel."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from grampus.tools.library.document_chunker import DocumentChunker
from grampus.tools.library.document_tools import read_docx_tool, read_excel_tool, read_pdf_tool
from grampus.tools.library.document_types import ChunkStrategy, DocumentMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta(source: str = "test.pdf", fmt: str = "pdf") -> DocumentMetadata:
    return DocumentMetadata(source=source, format=fmt, parser="test")


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


# ---------------------------------------------------------------------------
# DocumentChunker — unit tests
# ---------------------------------------------------------------------------


class TestDocumentChunker:
    def test_recursive_splits_large_paragraph_block(self) -> None:
        # 10 paragraphs × 30 words = 300 words; chunk_size=50 → several chunks
        text = "\n\n".join([_words(30)] * 10)
        chunker = DocumentChunker(strategy=ChunkStrategy.RECURSIVE, chunk_size=50)
        chunks = chunker.chunk(text, _meta())
        assert len(chunks) > 1

    def test_recursive_chunk_token_estimate_within_bounds(self) -> None:
        text = " ".join(["word"] * 200)
        chunker = DocumentChunker(strategy=ChunkStrategy.RECURSIVE, chunk_size=30)
        chunks = chunker.chunk(text, _meta())
        # Each chunk should have a token_estimate ≤ ~2× chunk_size (generous slack)
        for c in chunks:
            assert c.token_estimate <= 80

    def test_overlap_carries_tail_into_next_chunk(self) -> None:
        words = [f"w{i}" for i in range(100)]
        text = " ".join(words)
        chunker = DocumentChunker(strategy=ChunkStrategy.FIXED, chunk_size=20, overlap_ratio=0.10)
        chunks = chunker.chunk(text, _meta())
        assert len(chunks) > 1
        overlap = int(20 * 0.10)  # 2
        tail = chunks[0].content.split()[-overlap:]
        head = chunks[1].content.split()[:overlap]
        assert tail == head

    def test_fixed_strategy_produces_correct_chunk_count(self) -> None:
        text = _words(100)
        chunker = DocumentChunker(strategy=ChunkStrategy.FIXED, chunk_size=20, overlap_ratio=0.0)
        chunks = chunker.chunk(text, _meta())
        assert len(chunks) == 5  # 100 words ÷ 20 per chunk

    def test_context_header_uses_section_path(self) -> None:
        text = "some content for testing"
        chunker = DocumentChunker(chunk_size=512)
        chunks = chunker.chunk(text, _meta(), section_path=["Chapter 1", "Introduction"])
        assert len(chunks) > 0
        assert chunks[0].context_header == "Chapter 1 > Introduction"

    def test_context_header_falls_back_to_title(self) -> None:
        text = "some content for testing"
        chunker = DocumentChunker(chunk_size=512)
        meta = DocumentMetadata(source="doc.pdf", format="pdf", parser="test", title="My Doc")
        chunks = chunker.chunk(text, meta, section_path=[])
        assert len(chunks) > 0
        assert chunks[0].context_header == "My Doc"

    def test_context_header_falls_back_to_source_basename(self) -> None:
        text = "some content for testing"
        chunker = DocumentChunker(chunk_size=512)
        meta = DocumentMetadata(source="/data/report.pdf", format="pdf", parser="test", title=None)
        chunks = chunker.chunk(text, meta, section_path=[])
        assert len(chunks) > 0
        assert chunks[0].context_header == "report.pdf"

    def test_empty_text_returns_no_chunks(self) -> None:
        chunks = DocumentChunker().chunk("", _meta())
        assert chunks == []

    def test_whitespace_only_returns_no_chunks(self) -> None:
        chunks = DocumentChunker().chunk("   \n\n  \t  ", _meta())
        assert chunks == []

    def test_chunks_have_valid_uuid4_ids(self) -> None:
        chunks = DocumentChunker().chunk("Sample text for id test here", _meta())
        for c in chunks:
            parsed = uuid.UUID(c.id, version=4)
            assert str(parsed) == c.id

    def test_chunk_indices_are_sequential(self) -> None:
        text = "\n\n".join([_words(30)] * 5)
        chunks = DocumentChunker(chunk_size=50).chunk(text, _meta())
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)


# ---------------------------------------------------------------------------
# Tool: read_pdf_tool — unit tests
# ---------------------------------------------------------------------------


class TestReadPdfTool:
    async def test_missing_dep_returns_err(self) -> None:
        mock_stat = MagicMock()
        mock_stat.return_value.st_size = 1024
        with (
            patch.dict(sys.modules, {"fitz": None, "pypdf": None}),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "stat", mock_stat),
        ):
            result = await read_pdf_tool(path="/fake/doc.pdf")
        assert result["ok"] is False
        assert result["code"] == "MISSING_DEPENDENCY"

    async def test_file_not_found_returns_err(self, tmp_path: Path) -> None:
        result = await read_pdf_tool(path=str(tmp_path / "no_such.pdf"))
        assert result["ok"] is False
        assert result["code"] == "FILE_NOT_FOUND"

    async def test_wrong_extension_returns_err(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.docx"
        p.write_text("hello")
        result = await read_pdf_tool(path=str(p))
        assert result["ok"] is False
        assert result["code"] == "UNSUPPORTED_FORMAT"

    async def test_file_too_large_returns_err(self, tmp_path: Path) -> None:
        p = tmp_path / "big.pdf"
        p.touch()
        mock_stat = MagicMock()
        mock_stat.return_value.st_size = 60 * 1024 * 1024  # 60 MB
        with patch.object(Path, "stat", mock_stat):
            result = await read_pdf_tool(path=str(p))
        assert result["ok"] is False
        assert result["code"] == "FILE_TOO_LARGE"


# ---------------------------------------------------------------------------
# Tool: read_docx_tool — unit tests
# ---------------------------------------------------------------------------


class TestReadDocxTool:
    async def test_missing_dep_returns_err(self) -> None:
        mock_stat = MagicMock()
        mock_stat.return_value.st_size = 1024
        with (
            patch.dict(sys.modules, {"docx": None}),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "stat", mock_stat),
        ):
            result = await read_docx_tool(path="/fake/doc.docx")
        assert result["ok"] is False
        assert result["code"] == "MISSING_DEPENDENCY"

    async def test_file_not_found_returns_err(self, tmp_path: Path) -> None:
        result = await read_docx_tool(path=str(tmp_path / "no_such.docx"))
        assert result["ok"] is False
        assert result["code"] == "FILE_NOT_FOUND"

    async def test_wrong_extension_returns_err(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"fake")
        result = await read_docx_tool(path=str(p))
        assert result["ok"] is False
        assert result["code"] == "UNSUPPORTED_FORMAT"


# ---------------------------------------------------------------------------
# Tool: read_excel_tool — unit tests
# ---------------------------------------------------------------------------


class TestReadExcelTool:
    async def test_missing_dep_returns_err(self) -> None:
        mock_stat = MagicMock()
        mock_stat.return_value.st_size = 1024
        with (
            patch.dict(sys.modules, {"openpyxl": None}),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "stat", mock_stat),
        ):
            result = await read_excel_tool(path="/fake/sheet.xlsx")
        assert result["ok"] is False
        assert result["code"] == "MISSING_DEPENDENCY"

    async def test_file_not_found_returns_err(self, tmp_path: Path) -> None:
        result = await read_excel_tool(path=str(tmp_path / "no_such.xlsx"))
        assert result["ok"] is False
        assert result["code"] == "FILE_NOT_FOUND"

    async def test_wrong_extension_returns_err(self, tmp_path: Path) -> None:
        p = tmp_path / "sheet.csv"
        p.write_text("a,b,c")
        result = await read_excel_tool(path=str(p))
        assert result["ok"] is False
        assert result["code"] == "UNSUPPORTED_FORMAT"


# ---------------------------------------------------------------------------
# Integration tests — require grampus[documents] installed
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDocumentIntegration:
    @pytest.fixture()
    def pdf_fixture(self, tmp_path: Path) -> Path:
        fitz = pytest.importorskip("fitz", reason="requires pymupdf (grampus[documents])")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Chapter One\n\nThis is the body of chapter one.")
        pdf_path = tmp_path / "test.pdf"
        doc.save(str(pdf_path))
        doc.close()
        return pdf_path

    @pytest.fixture()
    def docx_fixture(self, tmp_path: Path) -> Path:
        docx = pytest.importorskip("docx", reason="requires python-docx (grampus[documents])")
        doc = docx.Document()
        doc.add_heading("Chapter 1", level=1)
        doc.add_paragraph("This is the first paragraph in chapter one.")
        doc.add_heading("Section 1.1", level=2)
        doc.add_paragraph("This is a nested section with more content here.")
        docx_path = tmp_path / "test.docx"
        doc.save(str(docx_path))
        return docx_path

    @pytest.fixture()
    def xlsx_fixture(self, tmp_path: Path) -> Path:
        openpyxl = pytest.importorskip("openpyxl", reason="requires openpyxl (grampus[documents])")
        wb = openpyxl.Workbook()
        ws1 = wb.active
        ws1.title = "Sheet1"
        ws1.append(["Name", "Value"])
        ws1.append(["Alice", 100])
        ws1.append(["Bob", 200])
        ws2 = wb.create_sheet("Sheet2")
        ws2.append(["Date", "Amount"])
        ws2.append(["2024-01-01", 500])
        xlsx_path = tmp_path / "test.xlsx"
        wb.save(str(xlsx_path))
        return xlsx_path

    async def test_read_pdf_integration(self, pdf_fixture: Path) -> None:
        result = await read_pdf_tool(path=str(pdf_fixture))
        assert result["ok"] is True
        assert result["total_chunks"] > 0
        chunks = result["chunks"]
        assert all(c["page"] is not None for c in chunks)

    async def test_read_docx_integration(self, docx_fixture: Path) -> None:
        result = await read_docx_tool(path=str(docx_fixture))
        assert result["ok"] is True
        assert result["total_chunks"] > 0
        chunks = result["chunks"]
        section_paths = [c["section_path"] for c in chunks]
        assert any(len(sp) > 0 for sp in section_paths)

    async def test_read_excel_integration(self, xlsx_fixture: Path) -> None:
        result = await read_excel_tool(path=str(xlsx_fixture))
        assert result["ok"] is True
        assert result["total_chunks"] > 0
        chunks = result["chunks"]
        sheets = {c["sheet"] for c in chunks}
        assert "Sheet1" in sheets
        assert "Sheet2" in sheets

    async def test_parsed_document_structure(self, docx_fixture: Path) -> None:
        result = await read_docx_tool(path=str(docx_fixture))
        assert result["ok"] is True
        for chunk in result["chunks"]:
            assert uuid.UUID(chunk["id"], version=4)
            assert isinstance(chunk["content"], str)
            assert isinstance(chunk["context_header"], str)
            assert len(chunk["context_header"]) > 0
            assert isinstance(chunk["token_estimate"], int)
            assert chunk["token_estimate"] > 0
