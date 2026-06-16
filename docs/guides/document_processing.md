# Document Processing

Grampus ships three built-in tools that let agents ingest PDF, Word (.docx), and Excel (.xlsx) files
as structured text chunks, ready to store in `EpisodicMemory` or feed directly into a RAG pipeline.

---

## Installation

Document processing requires optional extras:

```bash
pip install 'grampus-ai[documents]'
```

This installs: `pymupdf4llm` (PDF primary), `pypdf` (PDF fallback), `python-docx` (Word), `openpyxl` (Excel).

The core Grampus install **without** `[documents]` is unaffected. Agents that call a document tool
without the extras get a clear error: `code="MISSING_DEPENDENCY"`.

---

## Quick Start

```python
from grampus.tools.library import LIBRARY_REGISTRY

registry = LIBRARY_REGISTRY

# Call via the registry
result = await registry.get_or_raise("read_pdf").fn(
    path="/data/paper.pdf",
    chunk_size=512,
)

if result["ok"]:
    for chunk in result["chunks"]:
        print(chunk["context_header"], "—", chunk["content"][:80])
```

Or import the functions directly:

```python
from grampus.tools.library.document_tools import read_pdf_tool, read_docx_tool, read_excel_tool

result = await read_pdf_tool(path="/data/paper.pdf")
```

---

## Chunking Strategies

| Strategy | How it splits | When to use |
|---|---|---|
| `recursive` (default) | Paragraph → sentence → word boundaries | Prose, reports, research papers |
| `fixed` | Fixed word-count windows with 10% overlap | Dense tables, logs, structured data |

**Recursive** is the 2026 benchmark winner (69% E2E accuracy across 50 papers). It never breaks
mid-sentence when avoidable, making embeddings more semantically coherent.

**Fixed** with overlap is better when you need deterministic, overlapping windows — for example,
when matching short query phrases against a code listing or a financial statement.

```python
# Fixed strategy with 15% overlap
result = await read_pdf_tool(
    path="/data/report.pdf",
    chunk_size=256,
    chunk_strategy="fixed",
)
```

---

## Contextual Retrieval

Every chunk carries a `context_header` field — the heading breadcrumb above that text:

```
"Annual Report 2024 > Financial Results > Q3 Revenue"
```

This field is **not** part of `content`. The embedding layer should concatenate them for retrieval
so each chunk is self-contained when returned in isolation:

```python
for chunk in result["chunks"]:
    embedding_text = chunk["context_header"] + "\n\n" + chunk["content"]
    vector = await embed(embedding_text)
    await memory_manager.remember(chunk["content"], embedding=vector)
```

The `context_header` falls back to `metadata.title` if no headings are present, and to the
file's basename if neither is set.

---

## Excel Specifics

- Each sheet is processed independently; `chunk["sheet"]` identifies the source sheet.
- The first row is treated as the header.
- Sheets with more than **1000 rows** are truncated; a note is appended in the chunk content.
- Each sheet is rendered as a Markdown table before chunking.

```python
result = await read_excel_tool(path="/data/financials.xlsx")
for chunk in result["chunks"]:
    print(f"Sheet: {chunk['sheet']}, tokens: {chunk['token_estimate']}")
```

---

## Integrating with Memory

```python
from grampus.tools.library.document_tools import read_pdf_tool

result = await read_pdf_tool(path="/data/paper.pdf")
if not result["ok"]:
    raise RuntimeError(result["error"])

for chunk in result["chunks"]:
    await memory_manager.remember(
        content=chunk["content"],
        metadata={
            "source": chunk["metadata"]["source"],
            "page": chunk["page"],
            "section": chunk["context_header"],
        },
    )
```

---

## File Size Limits

The default limit is **50 MB** per file. Files larger than this return:

```json
{"ok": false, "code": "FILE_TOO_LARGE", "error": "File exceeds 50 MB size limit: ..."}
```

---

## Error Codes

| Code | Cause |
|---|---|
| `MISSING_DEPENDENCY` | Required library not installed — run `pip install 'grampus-ai[documents]'` |
| `FILE_NOT_FOUND` | Path does not exist |
| `UNSUPPORTED_FORMAT` | File extension does not match the tool (e.g. passing `.txt` to `read_pdf`) |
| `FILE_TOO_LARGE` | File exceeds the 50 MB size limit |
| `PARSE_ERROR` | Library raised an unexpected error while reading the file |
| `INVALID_STRATEGY` | `chunk_strategy` is not `'recursive'` or `'fixed'` |
