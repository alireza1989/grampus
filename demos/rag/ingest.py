"""RAG ingestion CLI: process documents → embed → upsert into pgvector."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import click

from demos.rag.config import EmbeddingProviderName, RAGConfig
from demos.rag.rag_store import ChunkRecord, RAGStore, make_document_id
from nexus.core.logging import get_logger
from nexus.memory.embedding_providers import (
    CohereEmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from nexus.memory.embeddings import EmbeddingService
from nexus.tools.library.document_chunker import DocumentChunker
from nexus.tools.library.document_types import ChunkStrategy, DocumentMetadata

_log = get_logger(__name__)

_SUPPORTED_EXTS = {".pdf", ".docx", ".xlsx", ".md", ".txt"}


class _NullCacheStore:
    """Duck-typed no-op cache — avoids Dapr/Redis dependency during batch ingest."""

    async def get(self, entity: str, key: str, type_: Any) -> tuple[None, None]:
        return None, None

    async def save(self, entity: str, key: str, data: bytes) -> None:
        pass


def _build_embedding_service(config: RAGConfig) -> EmbeddingService:
    if config.embedding_provider == EmbeddingProviderName.openai:
        try:
            import openai
        except ImportError as exc:
            raise SystemExit("OpenAI provider requires: pip install 'nexus-ai[openai]'") from exc
        client = openai.AsyncOpenAI(api_key=config.openai_api_key or None)
        provider = OpenAIEmbeddingProvider(client, config.embedding_model)
    elif config.embedding_provider == EmbeddingProviderName.cohere:
        try:
            import cohere
        except ImportError as exc:
            raise SystemExit("Cohere provider requires: pip install 'nexus-ai[cohere]'") from exc
        client = cohere.AsyncClientV2(api_key=config.cohere_api_key)
        provider = CohereEmbeddingProvider(client, config.embedding_model)
    else:
        provider = OllamaEmbeddingProvider(config.embedding_model, config.ollama_base_url)
    return EmbeddingService(provider, _NullCacheStore())


def _collect_files(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p] if p.suffix in _SUPPORTED_EXTS else []
    return sorted(f for f in p.rglob("*") if f.is_file() and f.suffix in _SUPPORTED_EXTS)


async def _process_file(
    file_path: Path,
    config: RAGConfig,
    embedding_service: EmbeddingService,
) -> list[ChunkRecord]:
    doc_chunks: list[dict[str, Any]] = []

    if file_path.suffix in {".md", ".txt"}:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        metadata = DocumentMetadata(
            source=str(file_path),
            format=file_path.suffix.lstrip(".") or "txt",
            parser="text",
        )
        chunker = DocumentChunker(
            strategy=ChunkStrategy(config.chunk_strategy),
            chunk_size=config.chunk_size,
        )
        chunks = chunker.chunk(text, metadata)
        doc_chunks = [c.model_dump() for c in chunks]

    elif file_path.suffix == ".pdf":
        from nexus.tools.library.document_tools import read_pdf_tool

        result = await read_pdf_tool(
            path=str(file_path),
            chunk_size=config.chunk_size,
            chunk_strategy=config.chunk_strategy,
        )
        if not result.get("ok"):
            _log.warning("pdf_parse_failed", path=str(file_path), error=result.get("error"))
            return []
        doc_chunks = result["chunks"]

    elif file_path.suffix == ".docx":
        from nexus.tools.library.document_tools import read_docx_tool

        result = await read_docx_tool(
            path=str(file_path),
            chunk_size=config.chunk_size,
            chunk_strategy=config.chunk_strategy,
        )
        if not result.get("ok"):
            _log.warning("docx_parse_failed", path=str(file_path), error=result.get("error"))
            return []
        doc_chunks = result["chunks"]

    elif file_path.suffix == ".xlsx":
        from nexus.tools.library.document_tools import read_excel_tool

        result = await read_excel_tool(
            path=str(file_path),
            chunk_size=config.chunk_size,
            chunk_strategy=config.chunk_strategy,
        )
        if not result.get("ok"):
            _log.warning("xlsx_parse_failed", path=str(file_path), error=result.get("error"))
            return []
        doc_chunks = result["chunks"]

    if not doc_chunks:
        return []

    texts = [c["content"] for c in doc_chunks]
    embeddings = await embedding_service.embed_batch(texts, input_type="search_document")

    document_id = make_document_id(str(file_path))
    return [
        ChunkRecord(
            id=str(uuid.uuid4()),
            namespace=config.namespace,
            document_id=document_id,
            source_path=str(file_path),
            chunk_index=i,
            content=chunk["content"],
            context_header=chunk.get("context_header", ""),
            metadata={"file": file_path.name, "suffix": file_path.suffix},
            embedding=embeddings[i],
        )
        for i, chunk in enumerate(doc_chunks)
    ]


@click.command()
@click.option(
    "--path", required=True, type=click.Path(exists=True), help="File or directory to ingest."
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(),
    help="Path to RAGConfig JSON/YAML file.",
)
@click.option("--namespace", default=None, help="Override config namespace.")
@click.option("--db-url", default=None, help="Override config db_url.")
@click.option(
    "--enrich/--no-enrich", default=False, help="LLM context enrichment per chunk (slower)."
)
@click.option(
    "--force/--no-force", default=False, help="Re-ingest documents even if already present."
)
def ingest(
    path: str,
    config_path: str | None,
    namespace: str | None,
    db_url: str | None,
    enrich: bool,
    force: bool,
) -> None:
    """Ingest documents into the RAG vector store."""
    asyncio.run(_ingest_main(path, config_path, namespace, db_url, enrich, force))


async def _ingest_main(
    path: str,
    config_path: str | None,
    namespace: str | None,
    db_url: str | None,
    enrich: bool,
    force: bool,
) -> None:
    config = RAGConfig.from_file(config_path) if config_path else RAGConfig.from_env()
    updates: dict[str, Any] = {}
    if namespace:
        updates["namespace"] = namespace
    if db_url:
        updates["db_url"] = db_url
    if enrich:
        updates["enrich_chunks"] = True
    if updates:
        config = config.model_copy(update=updates)

    click.echo(f"Connecting to {config.db_url} ...")
    embedding_service = _build_embedding_service(config)
    store = await RAGStore.create(config.db_url, dimensions=embedding_service.dimensions)

    files = _collect_files(path)
    click.echo(f"Found {len(files)} file(s) to process.")

    total_chunks = 0
    for i, file_path in enumerate(files):
        click.echo(f" [{i + 1}/{len(files)}] {file_path.name} ...", nl=False)
        chunks = await _process_file(file_path, config, embedding_service)
        if force and chunks:
            await store.delete_document(config.namespace, make_document_id(str(file_path)))
        n = await store.upsert_chunks(chunks)
        total_chunks += n
        click.echo(f" {n} chunks")

    stats = await store.get_stats(config.namespace)
    click.echo("\n--- Ingestion complete ---")
    click.echo(f"  Chunks upserted this run : {total_chunks}")
    click.echo(f"  Total chunks in store    : {stats['chunk_count']}")
    click.echo(f"  Total documents in store : {stats['document_count']}")
    await store.close()


if __name__ == "__main__":
    ingest()
