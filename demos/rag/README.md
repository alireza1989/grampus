# Nexus RAG Pipeline Template

Complete retrieval-augmented generation using the full Nexus stack: hybrid vector+BM25
search, pgvector HNSW indexing, multi-provider embeddings, and citation-grounded answers.

## Architecture

```
Documents (PDF, DOCX, MD, TXT)
│
▼
DocumentChunker (recursive, 512 tokens)
│  adds heading breadcrumbs as context_header
▼
EmbeddingService (OpenAI / Cohere / Ollama)
│  embed_batch(chunks, input_type="search_document")
▼
PostgreSQL + pgvector
│  HNSW index + tsvector for BM25
│
▼  ← Query time
Hybrid Search (BM25 + vector + RRF fusion)
│  lost-in-the-middle reordering
▼
AgentRunner with retrieve_context tool
│  system prompt enforces citation [1][2] format
▼
Answer with source citations
```

## Prerequisites

- PostgreSQL 16+ with pgvector extension installed
- Python 3.12+
- An OpenAI API key (or Cohere / Ollama for alternative providers)
- An Anthropic API key for answer generation

```bash
# Install Nexus with RAG dependencies
pip install 'nexus-ai[rag,openai]'

# For PDF/DOCX support add:
pip install 'nexus-ai[rag,openai,documents]'

# Or from source:
uv sync --extra rag --extra openai
```

## 5-Minute Quickstart

### 1. Start PostgreSQL with pgvector

```bash
# Using Docker:
docker run -d --name nexus-pg \
  -e POSTGRES_DB=nexus \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 2. Set environment variables

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export RAG_DB_URL=postgresql://postgres:postgres@localhost/nexus
```

### 3. Ingest the sample documents

```bash
python demos/rag/ingest.py --path demos/rag/sample_docs/
```

Output:

```
Connecting to postgresql://postgres:postgres@localhost/nexus ...
Found 1 file(s) to process.
  [1/1] nexus_overview.md ... 12 chunks
Ingested 12 chunks total.
Namespace 'default': 12 chunks, 1 documents
```

### 4. Ask a question

```bash
python demos/rag/rag_agent.py --query "What are the four memory layers in Nexus?"
```

Or interactive mode:

```bash
python demos/rag/rag_agent.py --interactive
```

### 5. Evaluate quality

```bash
python demos/rag/eval/rag_eval.py
```

## Configuration

Create `rag_config.json` to override any setting:

```json
{
  "namespace": "my-project",
  "embedding_model": "text-embedding-3-large",
  "chunk_size": 256,
  "top_k": 15,
  "max_context_chunks": 8,
  "model_id": "claude-opus-4-7"
}
```

Pass it to any command: `--config rag_config.json`

### Full configuration reference

| Field | Default | Description |
|---|---|---|
| `db_url` | `postgresql://localhost/nexus` | PostgreSQL connection string |
| `namespace` | `default` | Tenant/project scope — all queries filter by this |
| `embedding_provider` | `openai` | `openai`, `cohere`, or `ollama` |
| `embedding_model` | `text-embedding-3-small` | Model name for the chosen provider |
| `chunk_size` | `512` | Target tokens per chunk |
| `chunk_strategy` | `recursive` | `recursive` (recommended) or `fixed` |
| `enrich_chunks` | `false` | Add LLM-generated context per chunk at ingest time |
| `top_k` | `10` | Candidates from hybrid search before context trimming |
| `rrf_k` | `60` | RRF constant (research-validated default, rarely change) |
| `max_context_chunks` | `6` | Chunks sent to LLM — balances quality vs. token cost |
| `model_id` | `claude-haiku-4-5-20251001` | Generation model |
| `temperature` | `0.1` | Low for factual RAG |

## Ingestion Options

```bash
# Ingest a single PDF
python demos/rag/ingest.py --path report.pdf --namespace finance-q4

# Ingest entire directory recursively
python demos/rag/ingest.py --path ./knowledge-base/ --config prod.json

# Re-ingest (replace existing chunks for these documents)
python demos/rag/ingest.py --path ./updated-docs/ --force

# Enable LLM context enrichment (slower but higher retrieval quality)
python demos/rag/ingest.py --path ./docs/ --enrich
```

Supported formats: `.pdf` (requires `[documents]`), `.docx` (requires `[documents]`),
`.xlsx` (requires `[documents]`), `.md`, `.txt`

## Retrieval Architecture Details

### Hybrid BM25 + Vector Search

Pure vector search fails on keyword-heavy queries ("what does section 4.2 say about X").
Pure BM25 fails on paraphrase queries. This template uses both, fused with Reciprocal
Rank Fusion:

```
rrf_score(doc) = 1/(60 + vector_rank) + 1/(60 + bm25_rank)
```

BM25 is powered by PostgreSQL's built-in `tsvector` and `ts_rank` — no extra infrastructure.

### HNSW Indexing

The template uses HNSW (Hierarchical Navigable Small World) rather than IVFFlat:

- No training pass required — works immediately on an empty table
- Handles incremental inserts without reindexing
- Better recall at comparable query latency for datasets under 1M vectors

### Lost-in-the-Middle Mitigation

LLMs recall context at the beginning and end of the context window better than the middle.
Retrieved chunks are reordered before being sent: highest-scoring chunks at position `[0]`
and `[-1]`, lower-scoring chunks fill the middle.

## Customizing Embedding Providers

```bash
# Use Ollama with a local model (no API key required)
RAG_EMBEDDING_PROVIDER=ollama RAG_EMBEDDING_MODEL=nomic-embed-text \
    python demos/rag/ingest.py --path ./docs/

# Use Cohere Embed v3 (best for multilingual)
RAG_EMBEDDING_PROVIDER=cohere RAG_EMBEDDING_MODEL=embed-v4.0 \
COHERE_API_KEY=... python demos/rag/ingest.py --path ./docs/
```

**Important:** After switching embedding providers, drop and recreate the table (different
providers produce different vector dimensions and incompatible embeddings):

```sql
DROP TABLE rag_chunks;
```

Then re-run ingest.

## Multi-Namespace Usage

Namespaces scope all data to a tenant or project. Different namespaces never interfere:

```bash
# Ingest into separate namespaces
python demos/rag/ingest.py --path ./team-a-docs/ --namespace team-a
python demos/rag/ingest.py --path ./team-b-docs/ --namespace team-b

# Query a specific namespace
python demos/rag/rag_agent.py --namespace team-a --query "..."
```

## Evaluation

The evaluation script scores answers on two dimensions using LLM-as-judge (Claude):

- **Faithfulness (1–5):** Are all claims in the answer supported by retrieved context?
  Score 5 means zero hallucination.
- **Relevancy (1–5):** Does the answer address the user's question?

```bash
# Run with default sample Q&A set
python demos/rag/eval/rag_eval.py

# Use your own Q&A file
python demos/rag/eval/rag_eval.py --qa-file my_questions.json --output results.json
```

Q&A file format (`my_questions.json`):

```json
[
  {
    "id": "q1",
    "question": "What is the return policy?",
    "expected_topics": ["30 days", "receipt required"]
  }
]
```

## Production Checklist

Before using in production:

- [ ] Run PostgreSQL with SSL enabled and a dedicated database user with least-privilege access
- [ ] Set `namespace` per tenant — never share a namespace across customers
- [ ] Use `text-embedding-3-large` (3072d) instead of small for higher accuracy
- [ ] Enable `enrich_chunks=true` at ingest for 10–15% better retrieval quality
- [ ] Increase `top_k` to 20+ for large corpora, keep `max_context_chunks` at 6–8
- [ ] Set up a scheduled re-ingestion job with `--force` for documents that change
- [ ] Monitor `avg_faithfulness` via `rag_eval.py` as a quality regression test in CI
- [ ] Back up the `rag_chunks` table — it represents your processed knowledge base
- [ ] Wire up the `EmbeddingService` with a Dapr Redis cache store for embedding caching
  in high-volume deployments

## Extending This Template

Add a custom tool alongside RAG:

```python
# In rag_agent.py, create a second registry and merge:
rag_registry, _ = make_retrieve_tool(store, embedding_service, config)
rag_registry.register(my_custom_tool, name="my_tool", ...)
```

Use Nexus memory for conversation history:

```python
# Wire up MemoryManager and pass to AgentRunner for cross-session memory
runner = AgentRunner(model_client, tool_executor, memory_manager=memory_manager)
```

Plug into a Nexus Crew:

```python
from nexus.orchestration.crew import Crew
# The rag_agent can be one member of a multi-agent crew
```
