# Embedding Providers

Grampus ships three embedding providers — OpenAI, Cohere, and Ollama — that can be mixed and matched
per memory type via `EmbeddingRouter`. All providers share the same `EmbeddingService` interface,
so existing code needs no changes to use a different backend.

> **Dimension mismatch warning.** Switching providers without updating your pgvector column
> dimensions silently drops all writes — no error is raised, vectors are just quietly discarded.
> Always read `.dimensions` from your provider at setup time and validate it against your schema
> before the first write (see [Dimension validation](#dimension-validation)).

---

## OpenAI

Best for production workloads where quality matters most. Requires `pip install grampus-ai[openai]`.

```python
from openai import AsyncOpenAI
from grampus.memory.embedding_providers import OpenAIEmbeddingProvider
from grampus.memory.embeddings import EmbeddingService

client = AsyncOpenAI(api_key="sk-...")
provider = OpenAIEmbeddingProvider(client=client, model="text-embedding-3-small")
service = EmbeddingService(provider=provider, cache_store=dapr_cache)
```

| Model | Dimensions | Relative cost |
|---|---|---|
| `text-embedding-3-small` (default) | 1536 | Low |
| `text-embedding-3-large` | 3072 | Medium |
| `text-embedding-ada-002` | 1536 | Low (legacy) |

---

## Cohere

Best for multilingual content and when domain-tuned quality outweighs cost. Requires
`pip install grampus-ai[cohere]`.

```python
import cohere
from grampus.memory.embedding_providers import CohereEmbeddingProvider
from grampus.memory.embeddings import EmbeddingService

client = cohere.AsyncClientV2(api_key="co-...")
provider = CohereEmbeddingProvider(client=client, model="embed-english-v3.0")
service = EmbeddingService(provider=provider, cache_store=dapr_cache)
```

| Model | Dimensions | Notes |
|---|---|---|
| `embed-english-v3.0` (default) | 1024 | Best English quality |
| `embed-multilingual-v3.0` | 1024 | 100+ languages |
| `embed-english-light-v3.0` | 384 | Faster, lower cost |
| `embed-multilingual-light-v3.0` | 384 | Multilingual, fast |

**`input_type` matters for Cohere v3+.** Cohere distinguishes between content being stored
(`"search_document"`) and queries used for retrieval (`"search_query"`). Omitting `input_type`
silently degrades quality. Pass it explicitly when you know the context:

```python
# Storing a memory record — use search_document (the default)
vector = await service.embed(text, input_type="search_document")

# Retrieving — use search_query
query_vector = await service.embed(query, input_type="search_query")
```

---

## Ollama

Best for local/offline deployments and cost-sensitive working memory. Uses httpx (already a
core dep) — no extra install required. Run `ollama serve` before use.

```python
from grampus.memory.embedding_providers import OllamaEmbeddingProvider
from grampus.memory.embeddings import EmbeddingService

provider = OllamaEmbeddingProvider(model="nomic-embed-text", base_url="http://localhost:11434")
service = EmbeddingService(provider=provider, cache_store=dapr_cache)
```

| Model | Dimensions | Notes |
|---|---|---|
| `nomic-embed-text` (default) | 768 | Good quality, fast |
| `mxbai-embed-large` | 1024 | Higher quality |
| `all-minilm` | 384 | Very fast, small |
| `qwen3-embedding` | 2048 | Multilingual |

If Ollama is not running, calls raise `EmbeddingError` with a hint: `"Run: ollama serve"`.

---

## Per-memory-type routing

Use `EmbeddingRouter` to direct different memory types to the most cost-effective provider.
`EmbeddingRouter` is duck-type compatible with `EmbeddingService` for `.embed()`,
`.embed_batch()`, and `.dimensions`, so it can replace a service anywhere in your code.

```python
from grampus.memory.embedding_providers import (
    EmbeddingRouter,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from grampus.memory.embeddings import EmbeddingService

router = EmbeddingRouter({
    # Default for any unregistered purpose
    "default":    EmbeddingService(OpenAIEmbeddingProvider(client, "text-embedding-3-small"), cache),
    # High-quality large model for long-term semantic facts
    "semantic":   EmbeddingService(OpenAIEmbeddingProvider(client, "text-embedding-3-large"), cache),
    # Free local model for transient working memory
    "working":    EmbeddingService(OllamaEmbeddingProvider("nomic-embed-text"), cache),
    # Same local model for short-lived episodic records
    "episodic":   EmbeddingService(OllamaEmbeddingProvider("nomic-embed-text"), cache),
})

# Callers that only use .embed() / .embed_batch() / .dimensions need no changes:
vector = await router.embed(text, purpose="semantic")
vectors = await router.embed_batch(texts, purpose="working", input_type="search_document")

# Pass the router wherever an EmbeddingService is accepted:
memory_manager = MemoryManager(embedding_service=router, ...)
```

Unmapped purposes silently fall back to `"default"`.

---

## Dimension validation

Before writing to pgvector, validate that your provider's dimensions match your column width:

```python
PGVECTOR_DIMENSIONS = 1536  # what your schema was created with

provider = OpenAIEmbeddingProvider(client, model="text-embedding-3-large")  # 3072 dims
service = EmbeddingService(provider=provider, cache_store=cache)

if service.dimensions != PGVECTOR_DIMENSIONS:
    raise RuntimeError(
        f"Provider produces {service.dimensions}-dim vectors but pgvector column "
        f"expects {PGVECTOR_DIMENSIONS}. Update your schema or change the model."
    )
```

Without this check, a provider switch silently drops all writes — vectors arrive with the wrong
dimension and pgvector rejects them without raising a Python-level error.

---

## Migration from the old API

```python
# Old (still works via backward-compat shim — will be removed in a future release)
service = EmbeddingService(openai_client=client, cache_store=cache, model="text-embedding-3-small")

# New
from grampus.memory.embedding_providers import OpenAIEmbeddingProvider
provider = OpenAIEmbeddingProvider(client=client, model="text-embedding-3-small")
service = EmbeddingService(provider=provider, cache_store=cache)
```

The backward-compat `openai_client=` keyword is accepted in the current release but will be
removed in v0.2. Migrate to `provider=` before then.
