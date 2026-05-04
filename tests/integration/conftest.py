"""Shared fixtures for integration tests.

Infrastructure strategy:
- PostgreSQL and Redis are started via testcontainers.
- Dapr sidecar is NOT started by testcontainers (complex to wire up).
  Instead, integration tests that need real Dapr use a DaprStateStore
  backed by the testcontainer PostgreSQL directly via the Dapr HTTP API
  if a sidecar is available, OR skip with pytest.mark.skipif when
  DAPR_HTTP_PORT is not set in the environment.

This gives us two tiers:
  1. "container" tests — require Docker, start PG + Redis, run automatically
  2. "dapr" tests — require a live Dapr sidecar, skipped in CI unless
     DAPR_HTTP_PORT env var is set (set by the integration CI job)

Mark dapr-dependent tests with @pytest.mark.dapr
Mark container-only tests with @pytest.mark.integration
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest
import pytest_asyncio

from nexus.core.errors import ConcurrencyError
from nexus.core.models.base import ModelResponse
from nexus.core.types import (
    AgentDefinition,
    Message,
    TokenUsage,
    ToolCall,
    ToolParameter,
)
from nexus.tools.registry import ToolRegistry

DAPR_HTTP_PORT = os.environ.get("DAPR_HTTP_PORT", "")
skip_if_no_dapr = pytest.mark.skipif(
    not DAPR_HTTP_PORT,
    reason="No Dapr sidecar available. Set DAPR_HTTP_PORT to run dapr tests.",
)


# ---------------------------------------------------------------------------
# FakeStateStore — in-memory stand-in conforming to DaprStateStore duck-type
# ---------------------------------------------------------------------------


class FakeStateStore:
    """In-memory state store conforming to DaprStateStore interface.

    Supports Pydantic model values and raw bytes. ETags are monotonically
    incrementing integers represented as strings.
    """

    def __init__(self) -> None:
        self._data: dict[str, tuple[Any, str]] = {}
        self._etag_counter = 0

    async def save(
        self,
        entity: str,
        key: str,
        value: Any,
        *,
        etag: str | None = None,
        **_: Any,
    ) -> None:
        full_key = f"{entity}:{key}"
        if etag is not None:
            existing = self._data.get(full_key)
            if existing and existing[1] != etag:
                raise ConcurrencyError(
                    f"ETag mismatch for {full_key}",
                    code="CONCURRENCY_ERROR",
                )
        self._etag_counter += 1
        self._data[full_key] = (value, str(self._etag_counter))

    async def get(self, entity: str, key: str, cls: Any) -> tuple[Any, str]:
        full_key = f"{entity}:{key}"
        result = self._data.get(full_key)
        if result is None:
            return None, ""
        value, etag = result
        return value, etag

    async def delete(self, entity: str, key: str, **_: Any) -> None:
        self._data.pop(f"{entity}:{key}", None)

    async def bulk_get(self, entity: str, keys: list[str], cls: Any) -> list[tuple[Any, str]]:
        return [await self.get(entity, k, cls) for k in keys]

    async def save_bulk(self, entity: str, items: list[tuple[str, Any]]) -> None:
        for eid, model in items:
            await self.save(entity, eid, model)

    async def execute_transaction(
        self,
        entity: str,
        operations: list[tuple[str, str, str, Any]],
    ) -> None:
        for op_type, _etype, entity_id, model in operations:
            if op_type == "upsert" and model is not None:
                await self.save(entity, entity_id, model)
            elif op_type == "delete":
                await self.delete(entity, entity_id)


# ---------------------------------------------------------------------------
# FakeEmbeddingService — returns deterministic random-looking embeddings
# ---------------------------------------------------------------------------


class FakeEmbeddingService:
    """Embedding service that returns deterministic unit vectors for testing."""

    def __init__(self, dims: int = 8) -> None:
        self._dims = dims
        self._cache: dict[str, list[float]] = {}

    async def embed(self, text: str) -> list[float]:
        if text not in self._cache:
            h = hash(text) % (10**8)
            raw = [float((h >> i) & 0xFF) / 255.0 + 0.01 for i in range(self._dims)]
            norm = sum(x**2 for x in raw) ** 0.5 or 1.0
            self._cache[text] = [x / norm for x in raw]
        return self._cache[text]


# ---------------------------------------------------------------------------
# MockModelClient — configurable async LLM client
# ---------------------------------------------------------------------------


class MockModelClient:
    """Synchronous-friendly mock that returns preconfigured responses in sequence."""

    def __init__(
        self,
        responses: list[ModelResponse] | None = None,
        default_text: str = "Done.",
    ) -> None:
        self._responses = list(responses or [])
        self._default_text = default_text
        self._call_count = 0
        self.calls: list[list[Message]] = []

    def add_response(
        self,
        text: str = "Done.",
        tool_calls: list[ToolCall] | None = None,
        cost_usd: float = 0.0001,
    ) -> MockModelClient:
        self._responses.append(
            ModelResponse(
                content=text,
                tool_calls=tool_calls or [],
                token_usage=TokenUsage(
                    input_tokens=10,
                    output_tokens=10,
                    total_tokens=20,
                    cost_usd=cost_usd,
                    model="mock-model",
                ),
                model="mock-model",
                stop_reason="end_turn",
            )
        )
        return self

    async def complete(
        self,
        *,
        messages: list[Message],
        model: str = "mock-model",
        tools: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **_: Any,
    ) -> ModelResponse:
        self.calls.append(list(messages))
        if self._responses:
            resp = self._responses[self._call_count % len(self._responses)]
        else:
            resp = ModelResponse(
                content=self._default_text,
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=10,
                    output_tokens=10,
                    total_tokens=20,
                    cost_usd=0.0001,
                    model=model,
                ),
                model=model,
                stop_reason="end_turn",
            )
        self._call_count += 1
        return resp

    def stream(self, **_: Any) -> Any:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def fake_state_store() -> FakeStateStore:
    """Fresh in-memory state store for each test."""
    return FakeStateStore()


@pytest.fixture()
def fake_embedding_service() -> FakeEmbeddingService:
    return FakeEmbeddingService()


@pytest.fixture()
def mock_model_client() -> MockModelClient:
    """Default model client that always returns 'Done.'"""
    return MockModelClient(default_text="Done.")


@pytest.fixture()
def basic_agent_def() -> AgentDefinition:
    return AgentDefinition(
        name="test-agent",
        model="mock-model",
        system_prompt="You are a helpful assistant.",
        tools=[],
        max_iterations=5,
        temperature=0.0,
        memory_enabled=False,
        cost_budget_usd=None,
    )


# ---------------------------------------------------------------------------
# Tool registry / executor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_tool_registry() -> ToolRegistry:
    """ToolRegistry pre-loaded with: echo, add, fail."""
    registry = ToolRegistry()

    @registry.tool(
        name="echo",
        description="Echo the input text back",
        parameters=[
            ToolParameter(name="text", type="string", description="Text to echo", required=True)
        ],
    )
    async def echo(text: str) -> str:
        return text

    @registry.tool(
        name="add",
        description="Add two numbers",
        parameters=[
            ToolParameter(name="a", type="number", description="First number", required=True),
            ToolParameter(name="b", type="number", description="Second number", required=True),
        ],
    )
    def add(a: float, b: float) -> float:
        return a + b

    return registry


@pytest_asyncio.fixture()
async def tool_executor(mock_tool_registry: ToolRegistry) -> Any:
    from nexus.tools.executor import ToolExecutor

    return ToolExecutor(mock_tool_registry, timeout_seconds=5.0, max_retries=1)


# ---------------------------------------------------------------------------
# Memory fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def episodic_memory(
    fake_state_store: FakeStateStore,
    fake_embedding_service: FakeEmbeddingService,
) -> Any:
    from nexus.memory.episodic import EpisodicMemory

    return EpisodicMemory(
        fake_state_store,
        fake_embedding_service,
        agent_id="test-agent",
    )


@pytest_asyncio.fixture()
async def semantic_memory(fake_state_store: FakeStateStore) -> Any:
    from nexus.memory.semantic import SemanticMemory

    return SemanticMemory(fake_state_store, agent_id="test-agent")


@pytest_asyncio.fixture()
async def memory_manager(
    fake_state_store: FakeStateStore,
    fake_embedding_service: FakeEmbeddingService,
    mock_model_client: MockModelClient,
) -> Any:
    """MemoryManager wired with all memory types and no security middleware."""
    from nexus.memory.consolidation import ConsolidationPipeline
    from nexus.memory.episodic import EpisodicMemory
    from nexus.memory.manager import MemoryManager
    from nexus.memory.procedural import ProceduralMemory
    from nexus.memory.retriever import EpisodicRetriever
    from nexus.memory.semantic import SemanticMemory
    from nexus.memory.semantic_retriever import SemanticRetriever
    from nexus.memory.summarizer import Summarizer
    from nexus.memory.token_counter import TokenCounter
    from nexus.memory.working import WorkingMemory

    store = fake_state_store
    emb = fake_embedding_service

    episodic = EpisodicMemory(store, emb, agent_id="test-agent")
    semantic = SemanticMemory(store, agent_id="test-agent")
    procedural = ProceduralMemory(store, agent_id="test-agent")

    from nexus.memory.summarizer import SummarizationStrategy

    token_counter = TokenCounter("mock-model")
    summarizer = Summarizer(
        mock_model_client,
        "mock-model",
        SummarizationStrategy.TRUNCATE,
        token_counter,
    )
    working = WorkingMemory(
        store,
        token_counter,
        summarizer,
        agent_id="test-agent",
        session_id="test-session",
    )

    ep_retriever = EpisodicRetriever(episodic, emb)
    sem_retriever = SemanticRetriever(semantic, emb)
    consolidation = ConsolidationPipeline(
        episodic,
        semantic,
        mock_model_client,
        agent_id="test-agent",
    )

    return MemoryManager(
        working,
        episodic,
        semantic,
        procedural,
        ep_retriever,
        sem_retriever,
        consolidation,
        agent_id="test-agent",
    )


# ---------------------------------------------------------------------------
# Agent runner fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def agent_runner(
    mock_model_client: MockModelClient,
    tool_executor: Any,
) -> Any:
    from nexus.orchestration.runner import AgentRunner, RunnerConfig

    return AgentRunner(
        mock_model_client,
        tool_executor,
        config=RunnerConfig(max_iterations=5, enable_memory=False),
    )


# ---------------------------------------------------------------------------
# Safety pipeline fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def safety_pipeline() -> Any:
    from nexus.safety.injection import DetectionLevel, PromptInjectionDetector
    from nexus.safety.pii import PIIDetector
    from nexus.safety.pipeline import SafetyPipeline

    return SafetyPipeline(
        injection_detector=PromptInjectionDetector(level=DetectionLevel.BALANCED),
        pii_detector=PIIDetector(),
    )


# ---------------------------------------------------------------------------
# Session ID helper
# ---------------------------------------------------------------------------


def make_session_id() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"
