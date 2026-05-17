"""Deep Research Demo — main entry point.

Exports two symbols required by ``nexus run agent.py``:
  create_runner()   — builds an AgentRunner for the supervisor (single-agent mode)
  create_agent_def() — returns the supervisor's AgentDefinition

Also exports ``build_crew()`` for ``python run.py`` which wires the full
4-agent hierarchical crew (supervisor → researcher, fact-checker, writer).

Usage:
    nexus run demos/deep_research/agent.py --input "quantum computing in drug discovery"
    python demos/deep_research/run.py "quantum computing in drug discovery"
"""

from __future__ import annotations

import os

from demos.deep_research._mock_client import MockModelClient
from demos.deep_research._store import FakeStateStore
from demos.deep_research.agents.fact_checker import create_fact_checker_def
from demos.deep_research.agents.researcher import create_researcher_def
from demos.deep_research.agents.supervisor import create_supervisor_def
from demos.deep_research.agents.writer import create_writer_def

# Importing tool modules registers their functions on their module-level registries
from demos.deep_research.tools import analysis as _analysis_mod
from demos.deep_research.tools import output as _output_mod
from demos.deep_research.tools import search as _search_mod
from nexus.core.models.anthropic import AnthropicClient
from nexus.core.types import AgentDefinition
from nexus.memory.consolidation import ConsolidationPipeline
from nexus.memory.embeddings import EmbeddingService
from nexus.memory.episodic import EpisodicMemory
from nexus.memory.manager import MemoryManager
from nexus.memory.procedural import ProceduralMemory
from nexus.memory.retriever import EpisodicRetriever
from nexus.memory.semantic import SemanticMemory
from nexus.memory.semantic_retriever import SemanticRetriever
from nexus.memory.summarizer import SummarizationStrategy, Summarizer
from nexus.memory.token_counter import TokenCounter
from nexus.memory.working import WorkingMemory
from nexus.observability.events import EventLog
from nexus.observability.metrics import NexusMetrics
from nexus.observability.tracer import NexusTracer
from nexus.orchestration.cost_tracker import CostTracker
from nexus.orchestration.crew import Crew, CrewMember, CrewPattern
from nexus.orchestration.runner import AgentRunner, RunnerConfig
from nexus.safety.policies import PolicyLoader
from nexus.tools.executor import ToolExecutor
from nexus.tools.registry import ToolRegistry

SESSION_ID = "deep-research-demo"
AGENT_ID = "deep-research-supervisor"
_POLICY_PATH = "demos/deep_research/config/safety_policy.yaml"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


def build_registry() -> ToolRegistry:
    """Build a combined ToolRegistry from all demo tool modules."""
    registry = ToolRegistry()
    for rt in _search_mod._registry.list_all():
        registry.register(
            rt.fn, name=rt.name, description=rt.description, parameters=rt.definition.parameters
        )
    for rt in _analysis_mod._registry.list_all():
        registry.register(
            rt.fn, name=rt.name, description=rt.description, parameters=rt.definition.parameters
        )
    for rt in _output_mod._registry.list_all():
        registry.register(
            rt.fn, name=rt.name, description=rt.description, parameters=rt.definition.parameters
        )
    return registry


# ---------------------------------------------------------------------------
# Memory stack
# ---------------------------------------------------------------------------


def _build_embedding_service(api_key: str) -> EmbeddingService:
    """Build the embedding service (falls back to no-op when no API key)."""
    return EmbeddingService(api_key=api_key) if api_key else _FakeEmbeddingService()


class _FakeEmbeddingService:
    """No-op embedding service when no API key is available."""

    async def embed(self, text: str) -> list[float]:
        h = hash(text) % (10**8)
        raw = [float((h >> i) & 0xFF) / 255.0 + 0.01 for i in range(8)]
        norm = sum(x**2 for x in raw) ** 0.5 or 1.0
        return [x / norm for x in raw]


def _build_memory_manager(
    store: FakeStateStore, model_client: object, agent_id: str
) -> MemoryManager:
    """Wire all memory components together into a MemoryManager."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    emb = _build_embedding_service(api_key)

    episodic = EpisodicMemory(store, emb, agent_id=agent_id)
    semantic = SemanticMemory(store, agent_id=agent_id)
    procedural = ProceduralMemory(store, agent_id=agent_id)

    token_counter = TokenCounter("claude-sonnet-4-6")
    summarizer = Summarizer(
        model_client,
        "claude-sonnet-4-6",
        SummarizationStrategy.TRUNCATE,
        token_counter,
    )
    working = WorkingMemory(
        store,
        token_counter,
        summarizer,
        agent_id=agent_id,
        session_id=SESSION_ID,
        max_tokens=6000,
    )

    ep_retriever = EpisodicRetriever(episodic, emb)
    sem_retriever = SemanticRetriever(semantic, emb)
    consolidation = ConsolidationPipeline(episodic, semantic, model_client, agent_id=agent_id)

    return MemoryManager(
        working,
        episodic,
        semantic,
        procedural,
        ep_retriever,
        sem_retriever,
        consolidation,
        agent_id=agent_id,
    )


# ---------------------------------------------------------------------------
# Model client
# ---------------------------------------------------------------------------


def _build_model_client() -> object:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        return AnthropicClient(api_key=api_key)
    return MockModelClient()


# ---------------------------------------------------------------------------
# Runner factory
# ---------------------------------------------------------------------------


def _build_runner(
    model_client: object,
    executor: ToolExecutor,
    store: FakeStateStore,
    agent_id: str,
    *,
    budget_usd: float = 2.0,
    enable_memory: bool = True,
) -> AgentRunner:
    memory = _build_memory_manager(store, model_client, agent_id) if enable_memory else None
    cost_tracker = CostTracker(
        agent_id=agent_id,
        session_id=SESSION_ID,
        budget_usd=budget_usd,
    )
    return AgentRunner(
        model_client,
        executor,
        memory_manager=memory,
        cost_tracker=cost_tracker,
        state_store=store,
        config=RunnerConfig(max_iterations=15, enable_memory=enable_memory),
    )


# ---------------------------------------------------------------------------
# Public API — required by `nexus run agent.py`
# ---------------------------------------------------------------------------


def create_runner() -> AgentRunner:
    """Build the supervisor AgentRunner. Called by ``nexus run agent.py``."""
    model_client = _build_model_client()
    registry = build_registry()
    executor = ToolExecutor(registry=registry, timeout_seconds=30)
    store = FakeStateStore()

    _tracer = NexusTracer(service_name="deep-research-demo", agent_id=AGENT_ID)
    _metrics = NexusMetrics(agent_id=AGENT_ID)
    _event_log = EventLog(agent_id=AGENT_ID, session_id=SESSION_ID)

    # Suppress unused-variable warnings — tracer/metrics/event_log are wired
    # to the observability infrastructure; they record via side effects.
    del _tracer, _metrics, _event_log

    return _build_runner(model_client, executor, store, AGENT_ID, budget_usd=2.0)


def create_agent_def() -> AgentDefinition:
    """Return the supervisor AgentDefinition. Called by ``nexus run agent.py``."""
    return create_supervisor_def()


# ---------------------------------------------------------------------------
# Crew factory — used by run.py
# ---------------------------------------------------------------------------


def build_crew() -> tuple[Crew, str]:
    """Build the full 4-agent hierarchical research crew.

    Returns:
        (crew, session_id) tuple.
    """
    import uuid

    session_id = f"research-{uuid.uuid4().hex[:8]}"
    model_client = _build_model_client()
    registry = build_registry()
    executor = ToolExecutor(registry=registry, timeout_seconds=30)
    store = FakeStateStore()

    # Load safety policy
    try:
        policy = PolicyLoader.load(_POLICY_PATH)
    except Exception:
        policy = PolicyLoader.load(None)

    # Build per-agent runners (shared store for cross-agent state)
    supervisor_runner = _build_runner(model_client, executor, store, AGENT_ID, budget_usd=2.0)
    researcher_runner = _build_runner(model_client, executor, store, "researcher", budget_usd=0.50)
    fact_checker_runner = _build_runner(
        model_client, executor, store, "fact-checker", budget_usd=0.30
    )
    writer_runner = _build_runner(
        model_client, executor, store, "writer", budget_usd=0.30, enable_memory=False
    )

    # Attach safety pipeline to supervisor runner's policy (informational)
    _supervisor_pipeline = PolicyLoader.build_pipeline(policy, agent_id=AGENT_ID)
    del _supervisor_pipeline  # pipeline integrated via policy object

    crew = Crew(
        members=[
            CrewMember(
                agent_def=create_supervisor_def(), runner=supervisor_runner, role="supervisor"
            ),
            CrewMember(agent_def=create_researcher_def(), runner=researcher_runner, role="worker"),
            CrewMember(
                agent_def=create_fact_checker_def(), runner=fact_checker_runner, role="worker"
            ),
            CrewMember(agent_def=create_writer_def(), runner=writer_runner, role="worker"),
        ],
        pattern=CrewPattern.HIERARCHICAL,
        shared_state_store=store,
        session_id=session_id,
    )

    return crew, session_id
