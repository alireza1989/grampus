# Single-Agent Guide: Research Agent

## What you'll build

A research agent that:

1. Accepts a research question as input
2. Uses a `web_search` tool to gather information
3. Summarizes results with the LLM
4. Stores the research in episodic memory
5. Recalls previous research in future sessions

---

## Prerequisites

- Nexus installed: `pip install "nexus-ai[anthropic]"`
- Dapr and Docker running locally
- `NEXUS_MODEL__ANTHROPIC_API_KEY` set in your environment

---

## Step 1 — Define the tool

```python
# research_agent.py
import asyncio
import os
from typing import Any

from nexus.core.models.anthropic import AnthropicClient
from nexus.core.types import AgentDefinition, ToolParameter
from nexus.orchestration.runner import AgentRunner, RunnerConfig
from nexus.tools.executor import ToolExecutor
from nexus.tools.registry import ToolRegistry

# ── Tool Registry ─────────────────────────────────────────────────────────────

registry = ToolRegistry()


@registry.tool(
    name="web_search",
    description="Search the web for up-to-date information on a topic.",
    parameters=[
        ToolParameter(name="query", type="string", description="Search query", required=True),
        ToolParameter(
            name="num_results",
            type="integer",
            description="Number of results to return",
            required=False,
            default=5,
        ),
    ],
)
async def web_search(query: str, num_results: int = 5) -> dict[str, Any]:
    """Simulate a web search. Replace with a real search API in production."""
    # In production: call SerpAPI, Brave Search, or Tavily here
    return {
        "query": query,
        "results": [
            {
                "title": f"Result {i+1} for '{query}'",
                "snippet": f"This is a simulated result about {query}.",
                "url": f"https://example.com/result-{i+1}",
            }
            for i in range(num_results)
        ],
    }
```

---

## Step 2 — Wire up memory

```python
# continued from research_agent.py

from nexus.dapr.client import DaprClient
from nexus.dapr.state import DaprStateStore
from nexus.memory.embeddings import EmbeddingService
from nexus.memory.episodic import EpisodicMemory
from nexus.memory.manager import MemoryManager
from nexus.memory.procedural import ProceduralMemory
from nexus.memory.retriever import EpisodicRetriever
from nexus.memory.semantic import SemanticMemory
from nexus.memory.semantic_retriever import SemanticRetriever
from nexus.memory.summarizer import Summarizer
from nexus.memory.working import WorkingMemory
from nexus.memory.consolidation import ConsolidationPipeline


def build_memory_manager(agent_id: str, model_client: AnthropicClient) -> MemoryManager:
    dapr = DaprClient()
    state_store = DaprStateStore(dapr, namespace="research")
    embedding_service = EmbeddingService(model_client=model_client)

    working = WorkingMemory(
        summarizer=Summarizer(model_client=model_client),
        token_limit=100_000,
    )
    episodic = EpisodicMemory(state_store=state_store, embedding_service=embedding_service)
    semantic = SemanticMemory(state_store=state_store, embedding_service=embedding_service)
    procedural = ProceduralMemory(state_store=state_store)

    episodic_retriever = EpisodicRetriever(
        episodic_memory=episodic,
        recency_weight=0.3,
        similarity_weight=0.5,
        importance_weight=0.2,
    )
    semantic_retriever = SemanticRetriever(semantic_memory=semantic)
    consolidation = ConsolidationPipeline(
        episodic_memory=episodic,
        semantic_memory=semantic,
        model_client=model_client,
    )

    return MemoryManager(
        working_memory=working,
        episodic_memory=episodic,
        semantic_memory=semantic,
        procedural_memory=procedural,
        episodic_retriever=episodic_retriever,
        semantic_retriever=semantic_retriever,
        consolidation_pipeline=consolidation,
        agent_id=agent_id,
    )
```

---

## Step 3 — Add safety

```python
# continued from research_agent.py

from nexus.safety.action_guard import ActionGuard, AgentPolicy
from nexus.safety.injection import PromptInjectionDetector
from nexus.safety.pii import PIIDetector
from nexus.safety.pipeline import SafetyPipeline, SafetyPipelineConfig


def build_safety_pipeline() -> SafetyPipeline:
    injection_detector = PromptInjectionDetector(level="balanced")

    pii_detector = PIIDetector(
        action="redact",           # log | redact | block
        pii_types=["email", "phone", "ssn", "credit_card"],
    )

    # Allow web_search, deny everything else for this agent
    policy = AgentPolicy(
        allowed_tools=["web_search"],
        max_tool_calls_per_turn=10,
        max_consecutive_tool_calls=5,
    )
    action_guard = ActionGuard(policy=policy)

    return SafetyPipeline(
        injection_detector=injection_detector,
        pii_detector=pii_detector,
        action_guard=action_guard,
        config=SafetyPipelineConfig(
            check_user_input=True,
            check_tool_results=True,
            check_llm_output=True,
        ),
    )
```

---

## Step 4 — Build and run the agent

```python
# continued from research_agent.py

AGENT_ID = "research-agent-v1"


def create_runner() -> AgentRunner:
    client = AnthropicClient(api_key=os.environ["NEXUS_MODEL__ANTHROPIC_API_KEY"])
    executor = ToolExecutor(registry, timeout_seconds=30.0, max_retries=2)
    memory = build_memory_manager(AGENT_ID, client)
    config = RunnerConfig(max_iterations=10, memory_top_k=5, enable_memory=True)
    return AgentRunner(
        model_client=client,
        tool_executor=executor,
        memory_manager=memory,
        config=config,
        # Optional: pass a UserMemoryAdapter to enable per-user personalization.
        # user_memory_adapter=adapter,
    )


def create_agent_def() -> AgentDefinition:
    return AgentDefinition(
        name="research-agent",
        model="claude-sonnet-4-6",
        system_prompt=(
            "You are a research assistant. When given a question, search the web "
            "for relevant information and provide a comprehensive, well-sourced answer. "
            "Always cite your sources."
        ),
        tools=["web_search"],
        max_iterations=10,
        memory_enabled=True,
        cost_budget_usd=0.10,        # Hard stop at $0.10 per run
    )


async def main() -> None:
    runner = create_runner()
    agent = create_agent_def()

    question = "What are the latest developments in agentic AI frameworks?"
    print(f"Question: {question}\n")

    # Pass user_id to enable user modeling (optional — omit for anonymous sessions)
    result = await runner.run(agent, question, session_id="research-session-1", user_id="alice")

    print(f"Answer:\n{result.output}\n")
    print(f"Tool calls made: {result.tool_calls_made}")
    print(f"Steps taken:     {result.steps_taken}")
    print(f"Total cost:      ${result.token_usage.cost_usd:.4f}")

    # Second run in same session — agent recalls previous research
    followup = "How does Nexus compare to what you found?"
    result2 = await runner.run(agent, followup, session_id="research-session-1", user_id="alice")
    print(f"\nFollowup answer:\n{result2.output}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Step 5 — Run it

```bash
nexus run research_agent.py --input "What are the latest developments in agentic AI?"
```

---

## What you built

- A `ToolRegistry` with a `web_search` tool registered via decorator
- A `ToolExecutor` that validates arguments, applies timeouts and retries
- A `MemoryManager` with all four memory types — new research stored in episodic memory, facts extracted to semantic memory
- Optional `user_id` on every `run()` call to activate the user modeling tier — agent adapts to each individual's expertise and preferences across sessions (see [User Modeling Guide](user_modeling.md))
- A `SafetyPipeline` that detects injection in web search results and redacts PII
- An `AgentRunner` wiring everything together into a ReAct loop with a $0.10 cost cap

---

## Next steps

- **[Multi-Agent Crew →](multi-agent-crew.md)** — Split research, critique, and writing across three agents
- **[Memory guide →](memory.md)** — Configure retrieval weights, consolidation, and memory security
- **[Evaluation guide →](evaluation.md)** — Write an eval suite to test your research agent