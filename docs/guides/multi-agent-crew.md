# Multi-Agent Crew Guide

## What you'll build

A three-agent crew that collaborates to produce a research report:

- **Researcher** — gathers information using web search
- **Critic** — identifies gaps, contradictions, and missing citations
- **Writer** — produces the final polished report

The agents run sequentially: each agent's output becomes the next agent's input.

---

## Prerequisites

- Nexus installed with Anthropic support: `pip install "nexus-ai[anthropic]"`
- Dapr and Docker running locally
- `NEXUS_MODEL__ANTHROPIC_API_KEY` set

---

## Step 1 — Define the tools

```python
# crew_agent.py
import asyncio
import os
from typing import Any

from nexus.core.models.anthropic import AnthropicClient
from nexus.core.types import AgentDefinition, ToolParameter
from nexus.orchestration.crew import Crew, CrewMember, CrewPattern
from nexus.orchestration.runner import AgentRunner, RunnerConfig
from nexus.tools.executor import ToolExecutor
from nexus.tools.registry import ToolRegistry

# Each agent gets its own registry (tools can overlap)
researcher_registry = ToolRegistry()
critic_registry = ToolRegistry()
writer_registry = ToolRegistry()


@researcher_registry.tool(
    name="web_search",
    description="Search the web for information.",
    parameters=[
        ToolParameter(name="query", type="string", description="Search query", required=True),
    ],
)
async def web_search(query: str) -> dict[str, Any]:
    return {
        "results": [
            {"title": f"Article about {query}", "snippet": f"Key finding: {query} is important."},
            {"title": f"{query} overview", "snippet": f"Background on {query}."},
        ]
    }


@critic_registry.tool(
    name="fact_check",
    description="Check whether a claim can be verified.",
    parameters=[
        ToolParameter(name="claim", type="string", description="The claim to check", required=True),
    ],
)
async def fact_check(claim: str) -> dict[str, Any]:
    return {"claim": claim, "verdict": "unverified", "confidence": 0.5}
```

---

## Step 2 — Build the crew members

```python
# continued from crew_agent.py

def make_client() -> AnthropicClient:
    return AnthropicClient(api_key=os.environ["NEXUS_MODEL__ANTHROPIC_API_KEY"])


def make_researcher() -> CrewMember:
    client = make_client()
    executor = ToolExecutor(researcher_registry)
    runner = AgentRunner(
        model_client=client,
        tool_executor=executor,
        config=RunnerConfig(max_iterations=8, enable_memory=False),
    )
    agent_def = AgentDefinition(
        name="researcher",
        model="claude-sonnet-4-6",
        system_prompt=(
            "You are a thorough research analyst. Search for information on the given topic "
            "and produce a detailed factual summary with key findings. "
            "Use web_search to gather data. Include source URLs."
        ),
        tools=["web_search"],
        max_iterations=8,
    )
    return CrewMember(agent_def=agent_def, runner=runner, role="researcher")


def make_critic() -> CrewMember:
    client = make_client()
    executor = ToolExecutor(critic_registry)
    runner = AgentRunner(
        model_client=client,
        tool_executor=executor,
        config=RunnerConfig(max_iterations=5, enable_memory=False),
    )
    agent_def = AgentDefinition(
        name="critic",
        model="claude-sonnet-4-6",
        system_prompt=(
            "You are a rigorous fact-checker and editor. Review the research provided "
            "and identify: (1) unsupported claims, (2) missing context, (3) logical gaps. "
            "Use fact_check on specific claims. Output a structured critique."
        ),
        tools=["fact_check"],
        max_iterations=5,
    )
    return CrewMember(agent_def=agent_def, runner=runner, role="critic")


def make_writer() -> CrewMember:
    client = make_client()
    # Writer uses no tools — pure synthesis
    executor = ToolExecutor(writer_registry)
    runner = AgentRunner(
        model_client=client,
        tool_executor=executor,
        config=RunnerConfig(max_iterations=3, enable_memory=False),
    )
    agent_def = AgentDefinition(
        name="writer",
        model="claude-sonnet-4-6",
        system_prompt=(
            "You are a professional technical writer. Given research findings and a critic's "
            "notes, synthesize a polished, well-structured report. Use markdown headers. "
            "Address all gaps the critic identified."
        ),
        tools=[],
        max_iterations=3,
    )
    return CrewMember(agent_def=agent_def, runner=runner, role="writer")
```

---

## Step 3 — Assemble and run the crew

```python
# continued from crew_agent.py

async def main() -> None:
    crew = Crew(
        members=[make_researcher(), make_critic(), make_writer()],
        pattern=CrewPattern.SEQUENTIAL,   # researcher → critic → writer
        session_id="crew-report-001",
    )

    topic = "The current state of open-source agentic AI frameworks in 2025"
    print(f"Starting crew run on: {topic}\n{'='*60}\n")

    result = await crew.run(initial_input=topic)

    print("=== RESEARCHER OUTPUT ===")
    print(result.outputs.get("researcher", ""))

    print("\n=== CRITIC OUTPUT ===")
    print(result.outputs.get("critic", ""))

    print("\n=== WRITER OUTPUT (FINAL REPORT) ===")
    print(result.outputs.get("writer", ""))

    print(f"\n{'='*60}")
    print(f"Total cost:    ${result.total_cost_usd:.4f}")
    print(f"Duration:      {result.duration_seconds:.1f}s")
    print(f"Pattern:       {result.pattern}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Step 4 — Run it

```bash
nexus run crew_agent.py --input "The state of open-source agentic AI in 2025"
```

---

## How sequential context passing works

In `CrewPattern.SEQUENTIAL`, each agent receives a combined input:

```
[original_topic]

--- researcher output ---
[researcher's findings]
```

Then the critic receives:

```
[original_topic]

--- researcher output ---
[researcher's findings]

--- critic input ---
Please critique the above research.
```

And the writer receives all prior outputs concatenated, giving it full context for synthesis.

---

## Patterns

=== "SEQUENTIAL (recommended for pipelines)"

    ```python
    crew = Crew(
        members=[researcher, critic, writer],
        pattern=CrewPattern.SEQUENTIAL,
        session_id="session-1",
    )
    ```

    Each agent runs after the previous one completes. Output accumulates.

=== "PARALLEL (independent tasks)"

    ```python
    crew = Crew(
        members=[agent_a, agent_b, agent_c],
        pattern=CrewPattern.PARALLEL,
        session_id="session-2",
    )
    ```

    All agents run concurrently on the same input. Results are collected independently.

=== "HIERARCHICAL (supervisor delegates)"

    ```python
    supervisor = CrewMember(agent_def=..., runner=..., role="supervisor")
    worker_a = CrewMember(agent_def=..., runner=..., role="worker")
    worker_b = CrewMember(agent_def=..., runner=..., role="worker")

    crew = Crew(
        members=[supervisor, worker_a, worker_b],
        pattern=CrewPattern.HIERARCHICAL,
        session_id="session-3",
    )
    ```

    The supervisor (first member with `role="supervisor"`) orchestrates the workers.

---

## Error handling

```python
from nexus.core.errors import OrchestrationError

try:
    result = await crew.run(initial_input=topic)
except OrchestrationError as e:
    print(f"Crew failed: {e}")
    print(f"Error code: {e.code}")       # e.g. "CREW_MEMBER_FAILED"
    print(f"Details:    {e.details}")
```

---

## Next steps

- **[Memory guide →](memory.md)** — Add shared episodic memory across crew members
- **[Evaluation guide →](evaluation.md)** — Write eval cases to test crew output quality
- **[Orchestration API →](../reference/orchestration-api.md)** — Full `Crew` and `CrewMember` reference