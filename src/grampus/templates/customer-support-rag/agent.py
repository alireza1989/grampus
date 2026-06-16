"""{{project_name}} — customer support agent with RAG knowledge base."""

from __future__ import annotations

import asyncio
from pathlib import Path

from grampus.core.types import AgentDefinition
from grampus.orchestration.runner import AgentRunner

KNOWLEDGE_BASE_PATH = Path("{{knowledge_base_path}}")

SYSTEM_PROMPT = f"""You are a helpful customer support agent.
You have access to a knowledge base at {KNOWLEDGE_BASE_PATH}.
Use the file_read tool to retrieve relevant articles before answering.
Always be polite, accurate, and concise.
If you cannot find the answer in the knowledge base, say so clearly.
"""


async def main() -> None:
    agent = AgentRunner(
        agent_def=AgentDefinition(
            name="{{project_name}}",
            model="{{model}}",
            system_prompt=SYSTEM_PROMPT,
            max_iterations=10,
            temperature=0.1,
            memory_enabled=True,
        ),
    )

    import sys

    question = " ".join(sys.argv[1:]) or input("Customer question: ")
    result = await agent.run(question)
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
