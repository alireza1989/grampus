"""{{project_name}} — single-agent starter."""

from __future__ import annotations

import asyncio

from grampus.core.types import AgentDefinition
from grampus.orchestration.runner import AgentRunner


async def main() -> None:
    agent = AgentRunner(
        agent_def=AgentDefinition(
            name="{{project_name}}",
            model="{{model}}",
            system_prompt=(
                "You are a helpful assistant. Use the web_search tool to find "
                "up-to-date information when needed."
            ),
            max_iterations=10,
            temperature=0.2,
            memory_enabled=False,
        ),
    )

    import sys

    question = " ".join(sys.argv[1:]) or input("Ask me anything: ")
    result = await agent.run(question)
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
