# Model Providers

Nexus supports multiple LLM providers through a unified `ModelClient` interface — switch providers by changing one line of configuration without touching the rest of your agent code.

---

## Supported providers

| Provider | Class | Extra | Example models |
|----------|-------|-------|---------------|
| Anthropic | `AnthropicClient` | `nexus-ai[anthropic]` | `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5` |
| OpenAI | `OpenAIClient` | `nexus-ai[openai]` | `gpt-4o`, `gpt-4o-mini`, `o1`, `o3` |
| Google Gemini | `GeminiClient` | `nexus-ai[gemini]` | `gemini-2.0-flash`, `gemini-1.5-pro` |
| Ollama (local) | `OllamaClient` | `nexus-ai[ollama]` | `llama3.2`, `mistral`, `qwen2.5`, `phi4`, `deepseek-r1`, any pulled model |

---

## Using each provider

### Anthropic

```bash
pip install "nexus-ai[anthropic]"
```

```python
import os

from nexus.core.models.anthropic import AnthropicClient

client = AnthropicClient(api_key=os.environ["NEXUS_MODEL__ANTHROPIC_API_KEY"])
```

Environment variable: `NEXUS_MODEL__ANTHROPIC_API_KEY`

---

### OpenAI

```bash
pip install "nexus-ai[openai]"
```

```python
import os

from nexus.core.models.openai import OpenAIClient

client = OpenAIClient(api_key=os.environ["NEXUS_MODEL__OPENAI_API_KEY"])
```

Environment variable: `NEXUS_MODEL__OPENAI_API_KEY`

---

### Google Gemini

```bash
pip install "nexus-ai[gemini]"
```

```python
import os

from nexus.core.models.gemini import GeminiClient

client = GeminiClient(api_key=os.environ["NEXUS_MODEL__GEMINI_API_KEY"])
```

Environment variable: `NEXUS_MODEL__GEMINI_API_KEY`

---

### Ollama (local models)

Ollama lets you run open-weight models locally with no API cost or data leaving your machine.

```bash
pip install "nexus-ai[ollama]"
```

**Step 1 — Install Ollama:**

=== "macOS"

    ```bash
    brew install ollama
    ```

=== "Linux"

    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```

**Step 2 — Start the Ollama server:**

```bash
ollama serve
```

**Step 3 — Pull a model:**

```bash
ollama pull llama3.2
# or
ollama pull mistral
# or
ollama pull qwen2.5
```

**Step 4 — Use with Nexus:**

```python
from nexus.core.models.ollama import OllamaClient

# Default: connects to http://localhost:11434
client = OllamaClient(host="http://localhost:11434")
```

!!! note "Token usage with Ollama"
    Ollama models have zero API cost. Token usage is still tracked for context window management and working memory summarization triggers.

---

## Using providers with AgentRunner

Pass a client directly to `AgentRunner`, or set `model` in `AgentDefinition` — the `ModelRouter` resolves the client from the configured providers:

```python
import asyncio
import os

from nexus.core.models.gemini import GeminiClient
from nexus.core.types import AgentDefinition
from nexus.orchestration.runner import AgentRunner, RunnerConfig
from nexus.tools.executor import ToolExecutor
from nexus.tools.registry import ToolRegistry


async def main() -> None:
    client = GeminiClient(api_key=os.environ["NEXUS_MODEL__GEMINI_API_KEY"])
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    config = RunnerConfig(max_iterations=5, enable_memory=False)

    runner = AgentRunner(model_client=client, tool_executor=executor, config=config)
    agent_def = AgentDefinition(
        name="gemini-agent",
        model="gemini-2.0-flash",
        system_prompt="You are a helpful assistant.",
    )

    result = await runner.run(agent_def, "What is the capital of Japan?")
    print(result.output)


asyncio.run(main())
```

Switching from Gemini to Ollama is one-line:

```python
from nexus.core.models.ollama import OllamaClient

client = OllamaClient(host="http://localhost:11434")
agent_def = AgentDefinition(
    name="ollama-agent",
    model="llama3.2",
    system_prompt="You are a helpful assistant.",
)
```

---

## Model router

For production deployments, the `ModelRouter` automatically selects the cheapest model capable of handling each step, with fallback on failure. Models are grouped into tiers:

| Tier | Example models | Use case |
|------|---------------|----------|
| `fast` | `claude-haiku-4-5`, `gemini-2.0-flash`, `llama3.2` | Simple reasoning, tool arg generation |
| `balanced` | `claude-sonnet-4-6`, `gpt-4o-mini`, `qwen2.5` | Most tasks |
| `powerful` | `claude-opus-4-7`, `gpt-4o`, `o1` | Complex reasoning, synthesis |

Configure routing in `nexus.yaml`:

```yaml
model:
  default_model: claude-sonnet-4-6
  router:
    enabled: true
    fast: claude-haiku-4-5
    balanced: claude-sonnet-4-6
    powerful: claude-opus-4-7
```

See the [Observability guide](observability.md) for tracking cost per model tier.

---

## See also

- **[Prompt Playground →](playground.md)** — Test prompts across multiple providers interactively
- **[Cost Management →](cost-management.md)** — Track and alert on per-model spending
- **[Configuration reference →](../reference/config.md)** — Full `ModelConfig` field reference
