# Nexus Agentic Platform — Technical Overview

## What Is Nexus?

Nexus is an open-source production-grade agentic AI framework built on Dapr's distributed
runtime. It provides the agent intelligence layer — memory, orchestration, safety, and
observability — while Dapr handles infrastructure: state management, pub/sub messaging,
distributed locks, secrets, and mTLS encryption. The design goal is a framework that is as
simple as CrewAI to get started with, but as powerful as LangGraph for production deployments.

Nexus is written in Python 3.12+ and is async-first throughout. Every I/O operation uses
async/await. All public types are Pydantic v2 models, which means strict validation at
construction time, automatic JSON serialization, and JSON Schema generation for tool definitions.
The package is installed via `pip install nexus-ai` and the CLI entry point is `nexus`.

## Memory Architecture

Nexus implements four distinct memory layers, each solving a different aspect of agent memory:

**Working Memory** holds the current conversation context within a single session. It tracks
token usage using tiktoken and automatically summarizes when the context approaches the model's
window limit, preserving the most recent N messages at full fidelity while compressing older
history. The summarization strategy is configurable: `truncate` drops the oldest messages,
`summarize` uses an LLM call to produce a compact summary, and `hybrid` keeps recent messages
intact while summarizing the older portion.

**Episodic Memory** stores cross-session records with timestamps, importance scores, and
embedding vectors. Each record has a trust score derived from its source type: system-generated
content (score 1.0), user input (0.9), LLM-generated content (0.7), tool results (0.6), and
external data (0.3). Temporal decay reduces the retrieval score of old records automatically.
Episodic retrieval uses a hybrid scoring formula: `score = α×recency + β×similarity + γ×importance`.

**Semantic Memory** stores structured facts extracted from episodic records by a consolidation
pipeline that runs asynchronously in the background. Facts are represented as subject-predicate-
object triples. Deduplication runs at write time using cosine similarity — facts with similarity
above 0.90 merge rather than create duplicates. Conflicting facts are handled with confidence-
weighted replacement rather than silent overwrite.

**Procedural Memory** stores learned multi-step workflows extracted from successful task
executions. A procedure consists of named steps with action descriptions, tool names, parameter
templates, and expected outcomes. The procedure matcher uses semantic similarity to surface
relevant workflows for new tasks. Procedures have success and failure counts, enabling the
runtime to deprioritize unreliable workflows automatically.

All four layers are exposed through a single `MemoryManager` facade that the orchestration layer
talks to. Memory writes go through a provenance tracker — every write is annotated with source
type, source ID, trust level, and a SHA-256 content hash — and a validator that detects prompt
injection attempts using regex and heuristic patterns.

## Tool System

Nexus provides a `ToolRegistry` where functions are registered with their name, description, and
parameter schemas as `ToolParameter` objects. The registry serializes tool definitions to the JSON
Schema format expected by LLM APIs. Tools are called by name with validated arguments.

The built-in tool library includes: `file_read`, `file_write`, `http_request`, `web_search`,
`calculator`, `sql_query`, `send_email`, `read_pdf`, `read_docx`, `read_excel`, and five code
analysis tools (`analyze_file`, `lint_code`, `check_types`, `find_symbol`, `summarize_structure`).

Tools that require code execution run inside a Docker sandbox by default. Network access,
filesystem access, and resource limits are configured per-tool. MCP (Model Context Protocol)
support means any MCP-compatible tool server — filesystem servers, browser automation, database
connectors, and external APIs — can be used by Nexus agents without writing custom integration
code.

## Orchestration Engine

The `AgentRunner` implements the core agent execution loop. It accepts an `AgentDefinition`
(system prompt, model, tool list, budget), a `MemoryManager`, a set of tools, and optional
components like a `VersionRouter`, `UncertaintyMonitor`, `ReflexionEngine`, and `PluginManager`.

On each iteration the runner: loads relevant memories, calls the LLM, parses tool calls from the
response, executes them through the tool executor, updates memory with results, and checks
whether the loop should continue or return. Two execution patterns are supported: ReAct (reason
then act, iterative) and Plan-and-Execute via the `PlanningRunner` wrapper.

The `Graph` engine supports multi-node workflows with conditional branching, parallel execution
of independent branches via `asyncio.gather`, and checkpoint/restore via Dapr state. Pre-built
nodes include `LLMNode`, `ToolNode`, `ConditionalNode`, `HumanNode` (for human-in-the-loop
pausing), and `SubgraphNode` for nested workflow composition.

Multi-agent crews support sequential, parallel, and hierarchical patterns with shared state
mediated by Dapr distributed locks. A `DebateOrchestrator` enables multi-model consensus for
high-stakes decisions, with configurable debater panels, sycophancy resistance, and adaptive
routing that bypasses the full debate when confidence is already above threshold.

## Planning Runner

The `PlanningRunner` wraps `AgentRunner` to support long-horizon tasks by decomposing them into
subgoal DAGs before execution. Each subgoal executor receives only the global task, one-line
summaries of completed steps, and the current subgoal — never the full conversation history.
This reduces token usage by approximately 82% on long plans.

When a subgoal fails after `max_retries`, a pre-specified fallback strategy is attempted before
triggering a full replan. Replanning is always partial: only the downstream unfinished subgoals
are regenerated, and completed subgoal outputs are preserved. A cheap complexity estimate call
gates planning engagement — simple tasks estimated below a threshold delegate directly to
`AgentRunner`, avoiding planning overhead for approximately 40% of queries.

## Safety and Guardrails

The safety pipeline wraps every LLM call, tool call, and memory write with pre/post checks.
Prompt injection detection runs at three layers: regex patterns for known attack signatures,
heuristics for directive language ("remember that", "always", "in future conversations"), and
semantic classification for sophisticated embedding-space attacks. The detection level is
configurable: `strict`, `balanced`, or `permissive`.

PII detection supports email addresses, phone numbers, Social Security Numbers, credit card
numbers, and postal addresses with configurable actions: `log` (record but pass through),
`redact` (replace with a placeholder), or `block` (reject the call entirely). An optional spaCy
NER integration improves recall for named entities beyond pattern-matched types.

## Observability

Observability uses OpenTelemetry with custom span types for every agent action: `agent.run`,
`agent.llm_call`, `agent.tool_call`, `agent.memory_read`, `agent.memory_write`, and
`agent.decision`. Each span carries model name, token counts, cost in USD, duration, and the
agent ID as attributes, enabling cost analysis and latency profiling via trace queries.

Prometheus metrics expose token counts, total cost, tool call counts, error rates, active agent
gauge, and latency histograms for both LLM calls and tool execution. A `BehaviorMonitor` tracks
per-agent patterns over time via Dapr pub/sub and fires alerts when usage metrics drift beyond
configurable thresholds.

Every agent action writes to an append-only event log for full audit trails and time-travel
debugging. The log is structured, immutable, and replayable — a forensic record of exactly what
the agent decided and why at each step.

## Embedding Service

The `EmbeddingService` supports multiple providers: OpenAI (`text-embedding-3-small` with 1536
dimensions, `text-embedding-3-large` with 3072 dimensions), Cohere Embed v3 with the
`search_document` / `search_query` input type distinction required for quality retrieval, and
Ollama for local embeddings without any external API dependency.

The `.dimensions` property exposes the output vector dimension for pgvector column validation at
setup time, converting the silent dimension-mismatch failure mode — where switching providers
without updating the column width silently dropped all writes — into an explicit startup error.

The `EmbeddingRouter` enables per-memory-type provider routing: high-quality models for semantic
memory, faster local models for working memory, with automatic fallback to a configured default.

## Deployment

Nexus runs locally with Docker Compose (PostgreSQL with the pgvector extension, Redis 7, and the
Dapr placement service). Production deployment targets Kubernetes with the Dapr control plane
sidecar injected into each pod. The CLI provides `nexus init` (scaffold a new project), `nexus
run` (start agent with Dapr), `nexus eval` (run evaluation suites), `nexus deploy` (generate
Kubernetes manifests), and `nexus dev` (watch mode with auto-reload and live cost/trace display).

Configuration is loaded from environment variables (prefixed `NEXUS_`), YAML files, or code
construction. Sensitive values like API keys are stored as `SecretStr` and masked in log output.
Optional dependency groups keep the base install lightweight: `[openai]`, `[anthropic]`,
`[cohere]`, `[documents]`, `[rag]`, and `[all]` for everything.
