# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Long-Horizon Planning (Phase E34)** — `PlanningRunner` orchestrates complex multi-step
  tasks via a structured SubGoal DAG. The planner decomposes any user task into a
  topologically sorted set of `SubGoal` objects; independent subgoals execute concurrently
  via `asyncio.gather` (parallel waves). Each subgoal runs in a scoped context containing
  only the global task + one-line summaries of completed steps + the current subgoal
  description — eliminating ~82% of token overhead versus passing full conversation history
  (Task-Decoupled Planning, arXiv 2601.07577).

  Key features:
  - **Adaptive routing** — a single cheap complexity-estimate call skips planning entirely
    for tasks estimated at ≤ 4 tool calls, eliminating overhead for simple queries.
  - **FLARE-style lookahead** — before each subgoal, `LookaheadSimulator` generates `n`
    candidate execution paths and selects the highest-scoring approach as a hint
    (arXiv 2601.22311). Advisory only; parse failures are silently swallowed.
  - **Retry / fallback control flow** — `PASS` → done; `PARTIAL` → retry up to
    `max_retries`; `FAIL` → try `fallback_strategy` once; still failing → trigger
    partial replan (ReAcTree, arXiv 2511.02424).
  - **Partial replanning** — `Replanner` generates only the downstream subgoals, preserving
    all completed work (Google DeepMind Subgoal Framework, arXiv 2603.19685). Version
    counter increments on each replan; `MAX_REPLANS_EXCEEDED` raised after `max_replans`.
  - **`PostconditionVerifier`** — fast-model LLM call after each subgoal determines
    pass / partial / fail against a verifiable success criterion.
  - **`planning_node()`** — graph node factory that wraps `PlanningRunner` for composable
    multi-step pipelines; injects `PlanResult` into `AgentState.metadata["plan_result"]`
    and sets `state.status` to `COMPLETED` or `FAILED`.
  - **`PlanningError`** — new exception class with codes `CIRCULAR_DEPENDENCY`,
    `MAX_REPLANS_EXCEEDED`, `REPLAN_PARSE_FAILED`, `PLAN_PARSE_FAILED`, `NO_SUBGOALS`.
  - **33 new tests** in `tests/orchestration/test_planning.py` covering all components
    end-to-end with `FakeModelClient` and `FakeAgentRunner` — zero real LLM calls.
  - **Documentation** — new [Long-Horizon Planning guide](docs/guides/long-horizon-planning.md),
    updated [Orchestration API reference](docs/reference/orchestration-api.md) with full
    type tables, and updated [Error reference](docs/reference/errors.md) with
    `PlanningError` codes and remediation guidance.

## [0.1.0] - 2026-05-04

### Added

- **Core** — `GrampusConfig` with pydantic-settings, nested sub-configs (`ModelConfig`,
  `MemoryConfig`, `SafetyConfig`, `DaprConfig`, `ObservabilityConfig`), env-var loading
  with `GRAMPUS_` prefix and YAML override. Error hierarchy with 9 exception types
  (`ConfigError`, `MemoryError`, `MemorySecurityError`, `ToolError`, `ToolTimeoutError`,
  `OrchestrationError`, `BudgetExceededError`, `SafetyError`, `ModelError`) all rooted
  at `GrampusError` with machine-readable `code` fields. Structured logging via structlog
  with JSON (production) and console (development) renderers, ISO 8601 timestamps, and
  correlation ID propagation via contextvars.

- **Type system** — Pydantic v2 models for all public types: `Role`, `AgentStatus`,
  `ToolCall`, `ToolResult`, `Message`, `ToolParameter`, `ToolDefinition` (with
  `to_function_schema()`), `AgentDefinition`, `TokenUsage`, `AgentState`,
  `ExecutionResult`. All models round-trip serialize to and from JSON.

- **Model clients** — `ModelClient` ABC with `async complete()` and `async stream()`.
  `AnthropicClient` maps Grampus types to Anthropic SDK format, extracts `TokenUsage`
  from response, and wraps API errors in `ModelError`. `OpenAIClient` does the same
  for the OpenAI SDK. Both clients support streaming tool call assembly.

- **Dapr integration** — Typed Python wrappers for all Dapr building blocks:
  `DaprStateStore` with namespace-scoped keys (`{ns}:{entity}:{id}`), Pydantic
  serialization, ETag-based optimistic concurrency, bulk operations, and transactions.
  `DaprPubSub` with typed publish/subscribe and handler registration. `DaprLock` as
  async context manager. `DaprHealth` for sidecar readiness checks.

- **Working memory** — `WorkingMemory` with model-aware token counting via tiktoken,
  configurable sliding window, and three summarization strategies: `truncate` (drop
  oldest), `summarize` (LLM-generated summary preserving key facts), and `hybrid`
  (summarize old messages, keep N most recent at full fidelity). Full uncompressed
  history persisted to Dapr state for audit. Property-based tests via Hypothesis.

- **Episodic memory** — `EpisodicMemory` with cross-session record persistence via
  Dapr state (PostgreSQL backend). `EpisodicRecord` carries embedding, importance
  score (0–1), trust score, provenance, and access tracking. Temporal decay reduces
  relevance of old records at a configurable rate. `EpisodicRetriever` uses hybrid
  scoring: `score = α·recency + β·similarity + γ·importance` with configurable
  weights and top-K selection.

- **Semantic memory** — `SemanticMemory` stores subject-predicate-object facts with
  confidence scores. Incoming facts are deduplicated against existing knowledge;
  conflicts are resolved by confidence-weighted replacement and flagged for review.
  `ConsolidationPipeline` runs as an async background task, scanning recent episodic
  records, using an LLM to extract facts, and updating semantic memory without
  blocking agent execution.

- **Procedural memory** — `ProceduralMemory` stores learned workflows as `Procedure`
  objects containing `ProcedureStep` sequences with tool names, parameter templates,
  and expected outcomes. `ProcedureExtractor` analyzes completed tool-call sequences
  post-execution and uses an LLM to generalize them into reusable templates.
  `ProcedureMatcher` retrieves relevant procedures for new tasks via semantic matching.

- **Memory security** — `ProvenanceTracker` annotates every memory write with
  `Provenance(source_type, source_id, trust_level, timestamp, content_hash_sha256)`.
  Source types carry default trust levels: `SYSTEM` (1.0), `USER_INPUT` (0.9),
  `LLM_GENERATED` (0.7), `TOOL_RESULT` (0.6), `EXTERNAL_DATA` (0.3). `TrustScorer`
  applies temporal decay and access-count boosts. `MemoryValidator` pre-write pipeline
  detects instruction injection patterns, enforces size limits, and applies rate
  limiting per source. `MemoryAuditor` verifies content hash integrity and flags
  broken provenance chains. All writes go through validator + provenance before hitting
  Dapr state; writes without provenance are rejected.

- **Unified `MemoryManager`** — Single interface to all four memory types. Methods:
  `remember(content)`, `recall(query, memory_types)`, `forget(record_id)`,
  `consolidate()`. This is the only memory surface the orchestration layer touches.

- **Tool registry** — `ToolRegistry` with decorator API (`@registry.tool(...)`), JSON
  schema generation from type hints and docstrings, and version management. Tools are
  discoverable as lists of `ToolDefinition` objects compatible with LLM function-calling
  APIs.

- **Tool executor** — `ToolExecutor` validates arguments against schema, executes with
  configurable timeout and retry, records full execution traces (input, output, duration,
  error), and supports idempotency keys for workflow replay.

- **MCP client** — `MCPClient` implements JSON-RPC 2.0 MCP protocol. Discovers tool
  servers, lists available tools, and invokes them. External tool results are
  automatically tagged with `EXTERNAL_DATA` provenance.

- **Sandbox** — `SandboxManager` runs tool code in isolated Docker containers with
  configurable network access, filesystem mounts, memory limits, and CPU limits.
  Container pooling reduces warm-start overhead to ~10 ms. Subprocess fallback available
  for development environments without Docker.

- **Code executor** — `CodeExecutor` runs LLM-generated Python in the sandbox.
  Registered tools are injected as callables into the execution namespace. stdout,
  stderr, and return values are captured.

- **Action guard** — `ActionGuard` enforces per-agent tool allowlists and denylists,
  sliding-window rate limiting (configurable max calls per minute), and cost guards
  (max USD per action and per session).

- **Graph engine** — Async `Graph` with nodes (async callable handlers), typed edges
  (conditional and unconditional), and a builder-pattern API. Executes by walking the
  graph and passing `AgentState` through nodes. Independent branches run in parallel.
  State is checkpointed to Dapr after each node, enabling restart from the last
  successful node after a crash.

- **Pre-built graph nodes** — `LLMNode` (invoke model client), `ToolNode` (execute
  tool calls), `HumanNode` (pause and wait for input), `ConditionalNode` (route on
  state), `SubgraphNode` (compose nested graphs).

- **Model router** — `ModelRouter` routes steps to the cheapest capable model using
  three tiers (`fast`, `balanced`, `powerful`) with configurable routing rules and
  automatic fallback chains on API failures.

- **Cost tracker** — `CostTracker` accounts for token usage and cost per model, step,
  agent, and session. Budget enforcement halts execution when limits are reached. Cost
  events are published to Dapr pub/sub for external monitoring.

- **`AgentRunner`** — Main ReAct execution loop integrating memory, tools, graph,
  model router, cost tracker, and safety pipeline. Supports both ReAct and
  Plan-and-Execute patterns. Max-iterations guard prevents runaway loops.

- **Crew** — `Crew` multi-agent coordinator supporting sequential, parallel, and
  hierarchical patterns. Shared state is accessed via Dapr state + distributed locks.
  Supervisor pattern allows one agent to delegate to specialized sub-agents.

- **Safety pipeline** — `SafetyPipeline` middleware wraps all LLM calls, tool calls,
  and memory writes. `PromptInjectionDetector` operates in three strictness levels
  (strict, balanced, permissive) using regex and heuristic layers. `PIIDetector`
  recognizes six entity types (email, phone, SSN, credit card, address, name) with
  configurable actions (log, redact, block). All safety configuration is loadable
  from YAML policy files.

- **OTEL tracing** — `GrampusTracer` produces six custom span types: `agent.run`,
  `agent.llm_call`, `agent.tool_call`, `agent.memory_read`, `agent.memory_write`,
  `agent.decision`. All spans carry agent ID, session ID, model, token counts, cost,
  and step number. Session-level parent spans group all child spans.

- **Prometheus metrics** — In-process metrics exposition at `/metrics`. Counters:
  `total_tokens`, `total_cost_usd`, `tool_calls_total`, `errors_total`. Gauges:
  `active_agents`. Histograms: `llm_latency_seconds`, `tool_latency_seconds`.

- **Event log** — Append-only `EventLog` records every agent action as an immutable
  event. Events are replayable and audit-ready, stored via Dapr state.

- **Behavior monitor** — `BehaviorMonitor` tracks per-agent patterns over a rolling
  baseline and detects anomalies: cost spikes, tool usage shifts, error-rate increases.
  Consumes agent events via Dapr pub/sub and emits structured alerts.

- **Evaluation suite** — `EvalSuite` and `EvalCase` runner with tag filtering and
  configurable concurrency. Cases specify input, expected output, expected tool calls,
  and assertion lists.

- **15 assertion types** — `contains`, `not_contains`, `matches_regex`, `not_matches_regex`,
  `json_schema_valid`, `tool_was_called`, `tool_not_called`, `tool_call_count`,
  `semantic_similarity` (embedding-based), `llm_judge` (LLM-as-judge scoring),
  `no_pii` (PII safety), `no_injection` (injection safety), `max_cost_usd`,
  `max_latency_seconds`, `max_steps`.

- **Prompt version manager** — `PromptVersionManager` registers system prompt versions
  with content hashing, diffs between versions, runs A/B test scoring against eval
  suites, and supports rollback to any previous version.

- **Quality baseline** — `QualityBaseline` pins a reference run's scores, compares
  subsequent eval runs against the baseline, and reports regressions at configurable
  thresholds.

- **Eval reporter** — `EvalReporter` outputs results to stdout (human-readable),
  JSON (machine-readable), and JUnit XML (CI integration). Reports include per-case
  results, latency, cost, and quality scores. Results can be published to Dapr pub/sub.

- **CLI** — Six subcommands via Click:
  - `grampus init <name>` — scaffold a project from three templates (simple, crew, rag)
    with config, example agent, docker-compose, and Dapr components in under 10 seconds
  - `grampus run <agent.py>` — start an agent with interactive REPL or `--input` for
    single-shot execution; auto-starts Dapr sidecar in the background
  - `grampus eval <suite.py>` — run evaluation suite with `--fail-under` for CI gating
  - `grampus memory inspect/clear <agent_id>` — view or clear an agent's memory
  - `grampus cost` — show cost summary for recent sessions with per-model breakdown
  - `grampus dev` — watch mode with hot-reload, live cost display, and trace streaming

- **1,226 tests** — 1,070 unit tests and 156 integration/e2e tests covering all nine
  architecture layers. Property-based tests via Hypothesis for edge cases in memory
  and serialization.

- **Full documentation** — MkDocs Material site with Getting Started guide, 8
  topic guides, 9 API reference pages, 10 architecture decision records, and a
  security model overview.

[Unreleased]: https://github.com/grampus-ai/grampus-agentic-platform/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/grampus-ai/grampus-agentic-platform/releases/tag/v0.1.0