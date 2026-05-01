# PLAN.md — Nexus Implementation Plan

## How To Use This Plan

**For Claude Code sessions:**
1. Start a FRESH session for each phase (clean context)
2. Say: `Read @PLAN.md and implement Phase N. Plan first, don't implement yet.`
3. Review the plan Claude produces, correct anything wrong
4. Say: `Implement. Write tests first.`
5. After implementation, verify ALL acceptance criteria
6. Mark the checkbox and commit: `feat(module): phase N description`

**Rules:**
- One phase per session (start fresh to avoid context pollution)
- Large phases (marked ⚠️) should be split into sub-sessions (e.g., Phase 3a, 3b)
- Never skip acceptance criteria — they are the definition of done
- Every phase ends with: `uv run ruff check . && uv run mypy src/ && uv run pytest`

---

## Progress

- [ ] Phase 0 — Project Bootstrap
- [ ] Phase 1 — Core Abstractions
- [ ] Phase 2 — Dapr Integration Layer
- [ ] Phase 3 — Memory: Working & Episodic ⚠️
- [ ] Phase 4 — Memory: Semantic & Procedural ⚠️
- [ ] Phase 5 — Memory Security
- [ ] Phase 6 — Tool System & Sandbox ⚠️
- [ ] Phase 7 — Orchestration Engine ⚠️
- [ ] Phase 8 — Safety & Guardrails
- [ ] Phase 9 — Observability
- [ ] Phase 10 — Evaluation Framework
- [ ] Phase 11 — CLI & Developer Experience
- [ ] Phase 12 — Integration Tests, Docs, Release ⚠️

## Dependency Graph

```
Phase 0
  └► Phase 1
       ├► Phase 2
       │    ├► Phase 3 (working + episodic memory)
       │    │    └► Phase 4 (semantic + procedural memory)
       │    │         └► Phase 5 (memory security)
       │    ├► Phase 6 (tools + sandbox)
       │    └► Phase 7 (orchestration) ← needs Phase 3 + 6
       │         └► Phase 8 (safety)
       ├► Phase 9 (observability) — can start after Phase 2
       └► Phase 10 (evaluation) — can start after Phase 5
  Phase 11 (CLI) — after Phase 7
  Phase 12 (integration + release) — after ALL others
```

---

## Phase 0: Project Bootstrap

**Goal:** Runnable project skeleton with tooling, CI, and local infrastructure.

**Tasks:**
1. Initialize with `uv init` — configure `pyproject.toml` with all deps, ruff, mypy, pytest settings
2. Create directory structure matching the 9 architecture layers (core, dapr, memory, tools, orchestration, safety, observability, evaluation, cli) under `src/nexus/` with `__init__.py` files
3. Create matching `tests/` structure with `conftest.py`
4. Create `docker-compose.yml` with PostgreSQL (pgvector/pgvector:pg16), Redis 7, and Dapr placement service
5. Create `scripts/init-db.sql` enabling pgvector extension
6. Create Dapr component YAML files in `dapr/components/` for: statestore-postgres, statestore-redis (cache), pubsub-redis
7. Create Dapr config YAML with tracing enabled
8. Create `.github/workflows/ci.yml` with lint, typecheck, test, and integration-test jobs
9. Create `Makefile` with: install, lint, format, typecheck, test, test-integration, dev, clean targets

**Key Dependencies (pyproject.toml):**
```
Runtime: pydantic>=2.9, pydantic-settings>=2.5, httpx>=0.27, structlog>=24.4,
         click>=8.1, tiktoken>=0.7, opentelemetry-api>=1.27, opentelemetry-sdk>=1.27,
         opentelemetry-exporter-otlp>=1.27, dapr>=1.14, dapr-ext-grpc>=1.14
Optional: anthropic>=0.39, openai>=1.50
Dev: pytest>=8.3, pytest-asyncio>=0.24, pytest-cov>=5, hypothesis>=6.112,
     testcontainers>=4, ruff>=0.7, mypy>=1.12, mkdocs-material>=9
```

**Acceptance Criteria:**
- [ ] `uv sync` — no errors
- [ ] `uv run pytest` — 0 tests, 0 errors
- [ ] `uv run ruff check .` — passes
- [ ] `uv run mypy src/` — passes
- [ ] `docker compose up -d && docker compose ps` — all healthy
- [ ] Directory structure has all 9 layer packages under `src/nexus/`

---

## Phase 1: Core Abstractions & Configuration

**Goal:** Foundational types, config, errors, logging, and LLM clients everything else builds on.

**Tasks:**
1. `src/nexus/core/config.py` — `NexusConfig` using pydantic-settings with nested sub-configs: `ModelConfig`, `MemoryConfig`, `SafetyConfig`, `DaprConfig`, `ObservabilityConfig`. Load from env vars (NEXUS_ prefix), YAML file, or code. Include sensible defaults for local dev.
2. `src/nexus/core/errors.py` — Exception hierarchy rooted at `NexusError(message, code, details)`. Subclasses: `ConfigError`, `MemoryError`, `MemorySecurityError`, `ToolError`, `ToolTimeoutError`, `OrchestrationError`, `BudgetExceededError`, `SafetyError`, `ModelError`. All carry machine-readable `code` string.
3. `src/nexus/core/logging.py` — structlog setup with JSON (prod) or console (dev) output, ISO 8601 timestamps, correlation ID via contextvars. Export `get_logger(name)` function.
4. `src/nexus/core/types.py` — Pydantic v2 models:
   - `Role` enum (system, user, assistant, tool)
   - `AgentStatus` enum (idle, running, waiting_for_human, completed, failed)
   - `ToolCall(id, name, arguments)`, `ToolResult(tool_call_id, output, error, duration_ms)`
   - `Message(role, content, tool_calls, tool_results, metadata, timestamp)`
   - `ToolParameter(name, type, description, required, default, enum)`
   - `ToolDefinition(name, description, parameters, version)` with `to_function_schema()` method
   - `AgentDefinition(name, model, system_prompt, tools, max_iterations, temperature, memory_enabled, cost_budget_usd)`
   - `TokenUsage(input_tokens, output_tokens, total_tokens, cost_usd, model)`
   - `AgentState(agent_id, session_id, messages, status, current_step, total_token_usage, metadata, timestamps)`
   - `ExecutionResult(output, messages, tool_calls_made, token_usage, duration_seconds, steps_taken, status)`
5. `src/nexus/core/models/base.py` — `ModelClient` ABC with `async complete()` and `async stream()` methods. `ModelResponse(content, tool_calls, token_usage, model, stop_reason)` return type.
6. `src/nexus/core/models/anthropic.py` — Implement `ModelClient` using anthropic SDK. Convert Message list to Anthropic format, ToolDefinition to Anthropic tools format, extract TokenUsage from response.
7. `src/nexus/core/models/openai.py` — Same for OpenAI SDK.
8. Update `src/nexus/core/__init__.py` — export all public types.
9. Write tests for all of the above.

**Acceptance Criteria:**
- [ ] Config loads from env vars and YAML, SecretStr masks API keys
- [ ] All Pydantic models round-trip serialize (JSON → model → JSON matches)
- [ ] `ToolDefinition.to_function_schema()` produces valid JSON schema
- [ ] Error hierarchy: `isinstance(ToolTimeoutError(...), NexusError)` is True
- [ ] Structured logging outputs valid JSON with correlation IDs
- [ ] Model clients handle API errors and wrap in `ModelError`
- [ ] `uv run pytest tests/core/ -v` — all pass (15+ tests)
- [ ] `uv run mypy src/nexus/core/` — no errors

---

## Phase 2: Dapr Integration Layer

**Goal:** Typed Python wrapper around Dapr building blocks that all other layers use.

**Tasks:**
1. `src/nexus/dapr/client.py` — `DaprClient` wrapper providing typed access to state, pub/sub, service invocation, workflows, distributed lock, and secrets APIs. Uses httpx for HTTP calls to sidecar.
2. `src/nexus/dapr/state.py` — `DaprStateStore` class with: namespace-scoped keys (`{namespace}:{entity}:{id}`), automatic Pydantic serialization, optimistic concurrency via ETags, bulk get/set/delete, transaction support.
3. `src/nexus/dapr/pubsub.py` — `DaprPubSub` with: typed publish/subscribe, handler registration via decorator, dead letter queue config, message deduplication.
4. `src/nexus/dapr/workflow.py` — `DaprWorkflow` base class for defining durable workflows with checkpointing.
5. `src/nexus/dapr/lock.py` — Distributed lock as async context manager: `async with dapr_lock("resource-id", timeout=30): ...`
6. `src/nexus/dapr/health.py` — Health check that verifies sidecar availability.
7. `src/nexus/dapr/serialization.py` — Helpers for Pydantic ↔ Dapr JSON conversion.
8. Write tests. Unit tests with mocked HTTP. Integration tests (marked `@pytest.mark.integration`) against real Dapr sidecar using testcontainers.

**Acceptance Criteria:**
- [ ] State CRUD works against Dapr sidecar + PostgreSQL (integration test)
- [ ] Pub/sub publish and subscribe work (integration test)
- [ ] Optimistic concurrency rejects stale writes
- [ ] Namespace scoping prevents key collisions between components
- [ ] Health check returns correct status
- [ ] All operations have structlog logging
- [ ] `uv run pytest tests/dapr/ -v` — all pass

---

## Phase 3: Memory — Working & Episodic ⚠️

**Goal:** In-session working memory with auto-summarization and cross-session episodic memory.

**Split into two sub-sessions if context gets heavy.**

**Tasks (3a — Working Memory):**
1. `src/nexus/memory/token_counter.py` — Model-aware token counting using tiktoken. Support Claude, GPT, and fallback estimation.
2. `src/nexus/memory/summarizer.py` — `Summarizer` class that compresses conversation history. Configurable strategies: `truncate` (drop oldest), `summarize` (LLM-generated summary), `hybrid` (summarize old + keep recent full). Uses a ModelClient call.
3. `src/nexus/memory/working.py` — `WorkingMemory` class: holds messages, tracks token count, auto-summarizes at configurable threshold (default 80% of context window). Stores full uncompressed history to Dapr state for audit. Preserves N most recent messages at full fidelity.
4. Property-based tests with Hypothesis for edge cases (empty conversations, single message, exactly at threshold).

**Tasks (3b — Episodic Memory):**
1. `src/nexus/memory/types.py` — `EpisodicRecord(id, agent_id, user_id, session_id, timestamp, content, metadata, trust_score, provenance, embedding, importance_score, access_count, last_accessed)`
2. `src/nexus/memory/embeddings.py` — `EmbeddingService` wrapping OpenAI/Anthropic embedding APIs. Caches embeddings in Redis via Dapr cache store.
3. `src/nexus/memory/episodic.py` — `EpisodicMemory`: CRUD backed by Dapr state (PostgreSQL). Store with embedding vector. Temporal decay (configurable rate). Importance scoring (0-1). Access count tracking.
4. `src/nexus/memory/retriever.py` — `EpisodicRetriever` with hybrid search: `score = α×recency + β×similarity + γ×importance`. Configurable weights. Returns top-K results.
5. Write tests with realistic multi-turn conversation scenarios.

**Acceptance Criteria:**
- [ ] Working memory auto-summarizes at token limit
- [ ] Summarization preserves key facts (assertion checks on summary content)
- [ ] Full history persists to Dapr state even after summarization
- [ ] Episodic records persist across sessions
- [ ] Temporal decay reduces old record scores
- [ ] Semantic search returns relevant records
- [ ] Hybrid retrieval correctly blends recency + similarity + importance
- [ ] `uv run pytest tests/memory/ -v -k "working or episodic"` — all pass

---

## Phase 4: Memory — Semantic & Procedural ⚠️

**Goal:** Knowledge extraction from episodes (facts) and learned workflow storage.

**Tasks (4a — Semantic Memory):**
1. `src/nexus/memory/types.py` — Add `SemanticFact(id, subject, predicate, object, confidence, source_episode_ids, created_at, updated_at, access_count, embedding)`
2. `src/nexus/memory/semantic.py` — `SemanticMemory`: stores facts, deduplicates against existing, handles conflicts (confidence-weighted replacement), CRUD via Dapr state + pgvector.
3. `src/nexus/memory/consolidation.py` — `ConsolidationPipeline`: async background job that scans recent episodic records, uses LLM to extract facts, updates semantic memory, prunes redundant episodes. Runs on configurable interval. Uses Dapr Jobs API or asyncio background task.
4. `src/nexus/memory/semantic_retriever.py` — Fact retrieval by subject, predicate, similarity, or free-text query.

**Tasks (4b — Procedural Memory):**
1. `src/nexus/memory/types.py` — Add `Procedure(id, name, description, steps: list[ProcedureStep], trigger_conditions, success_count, failure_count, last_used, agent_id)` and `ProcedureStep(action, tool_name, parameters_template, expected_outcome)`
2. `src/nexus/memory/procedural.py` — `ProceduralMemory`: store/retrieve learned workflows via Dapr state.
3. `src/nexus/memory/procedure_extractor.py` — Post-execution hook: analyzes completed tool call sequences, uses LLM to generalize into reusable procedure templates.
4. `src/nexus/memory/procedure_matcher.py` — Given a task description, find relevant stored procedures via semantic matching.

**Tasks (4c — Unified Memory Interface):**
1. `src/nexus/memory/manager.py` — `MemoryManager` that provides a single interface to all four memory types. Methods: `remember(content)`, `recall(query, memory_types)`, `forget(record_id)`, `consolidate()`. This is what the orchestration layer talks to.

**Acceptance Criteria:**
- [ ] Consolidation extracts facts from episodic records
- [ ] Duplicate facts detected and merged
- [ ] Conflicting facts flagged with confidence scores
- [ ] Procedures extracted from completed multi-step executions
- [ ] Procedure matcher finds relevant procedures for new tasks
- [ ] `MemoryManager` provides unified access to all four types
- [ ] Consolidation runs async (doesn't block agent execution)
- [ ] `uv run pytest tests/memory/ -v -k "semantic or procedural or manager"` — all pass

---

## Phase 5: Memory Security

**Goal:** Provenance, trust scoring, validation, and poisoning defense on all memory writes.

**Tasks:**
1. `src/nexus/memory/provenance.py` — `ProvenanceTracker`: annotates every memory write with `Provenance(source_type: SourceType, source_id, trust_level, timestamp, content_hash_sha256)`. `SourceType` enum: `USER_INPUT` (0.9), `SYSTEM` (1.0), `LLM_GENERATED` (0.7), `TOOL_RESULT` (0.6), `EXTERNAL_DATA` (0.3).
2. `src/nexus/memory/trust.py` — `TrustScorer`: assigns/updates trust scores based on provenance. Temporal decay on trust. Access count boost for frequently validated memories.
3. `src/nexus/memory/validator.py` — `MemoryValidator`: pre-write pipeline. Checks: instruction detection (regex + heuristic for "remember that", "always", "in future conversations"), content sanitization, size anomaly detection, rate limiting per source.
4. `src/nexus/memory/auditor.py` — `MemoryAuditor`: periodic scan verifying content hash integrity, flagging broken provenance chains, generating compliance reports.
5. Integrate into `MemoryManager` write path — all writes go through validator + provenance before hitting Dapr state.

**Acceptance Criteria:**
- [ ] Every memory write has provenance metadata
- [ ] Trust scores reflect source type + temporal decay
- [ ] Instruction injection detected and blocked (test with OWASP-style payloads)
- [ ] Content hash integrity verified on read
- [ ] Rate limiting blocks burst writes from untrusted sources
- [ ] Auditor detects tampered entries
- [ ] `uv run pytest tests/memory/ -v -k "security or provenance or trust or validator"` — all pass

---

## Phase 6: Tool System & Sandbox ⚠️

**Goal:** Tool registry, MCP client, and sandboxed execution.

**Tasks (6a — Registry & MCP):**
1. `src/nexus/tools/registry.py` — `ToolRegistry`: register tools with decorator API (`@nexus.tool(name=..., description=...)`), list as JSON schema, version management.
2. `src/nexus/tools/mcp_client.py` — MCP protocol client: discover MCP servers, list tools, invoke tools, tag results with `EXTERNAL_DATA` provenance.
3. `src/nexus/tools/executor.py` — `ToolExecutor`: validate args against schema, execute with timeout + retry, record trace (input, output, duration, error), idempotency key support for workflow replay.

**Tasks (6b — Sandbox):**
1. `src/nexus/tools/sandbox/manager.py` — `SandboxManager`: Docker container sandbox by default. Config: network access, filesystem mounts, memory/CPU limits, execution timeout.
2. `src/nexus/tools/sandbox/docker.py` — Docker container lifecycle: create, execute, capture output, destroy. Container pooling for warm starts.
3. `src/nexus/tools/sandbox/code_executor.py` — `CodeExecutor`: execute LLM-generated Python in sandbox. Inject registered tools as callables into namespace. Capture stdout/stderr/return values.
4. `src/nexus/tools/boundaries.py` — `ActionGuard`: per-agent allowlist/denylist of tools, rate limiting (max N calls/minute), cost guard (max $ per action).

**Acceptance Criteria:**
- [ ] Tools registered via decorator, invoked by name
- [ ] MCP client discovers tools from test MCP server
- [ ] Timeout and retry policies respected
- [ ] Idempotent calls return cached results on replay
- [ ] Sandbox blocks filesystem access outside designated paths
- [ ] Sandbox blocks network when configured to deny
- [ ] Code execution captures output correctly
- [ ] Action boundaries block unauthorized calls
- [ ] `uv run pytest tests/tools/ -v` — all pass

---

## Phase 7: Orchestration Engine ⚠️

**Goal:** Graph engine, model router, cost tracker, agent execution loop, and multi-agent crews.

**Tasks (7a — Graph Engine):**
1. `src/nexus/orchestration/graph.py` — `Graph`: nodes (async callable handlers), edges (conditional/unconditional), builder pattern API. Executes by walking graph, passing `AgentState`. Parallel execution for independent branches. Checkpoint state to Dapr after each node.
2. `src/nexus/orchestration/nodes.py` — Pre-built nodes: `LLMNode`, `ToolNode`, `ConditionalNode`, `HumanNode` (pause + wait), `SubgraphNode` (nested).

**Tasks (7b — Model Router + Cost):**
1. `src/nexus/orchestration/model_router.py` — `ModelRouter`: route steps to cheapest capable model. Tiers: `fast`, `balanced`, `powerful`. Auto-fallback on failure. Configurable routing rules.
2. `src/nexus/orchestration/cost_tracker.py` — `CostTracker`: token + cost tracking per model/agent/session/step. Budget enforcement (hard limits). Cost events via Dapr pub/sub.

**Tasks (7c — Agent Loop + Crews):**
1. `src/nexus/orchestration/runner.py` — `AgentRunner`: the main loop. Input → load memory → plan → execute tools → update memory → respond. Supports ReAct and Plan-and-Execute patterns. Max iterations guard. Integrates memory (Phase 3-5), tools (Phase 6), graph (7a), model router (7b).
2. `src/nexus/orchestration/crew.py` — `Crew`: multi-agent orchestration. Sequential, parallel, hierarchical patterns. Shared state via Dapr + distributed locks. Supervisor pattern (one agent delegates).

**Acceptance Criteria:**
- [ ] Graph executes multi-node workflow with conditional branching
- [ ] Checkpoint/restore: kill mid-graph, restart, resumes from last checkpoint
- [ ] Model router selects appropriate tier
- [ ] Cost tracker reports accurate usage per step
- [ ] Budget enforcement stops execution at limit
- [ ] AgentRunner completes ReAct loop with tool calling
- [ ] Crew runs 3-agent workflow with shared state
- [ ] `uv run pytest tests/orchestration/ -v` — all pass

---

## Phase 8: Safety & Guardrails

**Goal:** Runtime safety middleware intercepting every agent action.

**Tasks:**
1. `src/nexus/safety/pipeline.py` — `SafetyPipeline`: middleware wrapping LLM calls, tool calls, memory writes. Pre/post execution checks. Configurable via YAML policy files.
2. `src/nexus/safety/injection.py` — `PromptInjectionDetector`: multi-layer (regex, heuristic, semantic classifier). Detect injection in tool results, user input, memory retrieval. Levels: strict, balanced, permissive.
3. `src/nexus/safety/pii.py` — `PIIDetector`: detect email, phone, SSN, credit card, addresses in tool I/O. Actions: log, redact, block. Regex + optional spaCy NER.
4. `src/nexus/safety/action_guard.py` — `ActionGuard`: per-agent boundaries, rate limiting, cost guard. (May reuse/extend from Phase 6 boundaries.py)
5. `src/nexus/safety/policies.py` — YAML policy loader for all safety config.
6. Integrate into `AgentRunner` execution path.

**Acceptance Criteria:**
- [ ] Injection attempts in tool results detected and blocked
- [ ] PII detected and redacted when configured
- [ ] Unauthorized tool calls blocked
- [ ] Rate limiting works
- [ ] <5ms overhead per check (benchmark test)
- [ ] Policies configurable via YAML
- [ ] `uv run pytest tests/safety/ -v` — all pass

---

## Phase 9: Observability

**Goal:** Three-layer observability: infrastructure (Dapr), execution (custom OTEL), behavior (analytics).

**Tasks:**
1. `src/nexus/observability/tracer.py` — `NexusTracer`: wraps OTEL SDK. Custom span types: `agent.run`, `agent.llm_call`, `agent.tool_call`, `agent.memory_read`, `agent.memory_write`, `agent.decision`. Annotates with model, tokens, cost, step, agent_id. Session-level parent spans.
2. `src/nexus/observability/metrics.py` — Prometheus-compatible metrics endpoint. Counters: total_tokens, total_cost, tool_calls, errors. Gauges: active_agents. Histograms: llm_latency, tool_latency.
3. `src/nexus/observability/behavior.py` — `BehaviorMonitor`: tracks per-agent patterns over time. Detects anomalies (tool usage shifts, cost spikes, memory access changes). Consumes events via Dapr pub/sub. Alerts on drift.
4. `src/nexus/observability/events.py` — Structured event log for every agent action. Append-only. Replayable. Audit-ready.

**Acceptance Criteria:**
- [ ] Agent runs produce OTEL traces viewable in Jaeger
- [ ] Spans include model, tokens, cost, duration
- [ ] Prometheus endpoint exposes key metrics
- [ ] Behavior monitor detects injected anomaly
- [ ] Event log captures all actions as immutable events
- [ ] `uv run pytest tests/observability/ -v` — all pass

---

## Phase 10: Evaluation Framework

**Goal:** Built-in testing and quality measurement for agent behaviors.

**Tasks:**
1. `src/nexus/evaluation/suite.py` — `EvalSuite` + `EvalCase(input, expected_output, expected_tool_calls, assertions)`. Run against agent, collect pass/fail with diagnostics.
2. `src/nexus/evaluation/assertions.py` — Assertion types: contains, not_contains, matches_regex, semantic_similarity, json_schema_valid, tool_was_called, tool_not_called.
3. `src/nexus/evaluation/prompt_versions.py` — `PromptVersionManager`: track system prompt versions, diff between versions, A/B test with automated scoring, rollback.
4. `src/nexus/evaluation/baseline.py` — `QualityBaseline`: establish baseline scores, compare on subsequent runs, alert on regression.
5. `src/nexus/evaluation/reporter.py` — Output reports to stdout, JSON, or Dapr pub/sub.

**Acceptance Criteria:**
- [ ] Eval suite runs against test agent, reports pass/fail
- [ ] Prompt versions tracked and diffable
- [ ] Quality baseline detects regression
- [ ] Reports include latency, cost, quality
- [ ] `uv run pytest tests/evaluation/ -v` — all pass

---

## Phase 11: CLI & Developer Experience

**Goal:** CLI that makes Nexus as easy to start as `create-react-app`.

**Tasks:**
1. `src/nexus/cli/main.py` — Click CLI group with subcommands
2. `nexus init` — scaffold project (config, example agent, docker-compose, dapr components)
3. `nexus run <agent.py>` — start agent with Dapr sidecar auto-started in background
4. `nexus eval <suite.py>` — run evaluation suite
5. `nexus deploy` — generate Kubernetes manifests + Helm chart
6. `nexus cost` — show cost summary for recent sessions
7. `nexus memory inspect/clear <agent_id>` — view/clear agent memory
8. `nexus dev` — watch mode: start everything, auto-reload on changes, show live cost/traces
9. Agent definition DSL: YAML for simple agents, Python decorators for code-first
10. Project templates: simple (single agent), crew (multi-agent), rag (document retrieval)

**Acceptance Criteria:**
- [ ] `nexus init` creates runnable project in <10 seconds
- [ ] `nexus run example.py` starts agent with Dapr, responds to input
- [ ] `nexus eval` prints pass/fail report
- [ ] YAML agent definition works
- [ ] Decorator API works
- [ ] `nexus dev` starts everything with auto-reload
- [ ] All commands have `--help`

---

## Phase 12: Integration Tests, Docs, Release ⚠️

**Goal:** End-to-end validation, documentation, and PyPI-ready package.

**Tasks (12a — Integration Tests):**
1. E2E: single agent + tools completes multi-step task
2. E2E: 3-agent crew with shared memory
3. E2E: memory persists across sessions
4. E2E: safety pipeline blocks injection in tool results
5. E2E: cost budget stops agent at limit
6. E2E: checkpoint/restart after simulated crash
7. Performance benchmark: latency overhead vs raw LLM call
8. All run against real Dapr (testcontainers)

**Tasks (12b — Documentation):**
1. Getting Started (5-minute quickstart)
2. Concepts (architecture, memory, orchestration, safety)
3. Tutorials: research agent, support crew, RAG agent
4. API reference (auto-generated from docstrings via mkdocstrings)
5. Configuration reference
6. Deployment guide (local, Docker, Kubernetes)
7. Contributing guide

**Tasks (12c — Release):**
1. Finalize pyproject.toml for PyPI
2. CHANGELOG.md
3. GitHub release workflow (tag → build → PyPI publish)
4. Docker image for nexus runtime
5. Helm chart for Kubernetes
6. README.md with badges, quickstart, architecture diagram

**Acceptance Criteria:**
- [ ] All integration tests pass
- [ ] Docs site builds without errors
- [ ] `pip install nexus-ai` works from built wheel
- [ ] Docker image runs example agent
- [ ] Helm deploys to Kind cluster
- [ ] Test coverage > 80%

---

## Definition of Done (Every Phase)

Before marking ANY phase complete:
1. ✅ All acceptance criteria checked
2. ✅ `uv run ruff check .` — clean
3. ✅ `uv run ruff format --check .` — clean
4. ✅ `uv run mypy src/` — clean
5. ✅ `uv run pytest` — all pass
6. ✅ All public APIs have type hints + docstrings
7. ✅ No TODOs (unless explicitly for future phase)
8. ✅ Committed: `feat(module): phase N description`
