# PLAN.md — Grampus Implementation Plan

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
       │    └── Phase 7 (orchestration) ← needs Phase 3 + 6
       │         └► Phase 8 (safety)
       ├► Phase 9 (observability) — can start after Phase 2
       └► Phase 10 (evaluation) — can start after Phase 5
  Phase 11 (CLI) — after Phase 7
  Phase 12 (integration + release) — after ALL others
```

---

## Web UI Strategy (Post-Launch Phases)

All post-launch phases that expose a visual interface (D9 Memory Inspector, D10 Eval Dashboard, and any future UI pages) **must build into a single consolidated web app** rather than separate apps or standalone endpoints.

**Technology stack:** HTMX (loaded from CDN — no build step, no Node.js) + Jinja2 templates + FastAPI. Served at `/ui/` from the existing Grampus server.

**Dependency:** Add `jinja2>=3.0` to the `server` optional dep group in `pyproject.toml`. No other new deps.

**Structure:**
- D9 builds the **shell**: `src/grampus/server/ui/` package, base Jinja2 template with sidebar nav, layout, and HTMX wiring. Every subsequent UI phase adds pages to this shell.
- Pages follow the pattern `/ui/<feature>/` with HTMX partial endpoints at `/ui/<feature>/_<partial>` for dynamic updates.
- Static assets (CSS, any minimal JS) live in `src/grampus/server/ui/static/`.
- FastAPI mounts the UI router at `/ui` via `app.include_router(ui_router, prefix="/ui")`.

**Planned pages:**
| Path | Phase | Description |
|---|---|---|
| `/ui/` | D9 | Dashboard: active agents, cost today, recent errors |
| `/ui/memory/` | D9 | Memory inspector: browse, search, filter, delete entries |
| `/ui/evals/` | D10 | Eval suite history, pass rates, regression trends |
| `/ui/cost/` | D10 | Cost analytics by model/agent/session/time |
| `/ui/alerts/` | D7+ | Alert rule management (supplements REST API) |
| `/ui/traces/` | future | Execution trace viewer |

**Exception:** The Visual Agent Builder (Phase C13 in roadmap) requires drag-and-drop graph editing which HTMX cannot support. That phase uses a minimal React SPA bundled separately under `src/grampus/server/ui/static/builder/` and served at `/ui/builder/`. It is the only phase permitted to introduce a frontend build step.

---

## Phase 0: Project Bootstrap

**Goal:** Runnable project skeleton with tooling, CI, and local infrastructure.

**Tasks:**
1. Initialize with `uv init` — configure `pyproject.toml` with all deps, ruff, mypy, pytest settings
2. Create directory structure matching the 9 architecture layers (core, dapr, memory, tools, orchestration, safety, observability, evaluation, cli) under `src/grampus/` with `__init__.py` files
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
- [ ] Directory structure has all 9 layer packages under `src/grampus/`

---

## Phase 1: Core Abstractions & Configuration

**Goal:** Foundational types, config, errors, logging, and LLM clients everything else builds on.

**Tasks:**
1. `src/grampus/core/config.py` — `GrampusConfig` using pydantic-settings with nested sub-configs: `ModelConfig`, `MemoryConfig`, `SafetyConfig`, `DaprConfig`, `ObservabilityConfig`. Load from env vars (GRAMPUS_ prefix), YAML file, or code. Include sensible defaults for local dev.
2. `src/grampus/core/errors.py` — Exception hierarchy rooted at `GrampusError(message, code, details)`. Subclasses: `ConfigError`, `MemoryError`, `MemorySecurityError`, `ToolError`, `ToolTimeoutError`, `OrchestrationError`, `BudgetExceededError`, `SafetyError`, `ModelError`. All carry machine-readable `code` string.
3. `src/grampus/core/logging.py` — structlog setup with JSON (prod) or console (dev) output, ISO 8601 timestamps, correlation ID via contextvars. Export `get_logger(name)` function.
4. `src/grampus/core/types.py` — Pydantic v2 models:
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
5. `src/grampus/core/models/base.py` — `ModelClient` ABC with `async complete()` and `async stream()` methods. `ModelResponse(content, tool_calls, token_usage, model, stop_reason)` return type.
6. `src/grampus/core/models/anthropic.py` — Implement `ModelClient` using anthropic SDK. Convert Message list to Anthropic format, ToolDefinition to Anthropic tools format, extract TokenUsage from response.
7. `src/grampus/core/models/openai.py` — Same for OpenAI SDK.
8. Update `src/grampus/core/__init__.py` — export all public types.
9. Write tests for all of the above.

**Acceptance Criteria:**
- [ ] Config loads from env vars and YAML, SecretStr masks API keys
- [ ] All Pydantic models round-trip serialize (JSON → model → JSON matches)
- [ ] `ToolDefinition.to_function_schema()` produces valid JSON schema
- [ ] Error hierarchy: `isinstance(ToolTimeoutError(...), GrampusError)` is True
- [ ] Structured logging outputs valid JSON with correlation IDs
- [ ] Model clients handle API errors and wrap in `ModelError`
- [ ] `uv run pytest tests/core/ -v` — all pass (15+ tests)
- [ ] `uv run mypy src/grampus/core/` — no errors

---

## Phase 2: Dapr Integration Layer

**Goal:** Typed Python wrapper around Dapr building blocks that all other layers use.

**Tasks:**
1. `src/grampus/dapr/client.py` — `DaprClient` wrapper providing typed access to state, pub/sub, service invocation, workflows, distributed lock, and secrets APIs. Uses httpx for HTTP calls to sidecar.
2. `src/grampus/dapr/state.py` — `DaprStateStore` class with: namespace-scoped keys (`{namespace}:{entity}:{id}`), automatic Pydantic serialization, optimistic concurrency via ETags, bulk get/set/delete, transaction support.
3. `src/grampus/dapr/pubsub.py` — `DaprPubSub` with: typed publish/subscribe, handler registration via decorator, dead letter queue config, message deduplication.
4. `src/grampus/dapr/workflow.py` — `DaprWorkflow` base class for defining durable workflows with checkpointing.
5. `src/grampus/dapr/lock.py` — Distributed lock as async context manager: `async with dapr_lock("resource-id", timeout=30): ...`
6. `src/grampus/dapr/health.py` — Health check that verifies sidecar availability.
7. `src/grampus/dapr/serialization.py` — Helpers for Pydantic ↔ Dapr JSON conversion.
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
1. `src/grampus/memory/token_counter.py` — Model-aware token counting using tiktoken. Support Claude, GPT, and fallback estimation.
2. `src/grampus/memory/summarizer.py` — `Summarizer` class that compresses conversation history. Configurable strategies: `truncate` (drop oldest), `summarize` (LLM-generated summary), `hybrid` (summarize old + keep recent full). Uses a ModelClient call.
3. `src/grampus/memory/working.py` — `WorkingMemory` class: holds messages, tracks token count, auto-summarizes at configurable threshold (default 80% of context window). Stores full uncompressed history to Dapr state for audit. Preserves N most recent messages at full fidelity.
4. Property-based tests with Hypothesis for edge cases (empty conversations, single message, exactly at threshold).

**Tasks (3b — Episodic Memory):**
1. `src/grampus/memory/types.py` — `EpisodicRecord(id, agent_id, user_id, session_id, timestamp, content, metadata, trust_score, provenance, embedding, importance_score, access_count, last_accessed)`
2. `src/grampus/memory/embeddings.py` — `EmbeddingService` wrapping OpenAI/Anthropic embedding APIs. Caches embeddings in Redis via Dapr cache store.
3. `src/grampus/memory/episodic.py` — `EpisodicMemory`: CRUD backed by Dapr state (PostgreSQL). Store with embedding vector. Temporal decay (configurable rate). Importance scoring (0-1). Access count tracking.
4. `src/grampus/memory/retriever.py` — `EpisodicRetriever` with hybrid search: `score = α×recency + β×similarity + γ×importance`. Configurable weights. Returns top-K results.
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
1. `src/grampus/memory/types.py` — Add `SemanticFact(id, subject, predicate, object, confidence, source_episode_ids, created_at, updated_at, access_count, embedding)`
2. `src/grampus/memory/semantic.py` — `SemanticMemory`: stores facts, deduplicates against existing, handles conflicts (confidence-weighted replacement), CRUD via Dapr state + pgvector.
3. `src/grampus/memory/consolidation.py` — `ConsolidationPipeline`: async background job that scans recent episodic records, uses LLM to extract facts, updates semantic memory, prunes redundant episodes. Runs on configurable interval. Uses Dapr Jobs API or asyncio background task.
4. `src/grampus/memory/semantic_retriever.py` — Fact retrieval by subject, predicate, similarity, or free-text query.

**Tasks (4b — Procedural Memory):**
1. `src/grampus/memory/types.py` — Add `Procedure(id, name, description, steps: list[ProcedureStep], trigger_conditions, success_count, failure_count, last_used, agent_id)` and `ProcedureStep(action, tool_name, parameters_template, expected_outcome)`
2. `src/grampus/memory/procedural.py` — `ProceduralMemory`: store/retrieve learned workflows via Dapr state.
3. `src/grampus/memory/procedure_extractor.py` — Post-execution hook: analyzes completed tool call sequences, uses LLM to generalize into reusable procedure templates.
4. `src/grampus/memory/procedure_matcher.py` — Given a task description, find relevant stored procedures via semantic matching.

**Tasks (4c — Unified Memory Interface):**
1. `src/grampus/memory/manager.py` — `MemoryManager` that provides a single interface to all four memory types. Methods: `remember(content)`, `recall(query, memory_types)`, `forget(record_id)`, `consolidate()`. This is what the orchestration layer talks to.

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
1. `src/grampus/memory/provenance.py` — `ProvenanceTracker`: annotates every memory write with `Provenance(source_type: SourceType, source_id, trust_level, timestamp, content_hash_sha256)`. `SourceType` enum: `USER_INPUT` (0.9), `SYSTEM` (1.0), `LLM_GENERATED` (0.7), `TOOL_RESULT` (0.6), `EXTERNAL_DATA` (0.3).
2. `src/grampus/memory/trust.py` — `TrustScorer`: assigns/updates trust scores based on provenance. Temporal decay on trust. Access count boost for frequently validated memories.
3. `src/grampus/memory/validator.py` — `MemoryValidator`: pre-write pipeline. Checks: instruction detection (regex + heuristic for "remember that", "always", "in future conversations"), content sanitization, size anomaly detection, rate limiting per source.
4. `src/grampus/memory/auditor.py` — `MemoryAuditor`: periodic scan verifying content hash integrity, flagging broken provenance chains, generating compliance reports.
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
1. `src/grampus/tools/registry.py` — `ToolRegistry`: register tools with decorator API (`@nexus.tool(name=..., description=...)`), list as JSON schema, version management.
2. `src/grampus/tools/mcp_client.py` — MCP protocol client: discover MCP servers, list tools, invoke tools, tag results with `EXTERNAL_DATA` provenance.
3. `src/grampus/tools/executor.py` — `ToolExecutor`: validate args against schema, execute with timeout + retry, record trace (input, output, duration, error), idempotency key support for workflow replay.

**Tasks (6b — Sandbox):**
1. `src/grampus/tools/sandbox/manager.py` — `SandboxManager`: Docker container sandbox by default. Config: network access, filesystem mounts, memory/CPU limits, execution timeout.
2. `src/grampus/tools/sandbox/docker.py` — Docker container lifecycle: create, execute, capture output, destroy. Container pooling for warm starts.
3. `src/grampus/tools/sandbox/code_executor.py` — `CodeExecutor`: execute LLM-generated Python in sandbox. Inject registered tools as callables into namespace. Capture stdout/stderr/return values.
4. `src/grampus/tools/boundaries.py` — `ActionGuard`: per-agent allowlist/denylist of tools, rate limiting (max N calls/minute), cost guard (max $ per action).

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
1. `src/grampus/orchestration/graph.py` — `Graph`: nodes (async callable handlers), edges (conditional/unconditional), builder pattern API. Executes by walking graph, passing `AgentState`. Parallel execution for independent branches. Checkpoint state to Dapr after each node.
2. `src/grampus/orchestration/nodes.py` — Pre-built nodes: `LLMNode`, `ToolNode`, `ConditionalNode`, `HumanNode` (pause + wait), `SubgraphNode` (nested).

**Tasks (7b — Model Router + Cost):**
1. `src/grampus/orchestration/model_router.py` — `ModelRouter`: route steps to cheapest capable model. Tiers: `fast`, `balanced`, `powerful`. Auto-fallback on failure. Configurable routing rules.
2. `src/grampus/orchestration/cost_tracker.py` — `CostTracker`: token + cost tracking per model/agent/session/step. Budget enforcement (hard limits). Cost events via Dapr pub/sub.

**Tasks (7c — Agent Loop + Crews):**
1. `src/grampus/orchestration/runner.py` — `AgentRunner`: the main loop. Input → load memory → plan → execute tools → update memory → respond. Supports ReAct and Plan-and-Execute patterns. Max iterations guard. Integrates memory (Phase 3-5), tools (Phase 6), graph (7a), model router (7b).
2. `src/grampus/orchestration/crew.py` — `Crew`: multi-agent orchestration. Sequential, parallel, hierarchical patterns. Shared state via Dapr + distributed locks. Supervisor pattern (one agent delegates).

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
1. `src/grampus/safety/pipeline.py` — `SafetyPipeline`: middleware wrapping LLM calls, tool calls, memory writes. Pre/post execution checks. Configurable via YAML policy files.
2. `src/grampus/safety/injection.py` — `PromptInjectionDetector`: multi-layer (regex, heuristic, semantic classifier). Detect injection in tool results, user input, memory retrieval. Levels: strict, balanced, permissive.
3. `src/grampus/safety/pii.py` — `PIIDetector`: detect email, phone, SSN, credit card, addresses in tool I/O. Actions: log, redact, block. Regex + optional spaCy NER.
4. `src/grampus/safety/action_guard.py` — `ActionGuard`: per-agent boundaries, rate limiting, cost guard. (May reuse/extend from Phase 6 boundaries.py)
5. `src/grampus/safety/policies.py` — YAML policy loader for all safety config.
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
1. `src/grampus/observability/tracer.py` — `GrampusTracer`: wraps OTEL SDK. Custom span types: `agent.run`, `agent.llm_call`, `agent.tool_call`, `agent.memory_read`, `agent.memory_write`, `agent.decision`. Annotates with model, tokens, cost, step, agent_id. Session-level parent spans.
2. `src/grampus/observability/metrics.py` — Prometheus-compatible metrics endpoint. Counters: total_tokens, total_cost, tool_calls, errors. Gauges: active_agents. Histograms: llm_latency, tool_latency.
3. `src/grampus/observability/behavior.py` — `BehaviorMonitor`: tracks per-agent patterns over time. Detects anomalies (tool usage shifts, cost spikes, memory access changes). Consumes events via Dapr pub/sub. Alerts on drift.
4. `src/grampus/observability/events.py` — Structured event log for every agent action. Append-only. Replayable. Audit-ready.

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
1. `src/grampus/evaluation/suite.py` — `EvalSuite` + `EvalCase(input, expected_output, expected_tool_calls, assertions)`. Run against agent, collect pass/fail with diagnostics.
2. `src/grampus/evaluation/assertions.py` — Assertion types: contains, not_contains, matches_regex, semantic_similarity, json_schema_valid, tool_was_called, tool_not_called.
3. `src/grampus/evaluation/prompt_versions.py` — `PromptVersionManager`: track system prompt versions, diff between versions, A/B test with automated scoring, rollback.
4. `src/grampus/evaluation/baseline.py` — `QualityBaseline`: establish baseline scores, compare on subsequent runs, alert on regression.
5. `src/grampus/evaluation/reporter.py` — Output reports to stdout, JSON, or Dapr pub/sub.

**Acceptance Criteria:**
- [ ] Eval suite runs against test agent, reports pass/fail
- [ ] Prompt versions tracked and diffable
- [ ] Quality baseline detects regression
- [ ] Reports include latency, cost, quality
- [ ] `uv run pytest tests/evaluation/ -v` — all pass

---

## Phase 11: CLI & Developer Experience

**Goal:** CLI that makes Grampus as easy to start as `create-react-app`.

**Tasks:**
1. `src/grampus/cli/main.py` — Click CLI group with subcommands
2. `grampus init` — scaffold project (config, example agent, docker-compose, dapr components)
3. `grampus run <agent.py>` — start agent with Dapr sidecar auto-started in background
4. `grampus eval <suite.py>` — run evaluation suite
5. `grampus deploy` — generate Kubernetes manifests + Helm chart
6. `grampus cost` — show cost summary for recent sessions
7. `grampus memory clear <agent_id>` — clear agent memory from CLI; visual inspection is handled by the web UI at `/ui/memory/` (see Web UI Strategy above)
8. `grampus dev` — watch mode: start everything, auto-reload on changes, show live cost/traces
9. Agent definition DSL: YAML for simple agents, Python decorators for code-first
10. Project templates: simple (single agent), crew (multi-agent), rag (document retrieval)

**Acceptance Criteria:**
- [ ] `grampus init` creates runnable project in <10 seconds
- [ ] `grampus run example.py` starts agent with Dapr, responds to input
- [ ] `grampus eval` prints pass/fail report
- [ ] YAML agent definition works
- [ ] Decorator API works
- [ ] `grampus dev` starts everything with auto-reload
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
4. Docker image for grampus runtime
5. Helm chart for Kubernetes
6. README.md with badges, quickstart, architecture diagram

**Acceptance Criteria:**
- [ ] All integration tests pass
- [ ] Docs site builds without errors
- [ ] `pip install grampus-ai` works from built wheel
- [ ] Docker image runs example agent
- [ ] Helm deploys to Kind cluster
- [ ] Test coverage > 80%

---

## Phase H52: Comprehensive Integration Test Suite ⚠️

**Goal:** Close the gap that mocked unit tests cannot cover. Two complementary suites: Suite A
tests the real infrastructure stack (PostgreSQL/pgvector, Redis, Dapr) using testcontainers —
zero API cost, runs on every PR to main. Suite B tests real LLM APIs (Anthropic, OpenAI) —
gated by environment variable, runs nightly and on manual dispatch. Together they cover all
eight framework layers.

**Why mocked tests are insufficient for this framework:**
- Dapr ETag concurrency under real latency cannot be simulated
- pgvector HNSW index behavior with real embeddings differs from fake float lists
- Streaming chunk formats split mid-JSON in ways no mock replicates
- Safety pipeline behavior against real model outputs has edge cases mocks never produce
- Token counting from the API response differs 5–15% from tiktoken estimates
- Back-to-back tool calls in a single response require accumulation logic mocks always get right

---

### Suite A: Infrastructure Integration Tests

**Gate:** `pytest -m integration` (or `RUN_INTEGRATION_TESTS=true` in CI)
**Trigger:** Automatic on every PR to `main` branch
**Cost:** Zero (testcontainers only)
**Location:** `tests/integration/infra/`

**New dev dependencies required:**
Add to `pyproject.toml` dev deps if not already present:
- `pytest-rerunfailures>=13` — retry on container startup flakiness
- `testcontainers[postgres,redis]>=4` — already present; verify postgres+redis extras

**Tasks (Suite A):**

1. `tests/integration/infra/conftest.py` — Shared fixtures:
   - `pg_container` (session-scoped): start `pgvector/pgvector:pg16` via testcontainers,
     yield `db_url: str`. Enable pgvector extension on startup.
   - `redis_container` (session-scoped): start `redis:7`, yield `redis_url: str`.
   - `asyncpg_pool` (function-scoped): connect via `asyncpg.create_pool(db_url)`, yield, close.
   - Mark all tests in this directory `@pytest.mark.integration` via `conftest.py` `pytestmark`.

2. `tests/integration/infra/test_rag_store.py` — Full RAG store lifecycle:
   - `test_setup_creates_schema` — `RAGStore.create()` creates table + HNSW index
   - `test_upsert_and_retrieve_returns_results` — ingest 5 chunks with real embeddings
     (use small fixed vectors, not API calls), hybrid search returns top match
   - `test_hybrid_search_fts_component` — insert chunks with keyword "dapr",
     query text "dapr", verify FTS component contributes to RRF score
   - `test_dimension_mismatch_raises` — create table with dim=3, try setup with dim=5,
     verify `RAGError(code="DIMENSION_MISMATCH")` raised
   - `test_delete_document_removes_chunks` — upsert 3 chunks, delete by document_id,
     verify count drops to 0
   - `test_upsert_is_idempotent` — upsert same chunk twice, verify count stays at 1
   - `test_namespace_isolation` — insert chunks in namespace "a" and "b",
     query namespace "a" only returns namespace "a" chunks
   - `test_get_stats` — verify chunk_count and document_count correct after inserts

3. `tests/integration/infra/test_dapr_state.py` — Dapr state store with real PostgreSQL:
   - `test_save_and_get_roundtrip` — save Pydantic model, retrieve, assert equal
   - `test_etag_concurrency_rejects_stale_write` — save, retrieve etag, save from
     another "process" (different client instance), verify optimistic lock failure
   - `test_namespace_scoping_isolates_keys` — two namespaces, same key, different values
   - `test_bulk_save_and_get` — 10 records, bulk save, bulk get, all match
   - `test_delete_removes_record` — save, delete, get returns None
   - Note: requires Dapr sidecar OR mock the Dapr HTTP API with `respx`. Use `respx`
     to mock the sidecar HTTP calls since testcontainers cannot run the Dapr binary.
     The goal is testing our serialization/namespacing logic, not Dapr itself.

4. `tests/integration/infra/test_embedding_cache.py` — Embedding cache with real Redis:
   - `test_cache_miss_then_hit` — first call to `EmbeddingService.embed()` stores in Redis;
     second call returns from cache without calling provider (mock the provider,
     verify it is called exactly once for two identical embed() calls)
   - `test_different_texts_different_cache_keys` — two different texts produce different
     cache entries, neither is returned for the other
   - `test_cache_survives_new_service_instance` — save embedding via one EmbeddingService,
     create a second EmbeddingService pointing to same Redis, verify it hits cache

5. `tests/integration/infra/test_memory_stack.py` — Memory system with real PostgreSQL:
   - `test_episodic_write_and_retrieve` — store 3 episodic records, retrieve by query,
     top result is semantically closest (use pre-computed embeddings)
   - `test_semantic_fact_deduplication` — store same fact twice, list_all returns 1 record
   - `test_memory_manager_remember_and_recall` — full write path through MemoryManager
     (provenance + validator + store), then recall returns the stored content
   - `test_consolidation_extracts_facts` — mock the LLM call in ConsolidationPipeline,
     run pipeline, verify SemanticFacts written to semantic memory

---

### Suite B: Real LLM Tests

**Gate:** `RUN_REAL_LLM_TESTS=true` environment variable AND API keys present
**Trigger:** Nightly cron (3am UTC) + manual `workflow_dispatch` in GitHub Actions
**Cost:** < $0.50/run using cheapest models with `max_tokens` caps
**Location:** `tests/integration/real_llm/`
**Models:** `claude-haiku-4-5-20251001` (Anthropic), `gpt-4o-mini` (OpenAI)

**Tasks (Suite B):**

1. `tests/integration/real_llm/conftest.py` — Shared fixtures and guards:
   - Module-level skip guard:
     ```python
     import os, pytest
     if not os.environ.get("RUN_REAL_LLM_TESTS"):
         pytest.skip("Set RUN_REAL_LLM_TESTS=true to run", allow_module_level=True)
     ```
   - `anthropic_client` fixture: `AnthropicClient(api_key=os.environ["ANTHROPIC_API_KEY"])`
   - `openai_client` fixture: `OpenAIClient(api_key=os.environ["OPENAI_API_KEY"])`
   - `cost_budget` session fixture: tracks accumulated `cost_usd` across the session.
     After each test, add result's `token_usage.cost_usd`. If total > $0.50, call
     `pytest.skip("Session cost budget exceeded")` on subsequent tests.
   - Mark all tests `@pytest.mark.real_llm`.

2. `tests/integration/real_llm/test_anthropic_client.py`:
   - `test_complete_basic` — single user message "Say OK", assert `result.content` is non-empty
     string, `result.token_usage.total_tokens > 0`, `result.cost_usd > 0`
   - `test_complete_returns_stop_reason` — assert `result.stop_reason` in `("end_turn", "tool_use")`
   - `test_streaming_no_content_loss` — stream the same prompt, concatenate all text chunks,
     assert final content matches a non-streaming call to same prompt (both at temperature=0)
   - `test_streaming_handles_empty_deltas` — stream a long response, assert no exception raised
     when processing zero-content delta chunks
   - `test_tool_call_roundtrip` — register a calculator tool schema, send "what is 7 * 8?",
     assert `result.tool_calls` is not None, first tool_call.name == "calculator",
     arguments contain "expression" key
   - `test_tool_call_streaming` — same as above but via `stream()`, accumulate tool call chunks,
     assert final tool_call is fully formed (not split/partial)
   - `test_multi_turn_context` — two messages: "My name is TestBot" then "What is my name?",
     assert second response contains "TestBot" (context was preserved)
   - `test_back_to_back_tool_calls` — prompt that should trigger two sequential tool calls,
     assert `len(result.tool_calls) >= 1` (structural, not semantic)
   - `test_budget_enforcement` — construct `AgentDefinition(cost_budget_usd=0.000001)`,
     run a query via `AgentRunner`, assert `BudgetExceededError` raised

3. `tests/integration/real_llm/test_openai_client.py`:
   - Same structure as Anthropic file above, adapted for OpenAI API format.
   - Additional: `test_tool_call_id_format` — OpenAI tool call IDs start with "call_",
     assert `result.tool_calls[0].id.startswith("call_")`

4. `tests/integration/real_llm/test_streaming_robustness.py`:
   - `test_split_json_tool_call_accumulation` — use a prompt that reliably produces a tool call
     with a long arguments JSON, stream the response, verify the accumulated tool_call.arguments
     is valid JSON (i.e., our accumulator correctly handles chunk boundaries)
   - `test_streaming_with_system_prompt` — system prompt + user message, stream, assert
     both `len(chunks) > 1` (actually streamed) and final content is non-empty
   - `test_stream_interrupt_cleanup` — start streaming, break out of loop after first chunk,
     verify no resource leak (no hanging connection, `asyncio.get_event_loop().is_running()`)

5. `tests/integration/real_llm/test_agent_runner_real.py`:
   - `test_agent_solves_math_with_tool` — AgentRunner with calculator tool, ask "what is
     sqrt(144) + 15?", assert tool was called AND final output contains "27"
   - `test_agent_respects_max_iterations` — set `max_iterations=1`, give a task requiring
     2+ steps, assert runner returns after 1 iteration without error
   - `test_agent_multi_turn_with_memory` — two separate `runner.run()` calls with same
     `session_id`, second call references fact from first, assert continuity
     (requires WorkingMemory wired up with in-memory store)

6. `tests/integration/real_llm/test_rate_limit_handling.py`:
   - `test_retry_on_529` — use `respx` to intercept one call and return 529, then pass
     through. Verify runner retries and ultimately succeeds (tests our retry logic
     against the shape of a real 529 response, not a mock string)
   - `test_token_usage_matches_api` — make a call, compare `result.token_usage.input_tokens`
     to tiktoken estimate for the same prompt. Assert within 20% (tests that we're reading
     the API's actual count, not just returning our estimate)

---

### GitHub Actions Wiring

**Tasks (CI wiring):**

1. Update `.github/workflows/ci.yml` — add `RUN_INTEGRATION_TESTS=true` to the existing
   test job that runs on PRs to `main`. Add `pytest -m integration` as a separate step
   after the unit tests. No new secrets needed (testcontainers only).

2. Create `.github/workflows/real-llm-tests.yml`:
   ```yaml
   name: Real LLM Integration Tests
   on:
     schedule:
       - cron: '0 3 * * *'   # nightly 3am UTC
     workflow_dispatch:
       inputs:
         provider:
           description: 'Which provider(s) to test'
           type: choice
           options: [all, anthropic, openai]
           default: all
   jobs:
     real-llm:
       runs-on: ubuntu-latest
       env:
         RUN_REAL_LLM_TESTS: 'true'
         ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
         OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
       steps:
         - uses: actions/checkout@v4
         - uses: astral-sh/setup-uv@v3
         - run: uv sync --all-extras
         - run: uv run pytest tests/integration/real_llm/ -v --tb=short -x
         - name: Post cost summary
           if: always()
           run: uv run python scripts/report_test_costs.py
   ```

3. Create `scripts/report_test_costs.py` — reads a `test_costs.json` file written by
   the `cost_budget` fixture during the session. Prints total cost and per-test breakdown.
   If cost > $1.00, print a warning to stdout (visible in CI logs).

---

### Acceptance Criteria

**Suite A (Infrastructure):**
- [ ] `pytest -m integration` passes with PostgreSQL + Redis containers (no Dapr sidecar needed)
- [ ] pgvector HNSW index created and queried successfully
- [ ] Dimension mismatch detected at setup time, not silently at write time
- [ ] Namespace isolation confirmed: namespace "a" queries never return namespace "b" data
- [ ] Embedding cache round-trip with real Redis: provider called once for two identical embed() calls
- [ ] Memory stack: episodic write → retrieve returns stored content
- [ ] All Suite A tests pass in < 3 minutes on CI

**Suite B (Real LLM):**
- [ ] `RUN_REAL_LLM_TESTS=true pytest tests/integration/real_llm/ -v` — all pass
- [ ] Both Anthropic and OpenAI tool call round-trips work end-to-end
- [ ] Streaming accumulates complete tool calls (no split-JSON failures)
- [ ] Back-to-back tool calls in one response handled correctly
- [ ] Budget enforcement stops AgentRunner before API call completes
- [ ] Total cost per nightly run < $0.50
- [ ] Nightly GitHub Actions workflow runs without manual intervention

**CI Wiring:**
- [ ] `.github/workflows/ci.yml` runs Suite A on every PR to main
- [ ] `.github/workflows/real-llm-tests.yml` runs on schedule (nightly) and `workflow_dispatch`
- [ ] Secrets `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` documented in `CONTRIBUTING.md`
- [ ] `scripts/report_test_costs.py` prints cost summary after each Suite B run

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
