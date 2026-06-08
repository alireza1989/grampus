# Architecture Decision Records

ADRs capture the reasoning behind major design decisions. Each decision is permanent unless explicitly superseded.

---

## ADR-001: Dapr as Infrastructure Backbone

**Status:** Accepted

**Context:** Need durable execution, state management, pub/sub, security, and observability without building from scratch. Building these correctly (with mTLS, ETag concurrency, distributed locks, workflow checkpointing) would require 6–12 months of distributed systems engineering.

**Decision:** Use Dapr runtime as the infrastructure layer. Agent code talks only to Dapr APIs, never directly to databases or message brokers.

**Consequences:**
- Requires Dapr sidecar running alongside agent process
- Adds operational complexity in Kubernetes (two containers per pod)
- All infrastructure is swappable via Dapr component YAML — no code changes to switch from Redis to Kafka for pub/sub
- mTLS between services is zero-configuration
- Dapr emits OTEL traces automatically for all state and messaging operations

---

## ADR-002: PostgreSQL + pgvector as Primary State Store

**Status:** Accepted

**Context:** Need ACID-compliant, auditable storage for memory with vector search capability. Running separate databases for relational data (facts, events) and vector data (embeddings) increases operational complexity.

**Decision:** PostgreSQL 16 with pgvector extension as primary state store. Redis as cache layer for working memory and tool result caching.

**Consequences:**
- Single database for relational + vector data reduces operational complexity
- pgvectorscale benchmarks show competitive performance (471 QPS at 99% recall on 50M vectors)
- Graph queries use recursive CTEs in PostgreSQL rather than a separate graph database (revisit if proven insufficient)
- ACID transactions enable reliable provenance and audit trails without saga complexity

---

## ADR-003: Pydantic v2 for All Data Models

**Status:** Accepted

**Context:** Need strict validation, serialization, and schema generation for agent definitions, tool parameters, memory records, and API contracts. Inconsistent validation across the codebase is a primary source of subtle bugs in agent frameworks.

**Decision:** All public types are Pydantic v2 `BaseModel` subclasses. Configuration uses `pydantic-settings`.

**Consequences:**
- Strict mode catches construction-time bugs (wrong types, missing required fields)
- JSON schema generation from `ToolDefinition.to_function_schema()` enables automatic tool documentation for LLMs
- Serialization is consistent: one code path for storage, HTTP, and logging
- Adds ~100ms import time (negligible for agent workflows where LLM calls dominate)

---

## ADR-004: Async-First Architecture

**Status:** Accepted

**Context:** Agents are I/O-bound: LLM API calls (1–30s), tool execution (10ms–30s), database operations (1–50ms). Synchronous Python would block the event loop on every I/O call, preventing concurrency.

**Decision:** All I/O operations use `async`/`await`. httpx for HTTP, asyncpg for PostgreSQL, aioredis for Redis. Synchronous tool functions run in thread pool executors.

**Consequences:**
- Entire call chain must be async; synchronous callers use `asyncio.run()`
- Test fixtures require `pytest-asyncio` with `asyncio_mode = "auto"`
- Enables serving many concurrent agent runs in a single process
- CPU-bound tool execution (e.g., image processing) runs in `loop.run_in_executor()`

---

## ADR-005: Event Sourcing for Agent Actions

**Status:** Accepted

**Context:** Need full auditability, replay capability, and time-travel debugging for agent executions. Compliance requirements in financial, healthcare, and legal domains require verifiable audit trails.

**Decision:** Every agent action (LLM call, tool call, memory read/write, decision point) is stored as an immutable event in an append-only log via Dapr state store.

**Consequences:**
- Current state is derived from event replay
- Enables forensic debugging: "why did the agent do X at step 5?"
- Adds write amplification (every action writes to both operational state and event log)
- Enables compliance-ready audit trails without additional infrastructure

---

## ADR-006: Memory Write Provenance as Non-Negotiable

**Status:** Accepted

**Context:** Memory poisoning attacks (MINJA, MemoryGraft) achieve 95%+ success rates against unprotected agents. Memory is the primary attack surface for persistent agent compromise — a successful poisoning persists across sessions and survives context window limits.

**Decision:** Every memory write must include provenance metadata (`source_type`, `source_id`, `trust_level`, `content_hash`). Memory writes without provenance are rejected. This is enforced at the `DaprStateStore` wrapper level, not at the application level, so it cannot be bypassed by application code.

**Consequences:**
- Adds ~2ms overhead per memory write (SHA-256 hash computation + metadata storage)
- All retrieval queries can filter by trust level
- Memory auditor can verify integrity via content hashes
- Makes memory poisoning attacks significantly harder — tampered content changes the hash, triggering auditor alerts

---

## ADR-007: Sandbox by Default for Tool Execution

**Status:** Accepted

**Context:** Agents execute LLM-generated code and call external APIs. Unsandboxed execution grants LLM-generated actions full access to the host system — filesystem, network, environment variables, and other processes. OpenClaw (2024) found 190 security advisories in popular agent frameworks due to unsandboxed execution.

**Decision:** All tool code execution runs in Docker container sandbox by default. Network access, filesystem access, and resource limits are configured per-tool. Opt-out requires explicit configuration.

**Consequences:**
- Adds ~200ms cold-start latency for first tool call in a session (container spin-up)
- Warm container reuse reduces subsequent calls to ~10ms overhead
- Prevents host system compromise via prompt injection → code execution
- Requires Docker daemon running alongside agent

---

## ADR-008: OpenTelemetry for All Observability

**Status:** Accepted

**Context:** Need distributed tracing, metrics, and logs that work with any backend (Jaeger, Prometheus, Grafana, Datadog, Honeycomb, etc.). Vendor lock-in to a specific observability platform would prevent adoption in organizations with existing tooling.

**Decision:** OpenTelemetry is the observability standard. Dapr provides infrastructure-level OTEL automatically. Nexus adds agent-specific custom spans: `agent.run`, `agent.llm_call`, `agent.tool_call`, `agent.memory_read`, `agent.memory_write`, `agent.decision`.

**Consequences:**
- Any OTEL-compatible backend works out of the box — change the exporter endpoint, not the code
- Custom spans enable agent-specific debugging that generic APM tools cannot provide
- Agents running in Kubernetes benefit from Dapr's automatic service mesh tracing
- Token cost and model information are captured as span attributes, enabling cost analysis via trace queries

---

## ADR-009: Code Agents as Primary, JSON Tool Calling as Fallback

**Status:** Accepted

**Context:** Smolagents research (2024) demonstrated that agents writing Python code compose tools more flexibly and handle data transformations more naturally than JSON tool calling. Code agents can chain tool calls, use Python data structures, and perform calculations without additional LLM calls.

**Decision:** Support both code agents (LLM writes Python executed in sandbox) and JSON tool calling (standard function calling). Code agents are the recommended default for complex, multi-step tasks.

**Consequences:**
- Requires robust sandboxing (ADR-007)
- Code execution captures stdout, stderr, and return values
- Sandbox Python namespace includes registered tools as callable functions
- Simpler tasks can use JSON tool calling to avoid sandbox overhead

---

## ADR-010: MCP + A2A Protocol Support from Day One

**Status:** Accepted

**Context:** MCP (Model Context Protocol) is becoming the standard for tool integration (97M monthly SDK downloads as of 2025). A2A (Agent-to-Agent) enables cross-framework agent discovery. Building custom tool protocols creates ecosystem lock-in and prevents Nexus agents from using the growing ecosystem of MCP-compatible tools.

**Decision:** Implement MCP client in the tool layer (Phase 6). Implement A2A discovery in the orchestration layer (Phase 7+). Both are standards-compliant implementations, not custom protocols.

**Consequences:**
- Nexus agents can use any MCP-compatible tool server (filesystem, browser, databases, APIs)
- Other frameworks' agents can discover and invoke Nexus agents via A2A
- Avoids ecosystem lock-in — Nexus works alongside LangGraph, CrewAI, and Autogen
- Requires tracking protocol evolution as both MCP and A2A mature

---

## ADR-011: Consolidated HTMX + Jinja2 Web UI

**Status:** Accepted

**Context:** Multiple post-launch phases require visual interfaces: memory inspector (D9), eval dashboard (D10), cost analytics, alert management, and an execution trace viewer. Two alternative approaches were considered: (a) separate CLI commands for each feature, or (b) separate web apps or SPAs per feature. Both create fragmentation — users must remember different URLs or commands, state cannot be shared across views (e.g., filtering by agent_id in the sidebar should filter all pages), and each feature reimplements the same table/chart components.

**Decision:** All web UI phases build into a single consolidated web app served at `/ui/` from the existing FastAPI server. Technology stack: HTMX (loaded from CDN — no npm, no build step) + Jinja2 templates for server-side rendering. The D9 phase builds the shell (base template, sidebar navigation, layout system) and all subsequent UI phases add pages to it. One new optional dependency: `jinja2>=3.0` added to the `server` extras group (already a transitive FastAPI dependency in practice).

**Exception:** The Visual Agent Builder (drag-and-drop graph editor) requires rich interactivity — sortable nodes, canvas pan/zoom, live edge drawing — that HTMX cannot support. That feature uses a minimal React SPA bundled at `src/nexus/server/ui/static/builder/` and served at `/ui/builder/`. It is the only component permitted to introduce a frontend build step.

**Consequences:**
- No Node.js toolchain required to run or develop the UI — `uv sync` is sufficient
- Single URL entry point; sidebar navigation shared across all views
- HTMX partial endpoints (`/ui/<feature>/_<partial>`) enable dynamic updates (live cost tickers, SSE-driven agent status) without full page reloads
- HTMX has limits on complex client-side interactivity — sufficient for developer dashboards, not for visual graph editors (see exception above)
- Static assets (CSS, minimal JS helpers) live in `src/nexus/server/ui/static/` and are served by FastAPI's `StaticFiles` mount
- Jinja2 templates live in `src/nexus/server/ui/templates/` with a `base.html` that all pages extend

---

## ADR-012: Multi-Agent Debate as a First-Class Orchestration Primitive

**Status:** Accepted

**Context:** High-stakes agent tasks (legal analysis, medical triage, financial decisions) cannot rely on a single LLM call because (a) individual models hallucinate on specialised questions and (b) there is no confidence signal that a single model can reliably self-report. Two prior approaches exist: prompt-level self-consistency (same model, multiple samples) and multi-agent crews (different agents, different roles). Self-consistency degrades on hard questions because sampling diversity is bounded by a single model's knowledge. Crews require pre-defined pipelines and do not provide a convergence signal. Research (Du et al. ICML 2024; M3MAD-Bench ICLR 2025) demonstrates that heterogeneous models arguing toward a shared answer reach substantially higher accuracy than either alternative.

**Decision:** Implement `DebateOrchestrator` as a standalone orchestration primitive in `src/nexus/orchestration/debate/`. It operates on a single question rather than a task pipeline, runs all debaters concurrently per round via `asyncio.gather`, and integrates with the existing `Graph` engine via `debate_node()`. Four specific research findings are baked into the design:

1. **Heterogeneous panels** — `DebaterConfig.model_id` allows mixing model families, not just temperatures. The aggregator uses `debater.weight` to handle unequal capability.
2. **Sycophancy resistance** — Round 2+ prompts require debaters to restate their prior answer verbatim before evaluating peers, and to cite specific logical evidence for any position change (ACL 2025 CONSENSAGENT).
3. **Adaptive routing** — If a fast routing model reports confidence ≥ threshold, the full debate is bypassed. This eliminates ~40% of unnecessary calls with no quality loss (arXiv 2504.05047).
4. **Act-vs-escalate** — When the final convergence score is below `escalate_threshold`, the result sets `escalate_to_human=True` rather than silently returning a low-confidence answer ("From Debate to Decision", April 2026).

**Consequences:**
- Zero new runtime dependencies — stdlib `json`, `asyncio`, `re`, `time` plus existing Pydantic and OTEL
- `debate_node()` integrates cleanly with the existing `Graph` conditional-edge API; human escalation uses the existing `human_node`
- Concurrent debaters within a round mean latency is bounded by the slowest debater, not the sum — no worse than a single LLM call per round
- Cost scales as `num_debaters × num_rounds` but adaptive routing mitigates this for easy questions
- The convergence detector uses Jaccard word-overlap clustering (no ML model, no embedding calls) — fast and deterministic

---

## ADR-013: Dual-Process Uncertainty Quantification as a First-Class Runner Feature

**Status:** Accepted

**Context:** Agents produce unreliable outputs at unknown rates. Single-call verbalized confidence (asking the model to write `"confidence": 0.8`) has a documented ECE of 0.377+ even on frontier models (arXiv 2412.14737, KDD 2025 survey) — aligned models cluster at 90–100% confidence regardless of factual accuracy. Existing frameworks either ignore this or apply per-call thresholds that do not account for how uncertainty accumulates across sequential steps. A grounding error in step 1 biases all downstream reasoning (the "Spiral of Hallucination"), so per-step overconfidence checking is insufficient. There is also no standard mechanism for agents to escalate irreversible actions (send_email, delete, deploy) to humans when confidence is too low.

**Decision:** Implement `UncertaintyMonitor` as an optional hook in `AgentRunner`, not as a separate layer. Four research findings are baked directly into the implementation:

1. **Dual-process estimation** (arXiv 2601.15703, Jan 2026) — System 1 (fast): P(True) self-evaluation fused with verbalized confidence, both calibrated. System 2 (slow, opt-in): adaptive semantic entropy sampling when fused confidence is in the uncertain middle zone.
2. **P(True) as primary fast signal** (Kadavath et al. 2022) — A single follow-up call asking "Is your answer correct?" achieves ECE ≈ 0.10 on frontier models without logit access. Verbalized confidence (weight 0.4) remains a weak supporting signal alongside P(True) (weight 0.6).
3. **Adaptive semantic entropy** (arXiv 2504.03579, 2025) — Start with 2 samples; early-stop if Jaccard ≥ 0.60 (saves ~47% cost); extend to `max_samples` on disagreement. Pessimistic fusion `min(fast, entropy_conf)` prevents over-optimism.
4. **SAUP propagation** (arXiv 2412.01033, ACL 2025 pp. 6064–6073) — Per-step situational weights (decision=0.70, llm=0.55, tool=0.45, memory_read=0.35) ensure a confident step cannot erase uncertain history. 20% AUROC improvement over single-step UQ.

The three-tier escalation ladder (Zylos Research, April 2026) maps propagated confidence → action: PROCEED → PROCEED_WITH_LOG → PAUSE_FOR_HUMAN → ABORT. Irreversible tool names trigger PAUSE at MEDIUM uncertainty. A System-2 reflection prompt is injected before PAUSE so the next LLM call sees explicit uncertainty acknowledgment.

**Consequences:**
- Zero new required dependencies — stdlib `math`, `json`, `re`, `asyncio` plus existing Pydantic and OTEL
- `uncertainty_monitor=None` (the default) means zero overhead for agents that don't need UQ
- Two hooks in the runner loop: post-LLM (checks response confidence) and pre-tool (checks before irreversible actions); both break the loop cleanly with `hit_limit = False`
- `UncertaintyError` (code `UNCERTAINTY_CRITICAL`) gives callers a machine-readable signal on ABORT
- `uncertainty_guard_node()` provides an explicit graph checkpoint between nodes — composable with the existing `debate_node()` and `human_node()` primitives
- OTEL spans (`uncertainty.estimate`, `uncertainty.semantic`, `uncertainty.escalate`) are emitted per step when a tracer is provided, enabling confidence dashboards alongside cost and latency metrics
