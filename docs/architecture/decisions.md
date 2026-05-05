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
