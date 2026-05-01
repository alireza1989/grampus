# Architecture Decision Records

## ADR-001: Dapr as Infrastructure Backbone
**Status:** Accepted
**Context:** Need durable execution, state management, pub/sub, security, and observability without building from scratch.
**Decision:** Use Dapr runtime as the infrastructure layer. Agent code talks only to Dapr APIs, never directly to databases or message brokers.
**Consequences:** Requires Dapr sidecar running alongside agent process. Adds operational complexity but saves 6-12 months of distributed systems engineering. All infrastructure is swappable via Dapr component YAML — no code changes to switch from Redis to Kafka for pub/sub.

## ADR-002: PostgreSQL + pgvector as Primary State Store
**Status:** Accepted
**Context:** Need ACID-compliant, auditable storage for memory with vector search capability.
**Decision:** PostgreSQL 16 with pgvector extension as primary state store. Redis as cache layer for working memory and tool result caching.
**Consequences:** Single database for relational + vector data reduces operational complexity. pgvectorscale benchmarks show competitive performance (471 QPS at 99% recall on 50M vectors). Graph queries will use recursive CTEs in PostgreSQL rather than a separate graph database unless proven insufficient.

## ADR-003: Pydantic v2 for All Data Models
**Status:** Accepted
**Context:** Need strict validation, serialization, and schema generation for agent definitions, tool parameters, memory records, and API contracts.
**Decision:** All public types are Pydantic v2 BaseModel subclasses. Use pydantic-settings for configuration.
**Consequences:** Strict mode catches bugs at construction time. JSON schema generation enables automatic tool documentation. Serialization is consistent across the codebase. Adds import time (~100ms) but this is negligible for agent workflows.

## ADR-004: Async-First Architecture
**Status:** Accepted
**Context:** Agents are I/O-bound (LLM API calls, tool execution, database operations). Need high concurrency.
**Decision:** All I/O operations use async/await. httpx for HTTP, asyncpg for PostgreSQL, aioredis for Redis.
**Consequences:** Entire call chain must be async. Test fixtures must use pytest-asyncio. Synchronous code (e.g., CPU-bound tool execution) runs in thread pool executor.

## ADR-005: Event Sourcing for Agent Actions
**Status:** Accepted
**Context:** Need full auditability, replay capability, and time-travel debugging for agent executions.
**Decision:** Every agent action (LLM call, tool call, memory read/write, decision point) is stored as an immutable event in an append-only log via Dapr state store.
**Consequences:** Current state is derived from event replay. Enables forensic debugging ("why did the agent do X at step 5?"). Adds write amplification but enables compliance-ready audit trails.

## ADR-006: Memory Write Provenance as Non-Negotiable
**Status:** Accepted
**Context:** Memory poisoning attacks (MINJA, MemoryGraft) achieve 95%+ success rates. Memory is the primary attack surface for persistent agent compromise.
**Decision:** Every memory write must include provenance metadata (source_type, source_id, trust_level, content_hash). Memory writes without provenance are rejected. This is enforced at the Dapr state store wrapper level, not at the application level, so it cannot be bypassed.
**Consequences:** Adds ~2ms overhead per memory write. All retrieval queries can filter by trust level. Memory auditor can verify integrity via content hashes.

## ADR-007: Sandbox by Default for Tool Execution
**Status:** Accepted
**Context:** Agents execute code and call external APIs. Unsandboxed execution grants LLM-generated actions access to host system.
**Decision:** All tool execution runs in Docker container sandbox by default. Network access, filesystem access, and resource limits are configured per-tool. Opt-out requires explicit configuration.
**Consequences:** Adds ~200ms cold-start latency for first tool call in a session (container spin-up). Warm containers reuse reduces subsequent calls to ~10ms overhead. Prevents the class of vulnerabilities found in OpenClaw (190 advisories).

## ADR-008: OpenTelemetry for All Observability
**Status:** Accepted
**Context:** Need distributed tracing, metrics, and logs that work with any backend (Jaeger, Prometheus, Grafana, Datadog, etc.).
**Decision:** OpenTelemetry is the observability standard. Dapr provides infrastructure-level OTEL. Nexus adds agent-specific custom spans (agent.run, agent.llm_call, agent.tool_call, etc.).
**Consequences:** Any OTEL-compatible backend works out of the box. Custom spans enable agent-specific debugging that generic APM tools cannot provide.

## ADR-009: Code Agents as Primary, JSON Tool Calling as Fallback
**Status:** Accepted
**Context:** Smolagents proved that agents writing Python code compose better and handle data transformations more naturally than JSON tool calling.
**Decision:** Support both code agents (LLM writes Python executed in sandbox) and JSON tool calling. Code agents are the recommended default for complex tasks.
**Consequences:** Requires robust sandboxing (ADR-007). Code execution captures stdout/stderr and return values. Sandbox namespace includes registered tools as callable functions.

## ADR-010: MCP + A2A Protocol Support from Day One
**Status:** Accepted
**Context:** MCP (Model Context Protocol) is becoming the standard for tool integration (97M monthly SDK downloads). A2A (Agent-to-Agent) enables cross-framework agent discovery.
**Decision:** Implement MCP client in the tool layer (Phase 6). Implement A2A discovery in the orchestration layer (Phase 7+). Both are standards-compliant, not custom protocols.
**Consequences:** Nexus agents can use any MCP-compatible tool server. Other frameworks' agents can discover and invoke Nexus agents via A2A. Avoids ecosystem lock-in.
