# Nexus — Production-Grade Agentic AI Framework

## What This Is
Open-source agentic AI framework built on Dapr's distributed runtime. Provides agent intelligence (memory, orchestration, safety, observability, evaluation) while Dapr handles infrastructure (state, pub/sub, workflows, security, scaling). Goal: as simple as CrewAI to start, as powerful as LangGraph for production.

## Tech Stack
- **Language:** Python 3.12+ (primary), TypeScript (future)
- **Infrastructure:** Dapr 1.17+ (sidecar), Docker, Kubernetes (production)
- **Databases:** PostgreSQL 16+ with pgvector (primary), Redis 7+ (cache)
- **Testing:** pytest, pytest-asyncio, hypothesis, testcontainers-python
- **Tooling:** ruff (lint+format), mypy (strict), uv (deps), structlog (logging)
- **Package:** pyproject.toml (PEP 621), hatchling build backend

## Commands
```bash
uv sync                           # Install deps
uv run pytest                     # Run tests
uv run pytest -x --tb=short       # Stop on first failure
uv run ruff check . && uv run ruff format . && uv run mypy src/  # Full check
dapr run --app-id nexus --app-port 8000 --resources-path ./dapr/components -- uv run python -m nexus
```

## Code Conventions
- Pydantic v2 models for ALL public types, validation, serialization
- Async-first: all I/O uses async/await
- Type hints on ALL functions (mypy strict enforced)
- Google-style docstrings on all public classes/methods
- structlog with JSON output — never print()
- Absolute imports only, snake_case files, PascalCase classes
- Tests mirror source: `src/nexus/memory/store.py` → `tests/memory/test_store.py`
- Max 50 lines per function. Decompose if longer.
- No global mutable state. Dependency injection for everything.
- Custom exceptions inherit from `NexusError` with machine-readable `code` field

## Architecture (bottom → top)
1. **Dapr Runtime** — state, pub/sub, workflows, actors, mTLS, OTEL
2. **Core** — config, errors, logging, base types, DI, model clients
3. **Memory** — working, episodic, semantic, procedural + security layer
4. **Tools** — registry, MCP client, sandboxed execution
5. **Orchestration** — graph engine, model router, cost tracker, agent loop
6. **Safety** — injection detection, PII, action boundaries, policies
7. **Observability** — agent OTEL spans, metrics, behavior monitoring
8. **Evaluation** — eval suites, prompt versioning, quality baselines
9. **CLI** — init, run, eval, deploy, dev commands

## Hard Rules
- NEVER commit secrets or API keys
- NEVER bypass sandbox for tool execution
- NEVER write to databases directly — always through Dapr State API
- ALWAYS write failing test first, then implement (TDD)
- ALWAYS run full check before marking any task complete
- Commit with conventional commits: `feat(module): description`

## Plans & Architecture
- @PLAN.md — full phased implementation plan (start here)
- @docs/architecture/decisions.md — architecture decision records

## On Compacting
Preserve: current phase, modified files, test results, unresolved errors, acceptance criteria.
