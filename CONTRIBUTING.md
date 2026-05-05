# Contributing to Nexus

Thank you for contributing. This document covers everything you need to go from zero to a merged pull request.

---

## Development setup

```bash
git clone https://github.com/nexus-ai/nexus-agentic-platform
cd nexus-agentic-platform
uv sync                  # install all dependencies including dev extras
docker compose up -d     # start PostgreSQL, Redis, and Dapr placement service
uv run pytest            # verify your setup — all tests must pass before you start
```

You need Python 3.12+, Docker, and [uv](https://docs.astral.sh/uv/) installed. On macOS: `brew install uv`.

---

## Running tests

```bash
uv run pytest                                     # full suite
uv run pytest -x --tb=short                       # stop on first failure
uv run pytest tests/memory/ -v                    # one module
uv run pytest -m "not integration"                # skip integration tests
uv run pytest -m integration                      # integration only (requires docker compose up)
```

Integration tests spin up real PostgreSQL and Redis via testcontainers. They are marked `@pytest.mark.integration` and require Docker running. The CI pipeline runs them separately.

---

## Code conventions

These rules are enforced by ruff, mypy, and CI — violations block merge.

- **Type hints on every function** — mypy strict mode, no `Any` escapes
- **Pydantic v2 for all public types** — no raw dicts crossing module boundaries
- **Async-first** — all I/O is `async`/`await`; synchronous tool functions use `run_in_executor`
- **structlog for all logging** — never `print()` or the standard `logging` module directly
- **Absolute imports only** — `from nexus.core.types import Message`, not relative paths
- **Max 50 lines per function** — decompose longer functions; no exceptions
- **No global mutable state** — inject dependencies, don't reach for globals
- **Custom exceptions inherit `NexusError`** — with a machine-readable `code` field

Run the full check before every commit:

```bash
uv run ruff check . && uv run ruff format . && uv run mypy src/
```

---

## TDD requirement

**Write the failing test first. Always.** This is not optional.

1. Write a test that asserts the behavior you are about to implement
2. Run it — watch it fail
3. Implement the minimum code to make it pass
4. Refactor; test must still pass

Tests live in `tests/` mirroring the source tree: `src/nexus/memory/store.py` → `tests/memory/test_store.py`. Property-based tests using Hypothesis are encouraged for any function that processes external input.

---

## Conventional commits

All commits must follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`

Scopes match architecture layers: `core`, `dapr`, `memory`, `tools`, `orchestration`, `safety`, `observability`, `evaluation`, `cli`

Examples:

```
feat(memory): add temporal decay to episodic retrieval scoring
fix(safety): correct PII regex for international phone formats
test(orchestration): add hypothesis tests for cost tracker budget enforcement
docs(cli): document --fail-under flag for nexus eval
```

---

## Pull request process

1. Fork the repository and create a branch from `main`
2. Write failing tests, implement, verify tests pass
3. Run `uv run ruff check . && uv run mypy src/ && uv run pytest`
4. Open a pull request using the [PR template](.github/pull_request_template.md)
5. CI runs lint, typecheck, unit tests, integration tests, and docs build
6. One approval from a maintainer is required to merge
7. Squash-merge to `main` with a conventional commit message

---

## Adding a new memory type

1. Add the record model to `src/nexus/memory/types.py`
2. Implement the store class in `src/nexus/memory/<name>.py` — it must use `DaprStateStore` for persistence
3. Add a retriever in `src/nexus/memory/<name>_retriever.py`
4. Expose it through `MemoryManager` in `src/nexus/memory/manager.py` — add a method to `remember()`, `recall()`, and `forget()`
5. All writes must go through `MemoryValidator` and `ProvenanceTracker`
6. Add tests in `tests/memory/test_<name>.py`

---

## Adding a new tool

1. Use the decorator API in your agent code — no framework changes needed for simple tools
2. For tools that should ship with Nexus, add them to `src/nexus/tools/builtins/`
3. Register in `src/nexus/tools/registry.py`
4. If the tool executes code or calls external systems, it must run through `SandboxManager`
5. Add tests in `tests/tools/test_<name>.py` — include a test that verifies sandbox isolation

---

## Adding a new eval assertion

1. Add the assertion class to `src/nexus/evaluation/assertions.py`
2. Inherit from `Assertion` base class and implement `async evaluate(result) -> AssertionResult`
3. Add the type to the `AssertionType` enum
4. Write tests in `tests/evaluation/test_assertions.py` — include both passing and failing cases

---

## Adding a new CLI command

1. Create `src/nexus/cli/commands/<name>.py` with a Click command function
2. Register the command in `src/nexus/cli/main.py`
3. Add `--help` text to all options
4. Write tests in `tests/cli/test_<name>.py` using Click's `CliRunner`

---

## Architecture decision records

Major design decisions are captured in [docs/architecture/decisions.md](docs/architecture/decisions.md). If your contribution changes a fundamental design choice — storage backend, protocol, concurrency model — open an issue to discuss it first and then update the ADR doc as part of your PR.

---

## Code of conduct

Be respectful and focus on the work. Critique ideas, not people. If something is wrong, explain why and propose an alternative. Maintainers reserve the right to close contributions that are disrespectful or off-topic.