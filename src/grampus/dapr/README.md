# `grampus/dapr/` — Infrastructure Abstraction Layer

This package owns every interaction with the Dapr sidecar: state CRUD, pub/sub, distributed locks, scheduled jobs, and health checks. It is the **only** code in the framework that may communicate with the Dapr HTTP/gRPC API.

It does **not** own PostgreSQL, Redis, or Kafka directly — those are configured as Dapr component YAML files in `dapr/components/` and are therefore swappable without any code changes. It does not contain any domain logic (memory records, agent state, tool results, etc.) — it only provides typed, namespaced, serialization-safe wrappers.

---

## Key abstractions

| Class | File | Role |
|---|---|---|
| `DaprGateway` | `client.py` | Async façade over the synchronous Dapr gRPC SDK; dispatches all blocking calls via `asyncio.to_thread` |
| `DaprStateStore` | `state.py` | Namespace-scoped CRUD (`{ns}:{entity_type}:{id}`) with Pydantic serialization and ETag concurrency |
| `DaprLock` | `lock.py` | Distributed lock as `async with DaprLock(…):` — raises `LockAcquisitionError` if not acquired |
| `DaprPubSub` | `pubsub.py` | Typed publish/subscribe with `@pubsub.subscribe` handler registration |
| `DaprJobs` | `jobs.py` | Dapr Jobs API for scheduled / cron-triggered tasks |
| `is_sidecar_healthy` | `health.py` | HTTP health probe against the sidecar's `/v1.0/healthz` endpoint |
| `to_dapr_bytes` / `from_dapr_bytes` | `serialization.py` | Pydantic model ↔ JSON bytes helpers used by `DaprStateStore` |

---

## How to use this package

```python
from grampus.dapr.client import DaprGateway
from grampus.dapr.state import DaprStateStore
from grampus.dapr.lock import DaprLock

# Build the gateway (lazy — SDK client created on first use)
gw = DaprGateway(host="localhost", port=3500)

# Namespaced state store
store = DaprStateStore(gw, store_name="statestore", namespace="memory")

# Save a Pydantic model
await store.save("episodic", record_id, my_record)

# Load back (returns (model | None, etag))
record, etag = await store.get("episodic", record_id, EpisodicRecord)

# Optimistic concurrency — pass etag to reject stale writes
await store.save("episodic", record_id, updated_record, etag=etag)
# → raises ConcurrencyError if another writer modified it first

# Distributed lock
async with DaprLock(gw, store_name="lockstore",
                    resource_id="crew:section-A",
                    lock_owner="worker-1",
                    expiry_seconds=30):
    # exclusive section
    ...

# Pub/sub
from grampus.dapr.pubsub import DaprPubSub
pubsub = DaprPubSub(gw, pubsub_name="pubsub-redis")
await pubsub.publish("cost-events", my_event_model)
```

### In unit tests — inject a mock client

```python
from unittest.mock import MagicMock
gw = DaprGateway(_client=MagicMock())
store = DaprStateStore(gw, store_name="statestore", namespace="test")
```

---

## Hard invariants

- **Agent code must NEVER talk to PostgreSQL, Redis, or any message broker directly.** All persistence and messaging goes through `DaprStateStore` and `DaprPubSub`. Changing the backend is a one-line change to a YAML component file — this only holds if no code bypasses the Dapr API.
- **All state keys follow `{namespace}:{entity_type}:{entity_id}`.** This is enforced by `_make_key()` in `DaprStateStore`. Never construct raw keys manually — cross-component key collisions are silent data corruption.
- **`DaprGateway._client` calls are always wrapped in `asyncio.to_thread`.** The Dapr Python SDK is synchronous. Calling it directly on the event loop blocks all concurrent agent runs. Never add a direct `self._client.method()` call without `asyncio.to_thread`.
- **`ConcurrencyError` must propagate to the caller when an ETag mismatch occurs.** Do not catch it silently inside state helpers. The caller must reload-and-retry or raise to the user. Silent swallowing causes data loss.
- **`DaprLock.__aexit__` always attempts unlock even on exception** (returns `False` so exceptions propagate). Never subclass `DaprLock` to suppress the unlock.
- **`from_dapr_bytes` raises `StateSerializationError` on malformed data** — never catch this in the store layer. Let it surface to the domain layer so corrupted state is visible.
- **`is_sidecar_healthy()` is a probe, not a gate.** Other methods do not call it before every operation — that would double latency. Use it only in startup health checks and CLI `grampus dev`.

---

## Key patterns

### Namespace isolation

Every Grampus subsystem uses its own `DaprStateStore` instance with a unique `namespace`. This means two subsystems can use the same Dapr store component without key collisions:

| Subsystem | Namespace | Store name |
|---|---|---|
| Memory | `memory` | `statestore` |
| Orchestration | `orchestration` | `statestore` |
| Causal world model | `causal` | `statestore` |
| Versioning | `versioning` | `statestore` |
| Events | `events` | `statestore` |

### Bulk operations

For N reads/writes, always use `get_bulk` / `save_bulk` rather than N individual calls. The Dapr SDK batches bulk operations in a single RPC call.

### Transactions

`execute_transaction` is an atomic batch of upsert/delete operations within one store. Use it when you need multi-key atomicity (e.g., updating a record and its index entry together).

---

## Extension guide

### Adding a new Dapr building block

1. Create `src/grampus/dapr/myblock.py`.
2. Import `DaprGateway` and wrap all SDK calls with `await asyncio.to_thread(self._gw._client.method, …)`.
3. All exceptions from the SDK should be caught and re-raised as `DaprError` subclasses.
4. Add tests with a mocked `DaprGateway._client` (inject via `_client=` kwarg).

### Switching the state store backend

Edit `dapr/components/statestore-postgres.yaml` — no Python changes needed. Common alternatives:
- PostgreSQL → DynamoDB: change `type: state.postgresql` to `type: state.aws.dynamodb`
- Redis cache → Memcached: change the `cache` component type

---

## Dependency map

```
dapr/ depends on:      core/ (for errors, logging, serialization helpers)
dapr/ is imported by:  memory/, orchestration/, causal/, versioning/,
                       observability/ (EventLog), evaluation/
dapr/ must NOT import from: memory/, tools/, orchestration/, safety/,
                            evaluation/, causal/, plugins/, versioning/
```

---

## ADR references

- **ADR-001** — Dapr as the infrastructure backbone; rationale for sidecar pattern
- **ADR-002** — PostgreSQL + pgvector as primary state store
- **ADR-005** — Event sourcing via append-only `EventLog` (uses `DaprStateStore`)
- **ADR-006** — Memory provenance enforced at `DaprStateStore` level
