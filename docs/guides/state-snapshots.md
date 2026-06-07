# Agent State Snapshots

Snapshots let you export a full agent session as a portable JSON file. Use them for forensic debugging of unexpected behavior, migrating sessions between environments (dev → staging → prod), creating reproducible eval baselines, or disaster recovery.

---

## Exporting a snapshot

=== "CLI"

    ```bash
    # Export by agent ID (latest session)
    nexus state export research-bot --output snapshot.json

    # Export a specific session
    nexus state export research-bot --session ses_abc123 --output snapshot.json

    # Tag the snapshot for later filtering
    nexus state export research-bot --session ses_abc123 \
      --tag env=production --tag reason=incident-2026-06-01 \
      --output snapshot.json
    ```

=== "Python API"

    ```python
    from nexus.dapr.state import DaprStateStore
    from nexus.orchestration.snapshot import SnapshotManager

    state_store = DaprStateStore(dapr_client, namespace="research")
    mgr = SnapshotManager(state_store=state_store)

    snapshot = await mgr.export_session(
        agent_id="research-bot",
        session_id="ses_abc123",
    )
    mgr.to_file(snapshot, "snapshot.json")
    print(f"Exported {len(snapshot.state.messages)} messages")
    print(f"Events in log: {snapshot.event_log_count}")
    ```

---

## Inspecting a snapshot

```bash
# Human-readable table view
nexus state show snapshot.json --format table

# Raw JSON
nexus state show snapshot.json --format json
```

Table output:

```
Field                Value
─────────────────────────────────────────────────────
schema_version       1
agent_id             research-bot
session_id           ses_abc123
status               completed
messages             14
event_log_count      31
source_environment   production
exported_at          2026-06-01T09:15:22Z
tags                 env=production, reason=incident-2026-06-01
```

### Available fields in a snapshot

| Field | Description |
|-------|-------------|
| `schema_version` | Snapshot format version |
| `agent_id` | Agent that produced this session |
| `session_id` | Unique session identifier |
| `state.messages` | Full message history (user, assistant, tool) |
| `state.status` | Final `AgentStatus` at export time |
| `event_log_count` | Number of immutable events captured |
| `tags` | Arbitrary key=value metadata attached at export |
| `source_environment` | Where the snapshot was taken (`NEXUS_ENVIRONMENT` env var) |

---

## Restoring a snapshot

```bash
# Preview what would be restored (no writes)
nexus state import snapshot.json --dry-run

# Restore to the current environment
nexus state import snapshot.json
```

!!! warning "Restore overwrites current state"
    Importing a snapshot overwrites the current state for that agent/session in Dapr. Any in-progress work for that session will be replaced. Use `--dry-run` first to verify what will change.

=== "Python API"

    ```python
    import_result = await mgr.import_session(snapshot, dry_run=False)
    print(f"Restored: agent={import_result.agent_id} session={import_result.session_id}")
    print(f"Messages restored: {import_result.messages_restored}")
    ```

---

## REST API

```bash
# Export via HTTP
curl "http://localhost:8000/agents/ses_abc123/snapshot" \
  --output snapshot.json

# Restore via HTTP
curl -X POST http://localhost:8000/agents/snapshot/restore \
  -H "Content-Type: application/json" \
  -d @snapshot.json
```

---

## Use cases

- **Forensic debugging**: export a session where the agent made an unexpected decision, then replay the event log step by step to find where behavior diverged from expectations
- **Environment migration**: export a dev session with realistic conversation history, import it into staging to run eval suites against production-like state
- **Eval baselines**: snapshot a known-good session and use its message history as the starting point for regression eval cases
- **Disaster recovery**: nightly snapshots of active agent sessions provide a restore point if the Dapr state store is corrupted or accidentally wiped

---

## See also

- **[CLI reference →](../reference/cli.md)** — Full `nexus state` command reference
- **[Observability guide →](observability.md)** — Event log and session replay
- **[Evaluation guide →](evaluation.md)** — Creating eval baselines from real sessions
