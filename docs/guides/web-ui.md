# Web UI

Nexus includes a built-in web interface at `/ui/` for visual inspection and monitoring of your agents. No separate installation is required — the UI ships with `grampus-ai[server]` and is served directly from the Grampus FastAPI server.

---

## Starting the UI

=== "grampus serve"

    ```bash
    pip install "grampus-ai[server]"
    grampus serve
    # Open http://localhost:8000/ui/
    ```

=== "uvicorn directly"

    ```bash
    uv run uvicorn grampus.server.app:create_app --factory --host 0.0.0.0 --port 8000
    # Open http://localhost:8000/ui/
    ```

=== "grampus dev (includes UI)"

    ```bash
    grampus dev
    # Open http://localhost:8000/ui/
    ```

The API documentation is available alongside the UI at `http://localhost:8000/docs`.

---

## Dashboard (`/ui/`)

The dashboard gives a real-time overview of your agent fleet:

- **Active agents** — count of currently running agent sessions
- **LLM calls today** — total model calls since midnight UTC
- **Cost today** — accumulated USD spend since midnight UTC
- **Errors today** — count of failed runs and safety violations

Below the stat cards is the **Recent alerts** table showing the last 20 alert events — which rule fired, severity, agent, and timestamp.

The dashboard auto-refreshes every 10 seconds via HTMX. No manual page reload needed.

---

## Memory Inspector (`/ui/memory/`)

The memory inspector lets you browse, search, and manage all memory entries for your agents without writing code.

### Filter bar

At the top of the page, filters narrow the displayed entries:

| Filter | Description |
|--------|-------------|
| Agent ID | Show only records for a specific agent |
| Memory type | `episodic`, `semantic`, `procedural`, or `all` |
| Search text | Full-text search across record content |
| Min trust score | Hide low-trust entries (e.g., set to `0.5` to see only moderate/high trust) |

Hit **Apply** to reload the table with new filters. Filters are bookmarkable — they update the URL query string.

### Memory table

Each row shows:

| Column | Description |
|--------|-------------|
| **Type** | Memory type badge: `episodic`, `semantic`, or `procedural` |
| **Content** | Truncated content preview (click row to expand) |
| **Trust** | Trust score with color coding |
| **Source** | `SourceType` of the provenance record |
| **Created** | Timestamp |
| **Actions** | Delete button |

### Trust score colors

| Color | Score range | Meaning |
|-------|-------------|---------|
| Green | ≥ 0.8 | High trust (system or validated user input) |
| Yellow | 0.5–0.8 | Moderate trust (LLM-generated or tool results) |
| Red | < 0.5 | Low trust (external data or flagged content) |

### Detail panel

Click any row to slide open the detail panel on the right, showing:

- Full content (untruncated)
- Complete `Provenance` metadata (source type, source ID, content hash)
- Importance score and access count (episodic)
- Subject/predicate/object and confidence (semantic)
- Procedure steps (procedural)

### Deleting entries

Click the trash icon in the row **Actions** column to delete a single entry. You will be prompted to confirm. Deletion is immediate and irreversible — the entry is removed from Dapr state.

!!! note "Upcoming pages"
    The **Evals** and **Cost** pages are coming in the next release (D10). The sidebar shows them as placeholders. Check the sidebar navigation: the active page is highlighted; placeholder pages are grayed out.

---

## Programmatic access

All data shown in the UI is available via the REST API — useful for automation and integration with external dashboards:

```bash
# List memory entries
curl "http://localhost:8000/memory?agent_id=research-bot&memory_type=episodic"

# Delete a memory entry
curl -X DELETE "http://localhost:8000/memory/ep-001"

# Dashboard stats
curl "http://localhost:8000/ui/_stats"

# Recent alerts
curl "http://localhost:8000/alerts/history?limit=20"
```

Full API documentation is at `http://localhost:8000/docs`.

---

## See also

- **[Memory guide →](memory.md)** — Programmatic memory management with `MemoryManager`
- **[Cost Management →](cost-management.md)** — Cost alert rules and notification channels
- **[Observability guide →](observability.md)** — Prometheus metrics and Grafana dashboard
