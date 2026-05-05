# CLI Reference

The `nexus` CLI is the primary interface for initializing, running, evaluating, and monitoring Nexus agents.

```bash
nexus --version    # nexus 0.1.0
nexus --help       # show all commands
```

---

## nexus init

Scaffold a new Nexus project.

```bash
nexus init [OPTIONS] [NAME]
```

| Option | Default | Description |
|--------|---------|-------------|
| `NAME` | (prompted) | Project directory name |
| `--name TEXT` | `"nexus-agent"` | Agent name used in config |
| `--template TEXT` | `"simple"` | Project template: `simple`, `crew`, `rag` |
| `--output-dir TEXT` | `"."` | Parent directory for the new project |

### Templates

| Template | Creates | Best for |
|----------|---------|----------|
| `simple` | Single agent with one tool | Getting started, learning Nexus |
| `crew` | Three-agent crew (researcher, critic, writer) | Multi-agent workflows |
| `rag` | RAG agent with document retrieval tool | Question answering over documents |

### Examples

```bash
# Create a simple agent in the current directory
nexus init my-agent

# Create a crew agent in a specific directory
nexus init --template crew --output-dir ~/projects research-crew

# Non-interactive (all defaults)
nexus init --name my-agent --template simple --output-dir .
```

### Generated files

```
my-agent/
├── agent.py              # Agent code with create_runner() and create_agent_def()
├── nexus.yaml            # Configuration
├── docker-compose.yml    # Local infrastructure
├── dapr/
│   ├── config.yaml       # Dapr tracing config
│   └── components/       # State store, pub/sub, cache components
└── .gitignore
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Directory already exists |
| `2` | Invalid template name |

---

## nexus run

Start an agent. Without `--input`, starts an interactive REPL. With `--input`, runs once and exits.

```bash
nexus run [OPTIONS] AGENT_FILE
```

| Argument/Option | Default | Description |
|-----------------|---------|-------------|
| `AGENT_FILE` | (required) | Path to agent Python file |
| `--config TEXT` | `"nexus.yaml"` | Path to nexus.yaml configuration file |
| `--session-id TEXT` | (auto-generated UUID) | Session identifier for memory persistence |
| `--input TEXT` | `None` | Single-shot input; omit for interactive REPL |

The agent file must export two functions:

- `create_runner() -> AgentRunner` — constructs the runner with all dependencies
- `create_agent_def() -> AgentDefinition` — returns the agent blueprint

### Examples

```bash
# Interactive REPL
nexus run agent.py

# Single-shot (useful in scripts and CI)
nexus run agent.py --input "What is the capital of France?"

# Use a specific config file
nexus run agent.py --config config/production.yaml --input "Hello"

# Persist memory across runs using a fixed session ID
nexus run agent.py --session-id user-123 --input "What did we discuss last time?"
```

### REPL commands

When running interactively:

| Command | Description |
|---------|-------------|
| `exit` or `quit` | End the session |
| `/cost` | Show cost summary for this session |
| `/memory` | Show current working memory window |
| `/clear` | Clear working memory (start fresh) |

### Output format

```
[nexus] Session: abc12345
[nexus] Agent: research-agent | Model: claude-sonnet-4-6
> What is the capital of Brazil?
Brasília is the capital of Brazil, established in 1960.

[cost] Input: 42 tokens | Output: 18 tokens | Total: $0.000018
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Agent file not found or invalid |
| `2` | Agent raised an unhandled error |
| `3` | Budget exceeded |

---

## nexus eval

Run an evaluation suite and report results.

```bash
nexus eval [OPTIONS] SUITE_FILE
```

| Argument/Option | Default | Description |
|-----------------|---------|-------------|
| `SUITE_FILE` | (required) | Path to Python file defining `EvalSuite` |
| `--format TEXT` | `"text"` | Output format: `text`, `json`, `junit` |
| `--output TEXT` | `None` | Write report to file (stdout if omitted) |
| `--fail-under FLOAT` | `None` | Exit code 1 if pass rate < threshold (0.0–1.0) |

The suite file must export a function `create_suite() -> EvalSuite`.

### Examples

```bash
# Run suite with text output
nexus eval tests/eval_suite.py

# JSON output to file
nexus eval tests/eval_suite.py --format json --output results.json

# JUnit XML for CI
nexus eval tests/eval_suite.py --format junit --output results.xml

# Fail if pass rate below 90% (CI gate)
nexus eval tests/eval_suite.py --fail-under 0.9
echo $?   # 0 = passed, 1 = below threshold
```

### Text output format

```
Suite: research-agent-suite
Running 12 cases...

  [PASS] basic_answer             (0.8s, $0.0003)
  [PASS] uses_web_search          (1.2s, $0.0005)
  [FAIL] cites_sources            (0.9s, $0.0004)
         contains("http"): not found in output
  [PASS] no_pii_in_output         (0.7s, $0.0002)
  ...

Results: 10/12 passed (83.3%)
Total cost: $0.0041
Avg duration: 0.94s
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | All cases passed (or pass rate >= `--fail-under`) |
| `1` | Some cases failed or pass rate below threshold |
| `2` | Suite file not found or invalid |

---

## nexus memory

Inspect and manage agent memory.

```bash
nexus memory COMMAND [OPTIONS] AGENT_ID
```

### nexus memory inspect

Show stored memories for an agent.

```bash
nexus memory inspect [OPTIONS] AGENT_ID
```

| Option | Default | Description |
|--------|---------|-------------|
| `AGENT_ID` | (required) | Agent identifier |
| `--session TEXT` | `None` | Filter to a specific session ID |
| `--type TEXT` | `"all"` | Memory type: `episodic`, `semantic`, `all` |

```bash
# All memories for an agent
nexus memory inspect research-agent

# Episodic memories for a specific session
nexus memory inspect research-agent --session session-42 --type episodic

# Semantic facts
nexus memory inspect research-agent --type semantic
```

Output:

```
Agent: research-agent
Type: episodic  Session: session-42

  [2025-01-15 12:34] (trust=0.60, importance=0.72)
  "User asked about pricing for enterprise plan."

  [2025-01-15 12:36] (trust=0.70, importance=0.45)
  "Research on agentic AI frameworks completed."

2 episodic records found.
```

### nexus memory clear

Delete stored memories.

```bash
nexus memory clear [OPTIONS] AGENT_ID
```

| Option | Default | Description |
|--------|---------|-------------|
| `AGENT_ID` | (required) | Agent identifier |
| `--session TEXT` | `None` | Limit deletion to a specific session |
| `--type TEXT` | `"all"` | Memory type: `episodic`, `semantic`, `all` |
| `--yes` | `False` | Skip confirmation prompt |

```bash
# Clear all memories (with confirmation)
nexus memory clear research-agent

# Clear episodic memories for one session (no confirmation)
nexus memory clear research-agent --session session-42 --type episodic --yes
```

### nexus memory stats

Show summary statistics.

```bash
nexus memory stats AGENT_ID
```

Output:

```
Agent: research-agent

  Episodic records:  147
  Semantic facts:     32
  Oldest record:  2025-01-10 09:15
  Newest record:  2025-01-15 14:22
  Avg trust score:   0.68
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Agent not found or Dapr unavailable |

---

## nexus cost

Show cost summary for recent agent runs.

```bash
nexus cost [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--agent TEXT` | `None` | Filter by agent ID |
| `--session TEXT` | `None` | Filter by session ID |
| `--last INT` | `20` | Show last N cost events |
| `--log-file TEXT` | `".nexus/cost_log.jsonl"` | Path to JSONL cost log |

### Examples

```bash
# Show last 20 cost events
nexus cost

# Show costs for a specific agent
nexus cost --agent research-agent --last 50

# Show costs for a specific session
nexus cost --session session-42
```

Output:

```
Cost summary (last 20 runs)

  2025-01-15 14:22  research-agent  session-42  claude-sonnet-4-6  $0.0023  2.1s
  2025-01-15 13:55  research-agent  session-41  claude-sonnet-4-6  $0.0018  1.8s
  2025-01-15 13:10  hello-agent     session-40  claude-haiku-4-5   $0.0002  0.5s
  ...

Total (20 runs): $0.0241
Avg per run:     $0.0012
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Log file not found |

---

## nexus dev

Start agent in development mode with auto-reload and live cost/trace output.

```bash
nexus dev [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config TEXT` | `"nexus.yaml"` | Path to nexus.yaml |
| `--port INT` | `8000` | Agent HTTP server port |

`nexus dev` validates `nexus.yaml` on startup and on every file change.

```bash
# Start dev mode (watches current directory)
nexus dev

# Use custom config and port
nexus dev --config staging.yaml --port 8001
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Clean exit (Ctrl+C) |
| `1` | Config validation failed or Dapr unavailable |
