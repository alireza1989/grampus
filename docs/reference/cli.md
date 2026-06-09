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

---

## nexus state

Manage agent state snapshots — export, inspect, and restore full session state.

```bash
nexus state COMMAND [OPTIONS]
```

### nexus state export

Export the state of an agent session to a portable JSON snapshot.

```bash
nexus state export [OPTIONS] AGENT_ID
```

| Argument/Option | Default | Description |
|-----------------|---------|-------------|
| `AGENT_ID` | (required) | Agent identifier |
| `--session TEXT` | (latest session) | Session ID to export |
| `--output TEXT` | `"<agent_id>_<session_id>.json"` | Output file path |
| `--tag KEY=VALUE` | (repeatable) | Metadata tag attached to the snapshot |

```bash
# Export the latest session for research-bot
nexus state export research-bot --output snapshot.json

# Export a specific session with tags
nexus state export research-bot \
  --session ses_abc123 \
  --output snapshot.json \
  --tag env=production \
  --tag reason=incident-review
```

### nexus state import

Restore a previously exported snapshot into Dapr state.

```bash
nexus state import [OPTIONS] FILE
```

| Argument/Option | Default | Description |
|-----------------|---------|-------------|
| `FILE` | (required) | Path to the snapshot JSON file |
| `--dry-run` | `False` | Print what would be restored without writing any state |

```bash
# Preview changes without writing
nexus state import snapshot.json --dry-run

# Restore the snapshot
nexus state import snapshot.json
```

### nexus state show

Inspect a snapshot file without restoring it.

```bash
nexus state show [OPTIONS] [FILE]
```

| Argument/Option | Default | Description |
|-----------------|---------|-------------|
| `FILE` | `None` | Path to the snapshot JSON file (stdin if omitted) |
| `--format TEXT` | `"table"` | Output format: `table`, `json` |

```bash
# Human-readable summary
nexus state show snapshot.json --format table

# Full JSON dump
nexus state show snapshot.json --format json
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | File not found, invalid snapshot format, or Dapr unavailable |
| `2` | Session not found for the given agent |

---

## nexus alerts

Manage cost alert rules and notification channels.

```bash
nexus alerts COMMAND [OPTIONS]
```

### nexus alerts list

Show all configured alert rules.

```bash
nexus alerts list
```

Output:

```
ID           NAME                THRESHOLD        TYPE              SEVERITY   ENABLED
rule_abc123  session-budget      $0.10            per_session_usd   warning    yes
rule_def456  daily-spend         $5.00            per_day_usd       critical   yes
rule_ghi789  per-run-spike       $0.25            per_run_usd       warning    no
```

### nexus alerts add

Create a new alert rule.

```bash
nexus alerts add [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--name TEXT` | (required) | Unique rule name |
| `--threshold-usd FLOAT` | (required) | USD threshold that triggers the alert |
| `--threshold-type TEXT` | (required) | `per_run_usd`, `per_session_usd`, `per_hour_usd`, `per_day_usd`, `per_month_usd` |
| `--severity TEXT` | `"warning"` | Alert severity: `info`, `warning`, `critical` |
| `--agent-id TEXT` | `None` | Scope to a specific agent (None = all agents) |
| `--cooldown INT` | `3600` | Minimum seconds between repeated fires for this rule |

```bash
nexus alerts add \
  --name "daily-spend" \
  --threshold-usd 5.00 \
  --threshold-type per_day_usd \
  --severity critical \
  --agent-id research-bot \
  --cooldown 86400
```

### nexus alerts remove

Delete an alert rule by ID.

```bash
nexus alerts remove RULE_ID
```

```bash
nexus alerts remove rule_abc123
```

### nexus alerts enable / disable

Enable or disable a rule without deleting it.

```bash
nexus alerts enable  RULE_ID
nexus alerts disable RULE_ID
```

```bash
nexus alerts disable rule_ghi789   # pause a noisy rule temporarily
nexus alerts enable  rule_ghi789   # re-enable it
```

### nexus alerts test

Fire a test notification for a rule to verify your notification channels are working.

```bash
nexus alerts test RULE_ID
```

```bash
nexus alerts test rule_abc123
# Sends a test alert to all configured notification channels
# Prints: "Test alert sent to 2 channels (slack, log)"
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Rule not found or server unavailable |
| `2` | Invalid option value |

---

## nexus playground

Interactive prompt playground for testing and comparing LLM responses.

```bash
nexus playground COMMAND [OPTIONS]
```

### nexus playground start

Launch the interactive REPL.

```bash
nexus playground start [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model TEXT` | `"claude-haiku-4-5"` | Starting model |
| `--system TEXT` | `None` | System prompt string |
| `--system-file PATH` | `None` | Load system prompt from a file |
| `--load TEXT` | `None` | Resume a previously saved session by name |

```bash
# Start with defaults
nexus playground start

# Start with a specific model and system prompt
nexus playground start --model gpt-4o-mini --system "You are a Python tutor."

# Resume a saved session
nexus playground start --load python-tutor
```

Inside the REPL, use `/help` to list all available commands.

### nexus playground run

Run a single prompt and exit (non-interactive).

```bash
nexus playground run [OPTIONS] MESSAGE
```

| Argument/Option | Default | Description |
|-----------------|---------|-------------|
| `MESSAGE` | (required) | The user message to send |
| `--model TEXT` | `"claude-haiku-4-5"` | Model to use |
| `--system TEXT` | `None` | System prompt |
| `--no-stream` | `False` | Disable streaming output |

```bash
nexus playground run "What is the capital of France?" --model claude-haiku-4-5
nexus playground run "Explain recursion." --model gpt-4o-mini --no-stream
```

### nexus playground compare

Run the same message against multiple models simultaneously.

```bash
nexus playground compare [OPTIONS] MESSAGE
```

| Argument/Option | Default | Description |
|-----------------|---------|-------------|
| `MESSAGE` | (required) | The user message to send to all models |
| `--models TEXT` | (required) | Comma-separated list of model names |
| `--system TEXT` | `None` | System prompt applied to all models |

```bash
nexus playground compare "Explain async/await." \
  --models claude-haiku-4-5,gpt-4o-mini,llama3.2
```

### nexus playground sessions

List all saved playground sessions.

```bash
nexus playground sessions
```

Output:

```
NAME             MODEL              TURNS  COST      SAVED
python-tutor     claude-haiku-4-5   8      $0.0012   2026-06-01 14:22
billing-tests    gpt-4o-mini        3      $0.0003   2026-05-30 09:15
```

### nexus playground show

Display the contents of a saved session.

```bash
nexus playground show [OPTIONS] NAME
```

| Option | Default | Description |
|--------|---------|-------------|
| `NAME` | (required) | Saved session name |
| `--format TEXT` | `"transcript"` | Output format: `transcript`, `json` |

```bash
nexus playground show python-tutor
nexus playground show python-tutor --format json
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success or clean REPL exit |
| `1` | Model provider not configured or session not found |

---

## nexus redteam

Run an adversarial red-team campaign against an agent to find security vulnerabilities before attackers do.

```bash
nexus redteam [OPTIONS] AGENT_FILE
```

| Argument/Option | Default | Description |
|-----------------|---------|-------------|
| `AGENT_FILE` | (required) | Path to agent adapter Python file |
| `--categories/-c TEXT` | all | Attack categories to run (repeatable): `prompt_injection`, `jailbreak`, `reasoning_hijack`, `memory_poison`, `tool_misuse`, `excessive_agency` |
| `--count/-n INT` | `5` | Number of payloads per strategy |
| `--output/-o TEXT` | `"text"` | Report format: `text`, `json` |
| `--stop-on-critical` | `False` | Halt campaign immediately on first CRITICAL finding |
| `--model TEXT` | `None` | Model ID for LLM-based judge + adaptive mutation (e.g. `claude-sonnet-4-6`) |

The agent file must expose two functions:

- `get_agent_config() -> RedTeamTargetConfig` — agent metadata and capability flags
- `async run_conversation(messages: list[tuple[str, str]]) -> str` — stateless or stateful conversation handler

### Examples

```bash
# Full campaign, all categories, text output
nexus redteam agents/my_agent.py

# Specific categories only
nexus redteam agents/my_agent.py --categories prompt_injection jailbreak

# Fast CI scan: 3 payloads per strategy, stop on CRITICAL
nexus redteam agents/my_agent.py --stop-on-critical --count 3

# Thorough pre-release audit with LLM judge
nexus redteam agents/my_agent.py --model claude-sonnet-4-6 --count 10

# JSON output for downstream processing
nexus redteam agents/my_agent.py --output json > redteam-report.json
```

### Attack categories

| Category | OWASP | What it tests |
|----------|-------|---------------|
| `prompt_injection` | ASI01:2026 | Direct and indirect instruction overrides |
| `jailbreak` | ASI01:2026 | Roleplay frames, encoding tricks, logic traps |
| `reasoning_hijack` | ASI01:2026 | Multi-turn context manipulation |
| `memory_poison` | ASI06:2026 | Persistent memory write injection |
| `tool_misuse` | ASI02:2026 | Infinite loops, chain escapes, enumeration |
| `excessive_agency` | LLM #2 | Scope escalation, implicit permission exploits |

### Report format

```
SUMMARY
  Total attacks:     30
  Successful:        4
  Attack success:    13.3%

SEVERITY BREAKDOWN
  HIGH       3
  MEDIUM     1

FINDINGS
  [HIGH] Prompt Injection — Direct Injection
    Category:    prompt_injection
    OWASP:       ASI01:2026
    Occurrences: 3
    Recommendation: Raise PromptInjectionDetector to STRICT...
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | No CRITICAL or HIGH findings |
| `1` | One or more CRITICAL or HIGH findings — suitable for CI gates |
| `2` | Agent file missing required functions or invalid category |
