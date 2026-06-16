# CLI Reference

The `grampus` CLI is the primary interface for initializing, running, evaluating, and monitoring Grampus agents.

```bash
grampus --version    # grampus 0.1.0
grampus --help       # show all commands
```

---

## grampus init

Scaffold a new Grampus project.

```bash
grampus init [OPTIONS] [NAME]
```

| Option | Default | Description |
|--------|---------|-------------|
| `NAME` | (prompted) | Project directory name |
| `--name TEXT` | `"grampus-agent"` | Agent name used in config |
| `--template TEXT` | `"simple"` | Project template: `simple`, `crew`, `rag` |
| `--output-dir TEXT` | `"."` | Parent directory for the new project |

### Templates

| Template | Creates | Best for |
|----------|---------|----------|
| `simple` | Single agent with one tool | Getting started, learning Grampus |
| `crew` | Three-agent crew (researcher, critic, writer) | Multi-agent workflows |
| `rag` | RAG agent with document retrieval tool | Question answering over documents |

### Examples

```bash
# Create a simple agent in the current directory
grampus init my-agent

# Create a crew agent in a specific directory
grampus init --template crew --output-dir ~/projects research-crew

# Non-interactive (all defaults)
grampus init --name my-agent --template simple --output-dir .
```

### Generated files

```
my-agent/
├── agent.py              # Agent code with create_runner() and create_agent_def()
├── grampus.yaml            # Configuration
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

## grampus run

Start an agent. Without `--input`, starts an interactive REPL. With `--input`, runs once and exits.

```bash
grampus run [OPTIONS] AGENT_FILE
```

| Argument/Option | Default | Description |
|-----------------|---------|-------------|
| `AGENT_FILE` | (required) | Path to agent Python file |
| `--config TEXT` | `"grampus.yaml"` | Path to grampus.yaml configuration file |
| `--session-id TEXT` | (auto-generated UUID) | Session identifier for memory persistence |
| `--input TEXT` | `None` | Single-shot input; omit for interactive REPL |

The agent file must export two functions:

- `create_runner() -> AgentRunner` — constructs the runner with all dependencies
- `create_agent_def() -> AgentDefinition` — returns the agent blueprint

### Examples

```bash
# Interactive REPL
grampus run agent.py

# Single-shot (useful in scripts and CI)
grampus run agent.py --input "What is the capital of France?"

# Use a specific config file
grampus run agent.py --config config/production.yaml --input "Hello"

# Persist memory across runs using a fixed session ID
grampus run agent.py --session-id user-123 --input "What did we discuss last time?"
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
[grampus] Session: abc12345
[grampus] Agent: research-agent | Model: claude-sonnet-4-6
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

## grampus eval

Run an evaluation suite and report results.

```bash
grampus eval [OPTIONS] SUITE_FILE
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
grampus eval tests/eval_suite.py

# JSON output to file
grampus eval tests/eval_suite.py --format json --output results.json

# JUnit XML for CI
grampus eval tests/eval_suite.py --format junit --output results.xml

# Fail if pass rate below 90% (CI gate)
grampus eval tests/eval_suite.py --fail-under 0.9
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

## grampus memory

Inspect and manage agent memory.

```bash
grampus memory COMMAND [OPTIONS] AGENT_ID
```

### grampus memory inspect

Show stored memories for an agent.

```bash
grampus memory inspect [OPTIONS] AGENT_ID
```

| Option | Default | Description |
|--------|---------|-------------|
| `AGENT_ID` | (required) | Agent identifier |
| `--session TEXT` | `None` | Filter to a specific session ID |
| `--type TEXT` | `"all"` | Memory type: `episodic`, `semantic`, `all` |

```bash
# All memories for an agent
grampus memory inspect research-agent

# Episodic memories for a specific session
grampus memory inspect research-agent --session session-42 --type episodic

# Semantic facts
grampus memory inspect research-agent --type semantic
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

### grampus memory clear

Delete stored memories.

```bash
grampus memory clear [OPTIONS] AGENT_ID
```

| Option | Default | Description |
|--------|---------|-------------|
| `AGENT_ID` | (required) | Agent identifier |
| `--session TEXT` | `None` | Limit deletion to a specific session |
| `--type TEXT` | `"all"` | Memory type: `episodic`, `semantic`, `all` |
| `--yes` | `False` | Skip confirmation prompt |

```bash
# Clear all memories (with confirmation)
grampus memory clear research-agent

# Clear episodic memories for one session (no confirmation)
grampus memory clear research-agent --session session-42 --type episodic --yes
```

### grampus memory stats

Show summary statistics.

```bash
grampus memory stats AGENT_ID
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

## grampus cost

Show cost summary for recent agent runs.

```bash
grampus cost [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--agent TEXT` | `None` | Filter by agent ID |
| `--session TEXT` | `None` | Filter by session ID |
| `--last INT` | `20` | Show last N cost events |
| `--log-file TEXT` | `".grampus/cost_log.jsonl"` | Path to JSONL cost log |

### Examples

```bash
# Show last 20 cost events
grampus cost

# Show costs for a specific agent
grampus cost --agent research-agent --last 50

# Show costs for a specific session
grampus cost --session session-42
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

## grampus dev

Start agent in development mode with auto-reload and live cost/trace output.

```bash
grampus dev [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config TEXT` | `"grampus.yaml"` | Path to grampus.yaml |
| `--port INT` | `8000` | Agent HTTP server port |

`grampus dev` validates `grampus.yaml` on startup and on every file change.

```bash
# Start dev mode (watches current directory)
grampus dev

# Use custom config and port
grampus dev --config staging.yaml --port 8001
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Clean exit (Ctrl+C) |
| `1` | Config validation failed or Dapr unavailable |

---

## grampus state

Manage agent state snapshots — export, inspect, and restore full session state.

```bash
grampus state COMMAND [OPTIONS]
```

### grampus state export

Export the state of an agent session to a portable JSON snapshot.

```bash
grampus state export [OPTIONS] AGENT_ID
```

| Argument/Option | Default | Description |
|-----------------|---------|-------------|
| `AGENT_ID` | (required) | Agent identifier |
| `--session TEXT` | (latest session) | Session ID to export |
| `--output TEXT` | `"<agent_id>_<session_id>.json"` | Output file path |
| `--tag KEY=VALUE` | (repeatable) | Metadata tag attached to the snapshot |

```bash
# Export the latest session for research-bot
grampus state export research-bot --output snapshot.json

# Export a specific session with tags
grampus state export research-bot \
  --session ses_abc123 \
  --output snapshot.json \
  --tag env=production \
  --tag reason=incident-review
```

### grampus state import

Restore a previously exported snapshot into Dapr state.

```bash
grampus state import [OPTIONS] FILE
```

| Argument/Option | Default | Description |
|-----------------|---------|-------------|
| `FILE` | (required) | Path to the snapshot JSON file |
| `--dry-run` | `False` | Print what would be restored without writing any state |

```bash
# Preview changes without writing
grampus state import snapshot.json --dry-run

# Restore the snapshot
grampus state import snapshot.json
```

### grampus state show

Inspect a snapshot file without restoring it.

```bash
grampus state show [OPTIONS] [FILE]
```

| Argument/Option | Default | Description |
|-----------------|---------|-------------|
| `FILE` | `None` | Path to the snapshot JSON file (stdin if omitted) |
| `--format TEXT` | `"table"` | Output format: `table`, `json` |

```bash
# Human-readable summary
grampus state show snapshot.json --format table

# Full JSON dump
grampus state show snapshot.json --format json
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | File not found, invalid snapshot format, or Dapr unavailable |
| `2` | Session not found for the given agent |

---

## grampus alerts

Manage cost alert rules and notification channels.

```bash
grampus alerts COMMAND [OPTIONS]
```

### grampus alerts list

Show all configured alert rules.

```bash
grampus alerts list
```

Output:

```
ID           NAME                THRESHOLD        TYPE              SEVERITY   ENABLED
rule_abc123  session-budget      $0.10            per_session_usd   warning    yes
rule_def456  daily-spend         $5.00            per_day_usd       critical   yes
rule_ghi789  per-run-spike       $0.25            per_run_usd       warning    no
```

### grampus alerts add

Create a new alert rule.

```bash
grampus alerts add [OPTIONS]
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
grampus alerts add \
  --name "daily-spend" \
  --threshold-usd 5.00 \
  --threshold-type per_day_usd \
  --severity critical \
  --agent-id research-bot \
  --cooldown 86400
```

### grampus alerts remove

Delete an alert rule by ID.

```bash
grampus alerts remove RULE_ID
```

```bash
grampus alerts remove rule_abc123
```

### grampus alerts enable / disable

Enable or disable a rule without deleting it.

```bash
grampus alerts enable  RULE_ID
grampus alerts disable RULE_ID
```

```bash
grampus alerts disable rule_ghi789   # pause a noisy rule temporarily
grampus alerts enable  rule_ghi789   # re-enable it
```

### grampus alerts test

Fire a test notification for a rule to verify your notification channels are working.

```bash
grampus alerts test RULE_ID
```

```bash
grampus alerts test rule_abc123
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

## grampus playground

Interactive prompt playground for testing and comparing LLM responses.

```bash
grampus playground COMMAND [OPTIONS]
```

### grampus playground start

Launch the interactive REPL.

```bash
grampus playground start [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model TEXT` | `"claude-haiku-4-5"` | Starting model |
| `--system TEXT` | `None` | System prompt string |
| `--system-file PATH` | `None` | Load system prompt from a file |
| `--load TEXT` | `None` | Resume a previously saved session by name |

```bash
# Start with defaults
grampus playground start

# Start with a specific model and system prompt
grampus playground start --model gpt-4o-mini --system "You are a Python tutor."

# Resume a saved session
grampus playground start --load python-tutor
```

Inside the REPL, use `/help` to list all available commands.

### grampus playground run

Run a single prompt and exit (non-interactive).

```bash
grampus playground run [OPTIONS] MESSAGE
```

| Argument/Option | Default | Description |
|-----------------|---------|-------------|
| `MESSAGE` | (required) | The user message to send |
| `--model TEXT` | `"claude-haiku-4-5"` | Model to use |
| `--system TEXT` | `None` | System prompt |
| `--no-stream` | `False` | Disable streaming output |

```bash
grampus playground run "What is the capital of France?" --model claude-haiku-4-5
grampus playground run "Explain recursion." --model gpt-4o-mini --no-stream
```

### grampus playground compare

Run the same message against multiple models simultaneously.

```bash
grampus playground compare [OPTIONS] MESSAGE
```

| Argument/Option | Default | Description |
|-----------------|---------|-------------|
| `MESSAGE` | (required) | The user message to send to all models |
| `--models TEXT` | (required) | Comma-separated list of model names |
| `--system TEXT` | `None` | System prompt applied to all models |

```bash
grampus playground compare "Explain async/await." \
  --models claude-haiku-4-5,gpt-4o-mini,llama3.2
```

### grampus playground sessions

List all saved playground sessions.

```bash
grampus playground sessions
```

Output:

```
NAME             MODEL              TURNS  COST      SAVED
python-tutor     claude-haiku-4-5   8      $0.0012   2026-06-01 14:22
billing-tests    gpt-4o-mini        3      $0.0003   2026-05-30 09:15
```

### grampus playground show

Display the contents of a saved session.

```bash
grampus playground show [OPTIONS] NAME
```

| Option | Default | Description |
|--------|---------|-------------|
| `NAME` | (required) | Saved session name |
| `--format TEXT` | `"transcript"` | Output format: `transcript`, `json` |

```bash
grampus playground show python-tutor
grampus playground show python-tutor --format json
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success or clean REPL exit |
| `1` | Model provider not configured or session not found |

---

## grampus redteam

Run an adversarial red-team campaign against an agent to find security vulnerabilities before attackers do.

```bash
grampus redteam [OPTIONS] AGENT_FILE
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
grampus redteam agents/my_agent.py

# Specific categories only
grampus redteam agents/my_agent.py --categories prompt_injection jailbreak

# Fast CI scan: 3 payloads per strategy, stop on CRITICAL
grampus redteam agents/my_agent.py --stop-on-critical --count 3

# Thorough pre-release audit with LLM judge
grampus redteam agents/my_agent.py --model claude-sonnet-4-6 --count 10

# JSON output for downstream processing
grampus redteam agents/my_agent.py --output json > redteam-report.json
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
