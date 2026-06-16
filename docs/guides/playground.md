# Prompt Playground

`grampus playground` is an interactive CLI for testing prompts, comparing model responses, and iterating on system prompts before wiring an agent into production. Think of it as a REPL for prompt engineering — with cost tracking, session history, and direct export to eval cases.

---

## Quick start

```bash
$ grampus playground start --model claude-haiku-4-5
```

```
Grampus Prompt Playground
Model: claude-haiku-4-5  |  Cost: $0.0000
Type /help for commands. Ctrl+C to exit.

[claude-haiku-4-5] > What is the capital of France?

╭─── claude-haiku-4-5 ──────────────────────────────────────────────────╮
│ Paris is the capital of France.                                         │
╰─ ↑45 ↓8 tokens · $0.0001 · 0.8s ─────────────────────────────────────╯

[claude-haiku-4-5] >
```

Responses stream in real time. Each response shows input/output token count, cost, and latency.

---

## REPL commands

| Command | Description |
|---------|-------------|
| `/model <name>` | Switch to a different model for subsequent turns |
| `/models` | List available model names from configured providers |
| `/system <text>` | Set the system prompt for this session |
| `/system file:<path>` | Load a system prompt from a local file |
| `/compare <model2> [model3]` | Run the last user message against additional models concurrently |
| `/cost` | Show accumulated cost for this session |
| `/reset` | Clear conversation history (system prompt preserved) |
| `/save [name]` | Save the current session to `~/.grampus/playground/` |
| `/load <name>` | Load a previously saved session |
| `/sessions` | List all saved sessions |
| `/export [path]` | Export the last turn as an `EvalCase` JSON file |
| `/version save <name>` | Save the current system prompt as a versioned entry |
| `/version diff <v1> <v2>` | Diff two saved system prompt versions |
| `/help` | Show all available commands |
| `/exit` | Exit the REPL |

---

## One-shot mode

Run a single prompt and exit — useful in scripts:

```bash
$ grampus playground run "What is the capital of France?" --model gpt-4o-mini
Paris is the capital of France.
↑52 ↓5 tokens · $0.00003 · 0.3s
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model TEXT` | `claude-haiku-4-5` | Model to use |
| `--system TEXT` | `None` | System prompt string |
| `--no-stream` | `False` | Disable streaming (wait for full response) |

---

## Comparing models

Test the same prompt across multiple models simultaneously to compare quality and cost:

```bash
$ grampus playground compare "Explain async/await in Python in one paragraph" \
    --models claude-haiku-4-5,gpt-4o-mini,llama3.2
```

```
Running on 3 models concurrently...

╭─── claude-haiku-4-5 ─────────────────────────────────────────────────────╮
│ async/await is Python's syntax for writing asynchronous code that runs    │
│ concurrently without blocking...                                           │
╰─ ↑62 ↓89 tokens · $0.0001 · 0.9s ────────────────────────────────────────╯

╭─── gpt-4o-mini ──────────────────────────────────────────────────────────╮
│ In Python, async/await enables non-blocking I/O operations by allowing    │
│ functions to pause execution while waiting for results...                  │
╰─ ↑62 ↓94 tokens · $0.0001 · 0.7s ────────────────────────────────────────╯

╭─── llama3.2 (ollama) ────────────────────────────────────────────────────╮
│ The async/await pattern lets Python programs handle multiple operations    │
│ simultaneously by yielding control during I/O waits...                     │
╰─ ↑62 ↓81 tokens · $0.0000 · 1.2s ────────────────────────────────────────╯

Total cost: $0.0002  |  Fastest: gpt-4o-mini (0.7s)  |  Cheapest: llama3.2 ($0.00)
```

You can also run `/compare gpt-4o-mini llama3.2` from inside the REPL to compare the most recent message against other models without retyping the prompt.

---

## Saving and reusing sessions

Sessions save everything: conversation history, system prompt, model choice, and cost summary.

```bash
# From inside the REPL
[claude-haiku-4-5] > /save python-tutor
Session saved: python-tutor

# List saved sessions
$ grampus playground sessions
NAME             MODEL              TURNS  COST      SAVED
python-tutor     claude-haiku-4-5   8      $0.0012   2026-06-01 14:22
billing-tests    gpt-4o-mini        3      $0.0003   2026-05-30 09:15

# Resume a session
$ grampus playground start --load python-tutor
```

Sessions are stored as JSON files in `~/.grampus/playground/`. They are portable — you can commit them to your repository as regression fixtures.

---

## Exporting to eval cases

Any turn can be exported as an `EvalCase` JSON that feeds directly into an `EvalSuite`:

```bash
# Inside the REPL, after getting a response you want to test
[claude-haiku-4-5] > What is the capital of Brazil?
╭─── claude-haiku-4-5 ─────────╮
│ Brasília is the capital ...   │
╰─ ↑42 ↓12 tokens · $0.0001 ───╯

[claude-haiku-4-5] > /export cases/capital_brazil.json
Exported: cases/capital_brazil.json
```

The exported file is an `EvalCase` with the user message pre-filled and a `contains` assertion based on the observed response:

```json
{
  "name": "capital_brazil",
  "input": "What is the capital of Brazil?",
  "assertions": [
    {"type": "contains", "expected": "Brasília"}
  ]
}
```

Load it in your eval suite:

```python
import json
from grampus.evaluation.suite import EvalCase, EvalSuite

with open("cases/capital_brazil.json") as f:
    data = json.load(f)

case = EvalCase(**data)
suite.add_case(case)
```

---

## See also

- **[Evaluation guide →](evaluation.md)** — Run exported cases in a full eval suite
- **[Model providers →](model-providers.md)** — Configure providers for the playground
- **[Cost Management →](cost-management.md)** — Track session costs across playground runs
