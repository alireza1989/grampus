# Code Analysis Tools (H45)

## Overview

Five tools give agents targeted structural queries over Python codebases. Rather than dumping
raw source code into the context window, agents ask specific questions:

- **"What functions and classes are in this file?"** → `analyze_file`
- **"Where is `MyClass` defined?"** → `find_symbol`
- **"What public API does this package expose?"** → `summarize_structure`
- **"Are there lint issues here?"** → `lint_code`
- **"Are there type errors here?"** → `check_types`

Research basis: arXiv 2603.27277 (Codebase-Memory, March 2026) showed that structured code
analysis tools reduce agent token usage by 10x and tool call count by 2.1x versus
grep+file-read exploration patterns.

All five tools are zero-dependency (stdlib `ast` only for the AST engine; `ruff` and `mypy`
are already in the Nexus dev toolchain). They never raise — all errors are returned as
`{"ok": false, "code": "..."}` payloads.

---

## `analyze_file` — Full structural analysis

Returns all functions, classes, imports, and their metadata for a single `.py` file.

**Input:**
```json
{"path": "src/nexus/tools/library/calculator.py"}
```

**Output (compact):**
```json
{
  "ok": true,
  "result": {
    "path": "/abs/path/calculator.py",
    "module_name": "nexus.tools.library.calculator",
    "docstring": "AST-based safe calculator tool.",
    "imports": [{"module": "ast", "names": [], "kind": "stdlib", "line": 3}],
    "functions": [
      {
        "name": "calculator",
        "qualname": "calculator",
        "line_start": 15,
        "line_end": 42,
        "is_async": true,
        "is_method": false,
        "decorators": [],
        "parameters": [{"name": "expression", "annotation": "str", "has_default": false}],
        "return_annotation": "dict[str, Any]",
        "docstring": "Evaluate an arithmetic expression.",
        "cyclomatic_complexity": 4,
        "complexity_rating": "low"
      }
    ],
    "classes": [],
    "total_lines": 55,
    "has_syntax_error": false,
    "syntax_error_message": null
  }
}
```

**Field reference:**

| Field | Description |
|---|---|
| `functions` | Top-level functions only (not methods — those are in `classes[].methods`) |
| `qualname` | `"ClassName.method"` for methods, `"function_name"` for top-level |
| `cyclomatic_complexity` | McCabe CC — branch count + 1 |
| `complexity_rating` | `low`, `medium`, `high`, or `very_high` |
| `has_syntax_error` | `true` if the file could not be parsed; rest of the payload is empty |

On syntax error, `ok` is still `true` — check `has_syntax_error` in the payload.

---

## Complexity scores

| Rating | CC range | Action |
|---|---|---|
| `low` | 1–5 | No action needed |
| `medium` | 6–10 | Consider refactoring if the function grows |
| `high` | 11–15 | Refactor: extract helpers, simplify conditionals |
| `very_high` | 16+ | Refactor immediately — testing is unreliable at this complexity |

---

## `lint_code` — Ruff linter

Runs [Ruff](https://docs.astral.sh/ruff/) on a file or directory. Returns structured findings,
not raw text.

**Input:**
```json
{"path": "src/nexus/tools/library/calculator.py", "select": ["E", "F"]}
```

**Output:**
```json
{
  "ok": true,
  "result": {
    "available": true,
    "findings": [
      {
        "filename": "calculator.py",
        "row": 12,
        "col": 1,
        "rule_id": "F401",
        "message": "'os' imported but unused",
        "fix_available": true
      }
    ],
    "total": 1
  }
}
```

**Graceful degradation:** When Ruff is not installed, `available` is `false` and a top-level
`hint` field contains the install command. The `ok` key is still `true` — no exception is raised.

**Using `select`:** Pass Ruff rule prefixes to narrow the check, e.g. `["E", "F"]` for pycodestyle
and pyflakes rules only, or `["S"]` for security checks. Omit to use Ruff's default selection.

---

## `check_types` — mypy type checker

Runs mypy on a `.py` file and returns structured type errors.

**Input:**
```json
{"path": "src/nexus/tools/library/calculator.py"}
```

**Output:**
```json
{
  "ok": true,
  "result": {
    "available": true,
    "errors": [
      {
        "filename": "calculator.py",
        "line": 27,
        "col": 4,
        "error_code": "return-value",
        "message": "Incompatible return value type (got \"str\", expected \"int\")",
        "severity": "error"
      }
    ],
    "total_errors": 1,
    "total_warnings": 0
  }
}
```

**Graceful degradation:** Same as `lint_code` — `available=false` + `hint` when mypy is absent.

**Note:** In production environments where mypy is not installed as a dev dependency, this tool
will return `available=false`. Install with `pip install mypy` or use the `nexus-ai[dev]` extras.

---

## `find_symbol` — Symbol search

Scans all `.py` files in a directory tree for a function, class, or method with a given name.

**Input:**
```json
{"name": "EpisodicMemory", "directory": "src/nexus/memory", "max_files": 200}
```

**Output:**
```json
{
  "ok": true,
  "result": {
    "name": "EpisodicMemory",
    "total": 1,
    "matches": [
      {
        "path": "/abs/path/src/nexus/memory/episodic.py",
        "line": 34,
        "kind": "class",
        "qualname": "EpisodicMemory",
        "name": "EpisodicMemory",
        "signature": "class EpisodicMemory(BaseModel)"
      }
    ]
  }
}
```

**Performance note:** Files larger than 500 KB are skipped. The `max_files` cap (default 200)
keeps search time interactive. For large monorepos, pass a more specific `directory`.

---

## `summarize_structure` — Lightweight module index

Returns a one-line-per-module summary of all public functions and classes. Much cheaper than
calling `analyze_file` on every module.

**Input:**
```json
{"directory": "src/nexus/memory", "max_files": 200}
```

**Output:**
```json
{
  "ok": true,
  "result": {
    "total_modules": 5,
    "modules": [
      {
        "path": "/abs/path/src/nexus/memory/episodic.py",
        "module_name": "nexus.memory.episodic",
        "public_functions": ["store", "retrieve", "delete"],
        "public_classes": ["EpisodicMemory", "EpisodicRecord"],
        "import_count": 8,
        "has_syntax_error": false
      }
    ]
  }
}
```

Private names (prefixed with `_`) are excluded. Use this as a **"start here"** tool before
deciding which specific file deserves a full `analyze_file` call.

---

## Integration with agents — Investigation workflow

Pseudocode for an agent investigating a bug in an unknown codebase:

```python
# Step 1: Get the lay of the land
structure = await summarize_structure_tool(directory="src/nexus/memory")
# → learn which files expose EpisodicMemory, store, retrieve

# Step 2: Find where the specific class is defined
matches = await find_symbol_tool(name="EpisodicMemory", directory="src/nexus/memory")
# → src/nexus/memory/episodic.py, line 34

# Step 3: Deep-dive the specific file
detail = await analyze_file_tool(path="src/nexus/memory/episodic.py")
# → all methods, their complexity, parameters, return types

# Step 4: Check for lint / type issues in the same file
lint = await lint_code_tool(path="src/nexus/memory/episodic.py")
types = await check_types_tool(path="src/nexus/memory/episodic.py")
# → structured findings rather than raw text output
```

This workflow uses 4 tool calls and injects only targeted structural data into the context
window — versus dozens of file-read calls that dump thousands of lines of source code.
