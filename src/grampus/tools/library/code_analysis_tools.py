"""H45 registered tool functions for Python code analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from grampus.tools.library._base import err, ok
from grampus.tools.library.code_analyzer import (
    analyze_file,
    find_symbol,
    summarize_directory,
)
from grampus.tools.library.code_lint_runner import run_mypy, run_ruff

_MAX_ANALYZE_BYTES = 1_000_000  # 1 MB


async def analyze_file_tool(path: str) -> dict[str, Any]:
    """Full structural analysis of a Python (.py) file.

    Args:
        path: Path to the .py file to analyze.

    Returns:
        ModuleInfo dict with functions, classes, imports, complexity, and line ranges.
        On syntax error, returns ok with has_syntax_error=True in the payload.
    """
    p = Path(path)
    if not p.exists():
        return err(f"File not found: {path!r}", code="FILE_NOT_FOUND")
    if not p.is_file():
        return err(f"Not a file: {path!r}", code="NOT_A_FILE")
    if p.suffix != ".py":
        return err(f"Not a Python file: {path!r}", code="UNSUPPORTED_FORMAT")
    if p.stat().st_size > _MAX_ANALYZE_BYTES:
        return err(f"File too large (> 1 MB): {path!r}", code="FILE_TOO_LARGE")
    module_info = await analyze_file(path)
    return ok(module_info.model_dump())


async def lint_code_tool(
    path: str,
    select: list[str] | None = None,
) -> dict[str, Any]:
    """Run Ruff linter on a Python file or directory.

    Args:
        path: File or directory path to lint.
        select: Ruff rule codes to check (e.g. ['E', 'F', 'S']). Defaults to Ruff's selection.

    Returns:
        Structured findings with rule ID, message, line, column, and fix availability.
        When Ruff is not installed, returns ok with available=False and a hint.
    """
    p = Path(path)
    if not p.exists():
        return err(f"Path not found: {path!r}", code="FILE_NOT_FOUND")
    result = await run_ruff(path, select=select)
    payload: dict[str, Any] = {
        "available": result.available,
        "findings": [f.model_dump() for f in result.findings],
        "total": result.total,
    }
    if not result.available:
        return ok(payload, hint="Ruff not found on PATH. Install with: pip install ruff")
    return ok(payload)


async def check_types_tool(path: str) -> dict[str, Any]:
    """Run mypy type checker on a Python file.

    Args:
        path: Path to the .py file to type-check.

    Returns:
        Structured type errors with error codes, line numbers, and severity.
        When mypy is not installed, returns ok with available=False and a hint.
    """
    p = Path(path)
    if not p.exists():
        return err(f"Path not found: {path!r}", code="FILE_NOT_FOUND")
    result = await run_mypy(path)
    payload: dict[str, Any] = {
        "available": result.available,
        "errors": [e.model_dump() for e in result.errors],
        "total_errors": result.total_errors,
        "total_warnings": result.total_warnings,
    }
    if not result.available:
        return ok(payload, hint="mypy not found on PATH. Install with: pip install mypy")
    return ok(payload)


async def find_symbol_tool(
    name: str,
    directory: str = ".",
    max_files: int = 200,
) -> dict[str, Any]:
    """Locate where a function or class is defined across a directory tree.

    Args:
        name: Python identifier to search for (function, class, or method name).
        directory: Root directory to search under.
        max_files: Maximum number of files to scan (default 200).

    Returns:
        List of matches with file path, line number, kind, and full signature.
    """
    if not name.isidentifier():
        return err(f"Invalid Python identifier: {name!r}", code="INVALID_IDENTIFIER")
    d = Path(directory)
    if not d.exists():
        return err(f"Directory not found: {directory!r}", code="DIRECTORY_NOT_FOUND")
    matches = await find_symbol(name, directory, max_files=max_files)
    return ok({"matches": [m.model_dump() for m in matches], "total": len(matches), "name": name})


async def summarize_structure_tool(
    directory: str = ".",
    max_files: int = 200,
) -> dict[str, Any]:
    """Return a lightweight public-API index for all Python modules in a directory.

    Args:
        directory: Root directory to index.
        max_files: Maximum number of files to index (default 200).

    Returns:
        One StructureSummary per module: public function names, public class names,
        import count, and syntax-error flag.
    """
    d = Path(directory)
    if not d.exists():
        return err(f"Directory not found: {directory!r}", code="DIRECTORY_NOT_FOUND")
    if not d.is_dir():
        return err(f"Not a directory: {directory!r}", code="NOT_A_DIRECTORY")
    summaries = await summarize_directory(directory, max_files=max_files)
    return ok({"modules": [s.model_dump() for s in summaries], "total_modules": len(summaries)})
