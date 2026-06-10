"""H45 subprocess runners for Ruff and mypy — graceful degradation when not on PATH."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel


class RuffFinding(BaseModel):
    """A single Ruff linter finding."""

    filename: str
    row: int
    col: int
    rule_id: str
    message: str
    fix_available: bool


class RuffResult(BaseModel):
    """Aggregate result from a Ruff linter run."""

    available: bool
    findings: list[RuffFinding]
    total: int
    error: str | None = None


class MypyError(BaseModel):
    """A single mypy type-check finding."""

    filename: str
    line: int
    col: int | None
    error_code: str | None
    message: str
    severity: str


class MypyResult(BaseModel):
    """Aggregate result from a mypy type-check run."""

    available: bool
    errors: list[MypyError]
    total_errors: int
    total_warnings: int
    error: str | None = None


async def _exec(args: list[str], timeout: float) -> tuple[bytes, bytes, int]:
    """Run a subprocess, capture output, return (stdout, stderr, returncode).

    Raises FileNotFoundError if the binary is not on PATH.
    """
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    code = proc.returncode if proc.returncode is not None else -1
    return stdout, stderr, code


def _build_ruff_cmd(path: str, select: list[str] | None) -> list[str]:
    cmd = ["ruff", "check", path, "--output-format=json", "--no-fix"]
    if select:
        cmd.append("--select=" + ",".join(select))
    return cmd


def _parse_ruff_output(stdout: bytes) -> list[RuffFinding]:
    if not stdout.strip():
        return []
    try:
        items: list[dict[str, Any]] = json.loads(stdout.decode(errors="replace"))
    except (json.JSONDecodeError, ValueError):
        return []
    findings: list[RuffFinding] = []
    for item in items:
        try:
            loc: dict[str, Any] = item["location"]
            findings.append(
                RuffFinding(
                    filename=str(item["filename"]),
                    row=int(loc["row"]),
                    col=int(loc["column"]),
                    rule_id=str(item.get("code") or ""),
                    message=str(item["message"]),
                    fix_available=item.get("fix") is not None,
                )
            )
        except (KeyError, TypeError):
            continue
    return findings


async def run_ruff(path: str, *, select: list[str] | None = None) -> RuffResult:
    """Run Ruff linter on path and return structured findings.

    Degrades gracefully (available=False) when Ruff is not on PATH.
    """
    cmd = _build_ruff_cmd(path, select)
    try:
        stdout, stderr, returncode = await _exec(cmd, timeout=30.0)
    except FileNotFoundError:
        return RuffResult(available=False, findings=[], total=0)
    except TimeoutError:
        return RuffResult(available=True, findings=[], total=0, error="Ruff timed out after 30s")
    if returncode == 2:
        msg = stderr.decode(errors="replace").strip() or "ruff internal error"
        return RuffResult(available=True, findings=[], total=0, error=msg)
    findings = _parse_ruff_output(stdout)
    return RuffResult(available=True, findings=findings, total=len(findings))


def _parse_mypy_output(stdout: bytes) -> list[MypyError]:
    errors: list[MypyError] = []
    for raw_line in stdout.decode(errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            item: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            errors.append(
                MypyError(
                    filename=str(item["file"]),
                    line=int(item["line"]),
                    col=int(item["column"]) if item.get("column") is not None else None,
                    error_code=str(item["code"]) if item.get("code") is not None else None,
                    message=str(item["message"]),
                    severity=str(item.get("severity", "error")),
                )
            )
        except (KeyError, TypeError):
            continue
    return errors


async def run_mypy(path: str) -> MypyResult:
    """Run mypy type checker on path and return structured type errors.

    Degrades gracefully (available=False) when mypy is not on PATH.
    """
    cmd = ["mypy", path, "--output=json", "--no-error-summary", "--ignore-missing-imports"]
    try:
        stdout, _stderr, _returncode = await _exec(cmd, timeout=60.0)
    except FileNotFoundError:
        return MypyResult(available=False, errors=[], total_errors=0, total_warnings=0)
    except TimeoutError:
        return MypyResult(
            available=True,
            errors=[],
            total_errors=0,
            total_warnings=0,
            error="mypy timed out after 60s",
        )
    errors = _parse_mypy_output(stdout)
    total_errors = sum(1 for e in errors if e.severity == "error")
    total_warnings = sum(1 for e in errors if e.severity == "warning")
    return MypyResult(
        available=True,
        errors=errors,
        total_errors=total_errors,
        total_warnings=total_warnings,
    )
