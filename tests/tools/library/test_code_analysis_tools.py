"""Tests for H45 code analysis tool functions and subprocess runners (Part 2)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

from grampus.tools.library.code_analysis_tools import (
    analyze_file_tool,
    check_types_tool,
    find_symbol_tool,
    lint_code_tool,
    summarize_structure_tool,
)
from grampus.tools.library.code_lint_runner import (
    MypyResult,
    RuffResult,
    run_mypy,
    run_ruff,
)

# ---------------------------------------------------------------------------
# Sample subprocess output fixtures
# ---------------------------------------------------------------------------

_RUFF_SAMPLE_JSON = json.dumps(
    [
        {
            "filename": "test.py",
            "location": {"row": 5, "column": 1},
            "end_location": {"row": 5, "column": 80},
            "code": "E501",
            "message": "Line too long (80 > 79 characters)",
            "fix": None,
            "url": "https://docs.astral.sh/ruff/rules/line-too-long",
        },
        {
            "filename": "test.py",
            "location": {"row": 12, "column": 1},
            "end_location": {"row": 12, "column": 10},
            "code": "F401",
            "message": "'os' imported but unused",
            "fix": {"applicability": "safe", "edits": []},
            "url": "https://docs.astral.sh/ruff/rules/unused-import",
        },
    ]
).encode()

_MYPY_SAMPLE_LINES = (
    json.dumps(
        {
            "file": "test.py",
            "line": 10,
            "column": 5,
            "message": 'Argument 1 to "foo" has incompatible type',
            "hint": None,
            "code": "arg-type",
            "severity": "error",
        }
    )
    + "\n"
    + json.dumps(
        {
            "file": "test.py",
            "line": 20,
            "column": 3,
            "message": "Missing return statement",
            "hint": "Add a return statement",
            "code": None,
            "severity": "warning",
        }
    )
).encode()


# ---------------------------------------------------------------------------
# RuffRunner tests
# ---------------------------------------------------------------------------


class TestRuffRunner:
    async def test_run_ruff_returns_findings_from_json(self) -> None:
        with patch(
            "grampus.tools.library.code_lint_runner._exec",
            new=AsyncMock(return_value=(_RUFF_SAMPLE_JSON, b"", 1)),
        ):
            result = await run_ruff("test.py")

        assert result.available is True
        assert len(result.findings) == 2
        assert result.total == 2
        assert result.findings[0].rule_id == "E501"
        assert result.findings[0].row == 5
        assert result.findings[0].col == 1
        assert result.findings[0].fix_available is False
        assert result.findings[1].rule_id == "F401"
        assert result.findings[1].fix_available is True

    async def test_run_ruff_graceful_when_not_on_path(self) -> None:
        with patch(
            "grampus.tools.library.code_lint_runner._exec",
            side_effect=FileNotFoundError("ruff: not found"),
        ):
            result = await run_ruff("test.py")

        assert result.available is False
        assert result.findings == []
        assert result.total == 0

    async def test_run_ruff_handles_ruff_internal_error(self) -> None:
        with patch(
            "grampus.tools.library.code_lint_runner._exec",
            new=AsyncMock(return_value=(b"", b"ruff: fatal error", 2)),
        ):
            result = await run_ruff("test.py")

        assert result.available is True
        assert result.error is not None
        assert len(result.error) > 0

    async def test_run_ruff_exit_zero_means_no_findings(self) -> None:
        with patch(
            "grampus.tools.library.code_lint_runner._exec",
            new=AsyncMock(return_value=(b"[]", b"", 0)),
        ):
            result = await run_ruff("test.py")

        assert result.available is True
        assert result.total == 0
        assert result.findings == []

    async def test_run_ruff_with_select_passes_flag(self) -> None:
        captured: list[list[str]] = []

        async def fake_exec(args: list[str], timeout: float) -> tuple[bytes, bytes, int]:
            captured.append(args)
            return b"[]", b"", 0

        with patch("grampus.tools.library.code_lint_runner._exec", side_effect=fake_exec):
            await run_ruff("test.py", select=["E", "F"])

        assert any("--select=E,F" in a for a in captured[0])


# ---------------------------------------------------------------------------
# MypyRunner tests
# ---------------------------------------------------------------------------


class TestMypyRunner:
    async def test_run_mypy_parses_json_lines(self) -> None:
        with patch(
            "grampus.tools.library.code_lint_runner._exec",
            new=AsyncMock(return_value=(_MYPY_SAMPLE_LINES, b"", 1)),
        ):
            result = await run_mypy("test.py")

        assert result.available is True
        assert len(result.errors) == 2
        assert result.total_errors == 1
        assert result.total_warnings == 1
        assert result.errors[0].error_code == "arg-type"
        assert result.errors[0].severity == "error"
        assert result.errors[1].severity == "warning"
        assert result.errors[1].error_code is None

    async def test_run_mypy_graceful_when_not_on_path(self) -> None:
        with patch(
            "grampus.tools.library.code_lint_runner._exec",
            side_effect=FileNotFoundError("mypy: not found"),
        ):
            result = await run_mypy("test.py")

        assert result.available is False
        assert result.errors == []
        assert result.total_errors == 0
        assert result.total_warnings == 0

    async def test_run_mypy_clean_exit_zero_no_errors(self) -> None:
        with patch(
            "grampus.tools.library.code_lint_runner._exec",
            new=AsyncMock(return_value=(b"", b"", 0)),
        ):
            result = await run_mypy("test.py")

        assert result.available is True
        assert result.total_errors == 0

    async def test_run_mypy_skips_non_json_lines(self) -> None:
        mixed = b"Success: no issues found\n" + _MYPY_SAMPLE_LINES
        with patch(
            "grampus.tools.library.code_lint_runner._exec",
            new=AsyncMock(return_value=(mixed, b"", 1)),
        ):
            result = await run_mypy("test.py")

        assert result.available is True
        # Non-JSON line should be silently skipped
        assert len(result.errors) == 2


# ---------------------------------------------------------------------------
# analyze_file_tool tests
# ---------------------------------------------------------------------------


class TestAnalyzeFileTool:
    async def test_returns_ok_with_module_info(self, tmp_path: Path) -> None:
        p = tmp_path / "mod.py"
        p.write_text("def hello(name: str) -> str:\n    return f'Hello {name}'\n")

        result = await analyze_file_tool(path=str(p))

        assert result["ok"] is True
        info = result["result"]
        assert info["has_syntax_error"] is False
        assert len(info["functions"]) == 1
        assert info["functions"][0]["name"] == "hello"

    async def test_file_not_found_returns_err(self, tmp_path: Path) -> None:
        result = await analyze_file_tool(path=str(tmp_path / "missing.py"))
        assert result["ok"] is False
        assert result["code"] == "FILE_NOT_FOUND"

    async def test_non_python_file_returns_err(self, tmp_path: Path) -> None:
        p = tmp_path / "data.txt"
        p.write_text("not python")
        result = await analyze_file_tool(path=str(p))
        assert result["ok"] is False
        assert result["code"] == "UNSUPPORTED_FORMAT"

    async def test_syntax_error_returns_ok_with_flag(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.py"
        p.write_text("def (:")
        result = await analyze_file_tool(path=str(p))
        assert result["ok"] is True
        assert result["result"]["has_syntax_error"] is True

    async def test_file_too_large_returns_err(self, tmp_path: Path) -> None:
        p = tmp_path / "big.py"
        p.touch()
        import stat as stat_mod
        from unittest.mock import MagicMock

        mock_stat_result = MagicMock()
        mock_stat_result.st_size = 2_000_000
        mock_stat_result.st_mode = stat_mod.S_IFREG | 0o644  # regular file
        with patch.object(Path, "stat", return_value=mock_stat_result):
            result = await analyze_file_tool(path=str(p))
        assert result["ok"] is False
        assert result["code"] == "FILE_TOO_LARGE"


# ---------------------------------------------------------------------------
# lint_code_tool tests
# ---------------------------------------------------------------------------


class TestLintCodeTool:
    async def test_unavailable_returns_ok_with_hint(self, tmp_path: Path) -> None:
        p = tmp_path / "mod.py"
        p.write_text("x = 1\n")
        mock_result = RuffResult(available=False, findings=[], total=0)

        with patch(
            "grampus.tools.library.code_analysis_tools.run_ruff",
            new=AsyncMock(return_value=mock_result),
        ):
            result = await lint_code_tool(path=str(p))

        assert result["ok"] is True
        assert result["result"]["available"] is False
        assert "hint" in result
        assert "ruff" in result["hint"].lower()

    async def test_path_not_found_returns_err(self, tmp_path: Path) -> None:
        result = await lint_code_tool(path=str(tmp_path / "missing.py"))
        assert result["ok"] is False
        assert result["code"] == "FILE_NOT_FOUND"

    async def test_with_findings_returns_ok(self, tmp_path: Path) -> None:
        p = tmp_path / "mod.py"
        p.write_text("x = 1\n")
        mock_result = RuffResult(available=True, findings=[], total=0)

        with patch(
            "grampus.tools.library.code_analysis_tools.run_ruff",
            new=AsyncMock(return_value=mock_result),
        ):
            result = await lint_code_tool(path=str(p))

        assert result["ok"] is True
        assert result["result"]["available"] is True
        assert result["result"]["total"] == 0


# ---------------------------------------------------------------------------
# check_types_tool tests
# ---------------------------------------------------------------------------


class TestCheckTypesTool:
    async def test_unavailable_returns_ok_with_hint(self, tmp_path: Path) -> None:
        p = tmp_path / "mod.py"
        p.write_text("x: int = 1\n")
        mock_result = MypyResult(available=False, errors=[], total_errors=0, total_warnings=0)

        with patch(
            "grampus.tools.library.code_analysis_tools.run_mypy",
            new=AsyncMock(return_value=mock_result),
        ):
            result = await check_types_tool(path=str(p))

        assert result["ok"] is True
        assert result["result"]["available"] is False
        assert "hint" in result
        assert "mypy" in result["hint"].lower()

    async def test_path_not_found_returns_err(self, tmp_path: Path) -> None:
        result = await check_types_tool(path=str(tmp_path / "missing.py"))
        assert result["ok"] is False
        assert result["code"] == "FILE_NOT_FOUND"

    async def test_with_type_errors_returns_ok(self, tmp_path: Path) -> None:
        p = tmp_path / "mod.py"
        p.write_text("x: int = 'oops'\n")
        mock_result = MypyResult(available=True, errors=[], total_errors=0, total_warnings=0)

        with patch(
            "grampus.tools.library.code_analysis_tools.run_mypy",
            new=AsyncMock(return_value=mock_result),
        ):
            result = await check_types_tool(path=str(p))

        assert result["ok"] is True
        assert result["result"]["available"] is True


# ---------------------------------------------------------------------------
# find_symbol_tool tests
# ---------------------------------------------------------------------------


class TestFindSymbolTool:
    async def test_invalid_identifier_returns_err(self, tmp_path: Path) -> None:
        result = await find_symbol_tool(name="123bad", directory=str(tmp_path))
        assert result["ok"] is False
        assert result["code"] == "INVALID_IDENTIFIER"

    async def test_missing_directory_returns_err(self, tmp_path: Path) -> None:
        result = await find_symbol_tool(name="my_func", directory=str(tmp_path / "no_such"))
        assert result["ok"] is False
        assert result["code"] == "DIRECTORY_NOT_FOUND"

    async def test_finds_symbol_across_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("def target_fn(): pass\n")
        (tmp_path / "b.py").write_text("def other_fn(): pass\n")

        result = await find_symbol_tool(name="target_fn", directory=str(tmp_path))

        assert result["ok"] is True
        assert result["result"]["total"] == 1
        assert result["result"]["name"] == "target_fn"
        assert result["result"]["matches"][0]["name"] == "target_fn"

    async def test_returns_empty_for_missing_symbol(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("def something_else(): pass\n")

        result = await find_symbol_tool(name="nonexistent", directory=str(tmp_path))

        assert result["ok"] is True
        assert result["result"]["total"] == 0
        assert result["result"]["matches"] == []

    async def test_finds_class_symbol(self, tmp_path: Path) -> None:
        (tmp_path / "c.py").write_text("class MyClass:\n    pass\n")

        result = await find_symbol_tool(name="MyClass", directory=str(tmp_path))

        assert result["ok"] is True
        assert result["result"]["total"] == 1
        assert result["result"]["matches"][0]["kind"] == "class"


# ---------------------------------------------------------------------------
# summarize_structure_tool tests
# ---------------------------------------------------------------------------


class TestSummarizeStructureTool:
    async def test_returns_module_index(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("def pub_a(): pass\nclass PubClass: pass\n")
        (tmp_path / "b.py").write_text("def pub_b(): pass\n")

        result = await summarize_structure_tool(directory=str(tmp_path))

        assert result["ok"] is True
        data = result["result"]
        assert data["total_modules"] == 2
        names = {m["path"].split("/")[-1] for m in data["modules"]}
        assert "a.py" in names
        assert "b.py" in names
        pub_fns = {fn for m in data["modules"] for fn in m["public_functions"]}
        assert "pub_a" in pub_fns
        assert "pub_b" in pub_fns

    async def test_directory_not_exist_returns_err(self, tmp_path: Path) -> None:
        result = await summarize_structure_tool(directory=str(tmp_path / "no_such"))
        assert result["ok"] is False
        assert result["code"] == "DIRECTORY_NOT_FOUND"

    async def test_not_a_directory_returns_err(self, tmp_path: Path) -> None:
        p = tmp_path / "file.txt"
        p.write_text("hello")
        result = await summarize_structure_tool(directory=str(p))
        assert result["ok"] is False
        assert result["code"] == "NOT_A_DIRECTORY"

    async def test_excludes_private_names(self, tmp_path: Path) -> None:
        (tmp_path / "mod.py").write_text(
            textwrap.dedent("""\
                def public_fn(): pass
                def _private_fn(): pass
                class PublicClass: pass
                class _PrivateClass: pass
            """)
        )
        result = await summarize_structure_tool(directory=str(tmp_path))
        assert result["ok"] is True
        m = result["result"]["modules"][0]
        assert "public_fn" in m["public_functions"]
        assert "_private_fn" not in m["public_functions"]
        assert "PublicClass" in m["public_classes"]
        assert "_PrivateClass" not in m["public_classes"]


# ---------------------------------------------------------------------------
# Registration smoke test
# ---------------------------------------------------------------------------


def test_tools_registered_in_library() -> None:
    """All five tools appear in the LIBRARY_REGISTRY after import."""
    from grampus.tools.library import LIBRARY_REGISTRY

    names = {t.name for t in LIBRARY_REGISTRY.list_all()}
    assert "analyze_file" in names
    assert "lint_code" in names
    assert "check_types" in names
    assert "find_symbol" in names
    assert "summarize_structure" in names
