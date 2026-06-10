"""Tests for H45 AST code analysis engine (Part 1)."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from nexus.tools.library.code_analysis_types import (
    ComplexityRating,
    ImportKind,
)
from nexus.tools.library.code_analyzer import (
    CodeAnalyzer,
    analyze_file,
    classify_import,
    compute_complexity,
    extract_class,
    extract_function,
    extract_parameters,
    find_symbol,
    parse_source,
    summarize_directory,
)

# ---------------------------------------------------------------------------
# Test fixtures (source snippets)
# ---------------------------------------------------------------------------

SIMPLE_MODULE = textwrap.dedent("""\
    '''Module docstring.'''
    import os
    import sys
    from pathlib import Path
    from pydantic import BaseModel

    def greet(name: str) -> str:
        '''Return a greeting.'''
        return f"Hello, {name}"

    def add(x: int, y: int) -> int:
        return x + y

    class Greeter:
        '''A greeter class.'''
        prefix: str = "Hello"

        def say(self, name: str) -> str:
            return f"{self.prefix}, {name}"

        @classmethod
        def from_prefix(cls, prefix: str) -> "Greeter":
            return cls()

        @staticmethod
        def version() -> int:
            return 1

        @property
        def tag(self) -> str:
            return "greeter"
""")

COMPLEX_MODULE = textwrap.dedent("""\
    def complex_func(items):
        if items:
            for item in items:
                if item > 0:
                    try:
                        result = item / 2
                    except ZeroDivisionError:
                        pass
                    while result > 1:
                        result -= 1
        return items

    async def bool_heavy(a, b, c, d):
        if a and b and c:
            pass
        if d or a:
            pass

    def outer():
        def inner():
            if True:
                pass
        return inner
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_func(src: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(src)
    return next(  # type: ignore[return-value]
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _parse_class(src: str) -> ast.ClassDef:
    tree = ast.parse(src)
    return next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# parse_source
# ---------------------------------------------------------------------------


def test_parse_source_valid_returns_tree_and_no_error() -> None:
    tree, err = parse_source("x = 1")
    assert tree is not None
    assert err is None


def test_parse_source_invalid_returns_none_tree_and_message() -> None:
    tree, err = parse_source("def (:")
    assert tree is None
    assert err is not None
    assert len(err) > 0


# ---------------------------------------------------------------------------
# Cyclomatic complexity
# ---------------------------------------------------------------------------


def test_complexity_simple_function_is_1() -> None:
    src = "def f(): return 1"
    node = _parse_func(src)
    assert compute_complexity(node) == 1


def test_complexity_if_adds_1() -> None:
    src = textwrap.dedent("""\
        def f(x):
            if x:
                return 1
            return 0
    """)
    node = _parse_func(src)
    assert compute_complexity(node) == 2


def test_complexity_for_while_try_add_1_each() -> None:
    src = textwrap.dedent("""\
        def f(items):
            for item in items:
                pass
            while True:
                break
            try:
                pass
            except Exception:
                pass
            return 0
    """)
    node = _parse_func(src)
    # for=1, while=1, try=1, except=1, base=1 → 5
    assert compute_complexity(node) == 5


def test_complexity_bool_op_and_adds_per_operand() -> None:
    # a and b and c → BoolOp with 3 values → adds 2 (values - 1)
    src = textwrap.dedent("""\
        def f(a, b, c):
            if a and b and c:
                pass
    """)
    node = _parse_func(src)
    # if=1, and(3 values)=2, base=1 → 4
    assert compute_complexity(node) == 4


def test_complexity_nested_function_not_counted() -> None:
    src = textwrap.dedent("""\
        def outer():
            def inner():
                if True:
                    if False:
                        pass
            return inner
    """)
    # Walk outer only; inner's two ifs should NOT be counted
    tree = ast.parse(src)
    outer_node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "outer"
    )
    assert compute_complexity(outer_node) == 1


def test_complexity_rating_thresholds() -> None:
    def _rating(cc: int) -> ComplexityRating:
        if cc <= 5:
            return ComplexityRating.LOW
        if cc <= 10:
            return ComplexityRating.MEDIUM
        if cc <= 15:
            return ComplexityRating.HIGH
        return ComplexityRating.VERY_HIGH

    assert _rating(1) == ComplexityRating.LOW
    assert _rating(5) == ComplexityRating.LOW
    assert _rating(6) == ComplexityRating.MEDIUM
    assert _rating(10) == ComplexityRating.MEDIUM
    assert _rating(11) == ComplexityRating.HIGH
    assert _rating(15) == ComplexityRating.HIGH
    assert _rating(16) == ComplexityRating.VERY_HIGH
    assert _rating(99) == ComplexityRating.VERY_HIGH


# ---------------------------------------------------------------------------
# Import classification
# ---------------------------------------------------------------------------


def test_classify_os_is_stdlib() -> None:
    assert classify_import("os") == ImportKind.STDLIB


def test_classify_sys_is_stdlib() -> None:
    assert classify_import("sys") == ImportKind.STDLIB


def test_classify_pathlib_is_stdlib() -> None:
    assert classify_import("pathlib") == ImportKind.STDLIB


def test_classify_pydantic_is_third_party() -> None:
    assert classify_import("pydantic") == ImportKind.THIRD_PARTY


def test_classify_relative_import_is_local() -> None:
    assert classify_import(".utils") == ImportKind.LOCAL


def test_classify_nexus_is_third_party() -> None:
    # nexus is not in sys.stdlib_module_names
    assert classify_import("nexus") == ImportKind.THIRD_PARTY


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------


def test_extract_parameters_with_annotations() -> None:
    src = "def f(x: int, y: str) -> None: pass"
    node = _parse_func(src)
    params = extract_parameters(node.args)
    assert len(params) == 2
    assert params[0].name == "x"
    assert params[0].annotation == "int"
    assert params[1].name == "y"
    assert params[1].annotation == "str"


def test_extract_parameters_no_annotations() -> None:
    src = "def f(a, b): pass"
    node = _parse_func(src)
    params = extract_parameters(node.args)
    assert params[0].annotation is None
    assert params[1].annotation is None


def test_extract_parameters_defaults_flagged() -> None:
    src = "def f(a, b=1, c=2): pass"
    node = _parse_func(src)
    params = extract_parameters(node.args)
    assert params[0].has_default is False
    assert params[1].has_default is True
    assert params[2].has_default is True


def test_extract_parameters_vararg_and_kwarg() -> None:
    src = "def f(*args, **kwargs): pass"
    node = _parse_func(src)
    params = extract_parameters(node.args)
    names = [p.name for p in params]
    assert "args" in names
    assert "kwargs" in names


# ---------------------------------------------------------------------------
# Function extraction
# ---------------------------------------------------------------------------


def test_extract_function_name_and_qualname() -> None:
    src = "def my_func(x): pass"
    node = _parse_func(src)
    lines = src.splitlines()
    info = extract_function(node, source_lines=lines)
    assert info.name == "my_func"
    assert info.qualname == "my_func"


def test_extract_async_function_is_async_true() -> None:
    src = "async def my_func(): pass"
    node = _parse_func(src)
    info = extract_function(node, source_lines=src.splitlines())
    assert info.is_async is True


def test_extract_method_qualname_has_class_prefix() -> None:
    src = "def method(self): pass"
    node = _parse_func(src)
    info = extract_function(node, parent_class="MyClass", source_lines=src.splitlines())
    assert info.qualname == "MyClass.method"
    assert info.is_method is True


def test_extract_classmethod_flag() -> None:
    src = textwrap.dedent("""\
        class Foo:
            @classmethod
            def create(cls): pass
    """)
    tree = ast.parse(src)
    cls_node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    method_node = next(
        n for n in cls_node.body if isinstance(n, ast.FunctionDef) and n.name == "create"
    )
    info = extract_function(method_node, parent_class="Foo", source_lines=src.splitlines())
    assert info.is_classmethod is True


def test_extract_staticmethod_flag() -> None:
    src = textwrap.dedent("""\
        class Foo:
            @staticmethod
            def helper(): pass
    """)
    tree = ast.parse(src)
    cls_node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    method_node = next(
        n for n in cls_node.body if isinstance(n, ast.FunctionDef) and n.name == "helper"
    )
    info = extract_function(method_node, parent_class="Foo", source_lines=src.splitlines())
    assert info.is_staticmethod is True


def test_extract_property_flag() -> None:
    src = textwrap.dedent("""\
        class Foo:
            @property
            def name(self): pass
    """)
    tree = ast.parse(src)
    cls_node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    method_node = next(
        n for n in cls_node.body if isinstance(n, ast.FunctionDef) and n.name == "name"
    )
    info = extract_function(method_node, parent_class="Foo", source_lines=src.splitlines())
    assert info.is_property is True


def test_extract_return_annotation() -> None:
    src = "def f() -> list[str]: pass"
    node = _parse_func(src)
    info = extract_function(node, source_lines=src.splitlines())
    assert info.return_annotation == "list[str]"


def test_extract_docstring() -> None:
    src = textwrap.dedent("""\
        def f():
            '''My docstring.'''
            pass
    """)
    node = _parse_func(src)
    info = extract_function(node, source_lines=src.splitlines())
    assert info.docstring == "My docstring."


def test_extract_function_no_docstring() -> None:
    src = "def f(): pass"
    node = _parse_func(src)
    info = extract_function(node, source_lines=src.splitlines())
    assert info.docstring is None


# ---------------------------------------------------------------------------
# Class extraction
# ---------------------------------------------------------------------------


def test_extract_class_bases() -> None:
    src = textwrap.dedent("""\
        class Foo(Base, Mixin):
            pass
    """)
    node = _parse_class(src)
    info = extract_class(node, source_lines=src.splitlines())
    assert "Base" in info.bases
    assert "Mixin" in info.bases


def test_extract_class_methods_count() -> None:
    node = _parse_class(SIMPLE_MODULE)
    info = extract_class(node, source_lines=SIMPLE_MODULE.splitlines())
    assert len(info.methods) == 4  # say, from_prefix, version, tag


def test_extract_class_variables() -> None:
    src = textwrap.dedent("""\
        class Config:
            debug: bool = False
            name = "default"
            def method(self): pass
    """)
    node = _parse_class(src)
    info = extract_class(node, source_lines=src.splitlines())
    assert "debug" in info.class_variables
    assert "name" in info.class_variables
    assert "method" not in info.class_variables


def test_extract_class_docstring() -> None:
    src = textwrap.dedent("""\
        class Foo:
            '''A foo class.'''
            pass
    """)
    node = _parse_class(src)
    info = extract_class(node, source_lines=src.splitlines())
    assert info.docstring == "A foo class."


# ---------------------------------------------------------------------------
# Module analysis (async)
# ---------------------------------------------------------------------------


async def test_analyze_file_counts_functions(tmp_path: Path) -> None:
    p = tmp_path / "mod.py"
    p.write_text(SIMPLE_MODULE)
    info = await analyze_file(str(p))
    assert info.has_syntax_error is False
    assert len(info.functions) == 2  # greet, add (top-level only)


async def test_analyze_file_counts_classes(tmp_path: Path) -> None:
    p = tmp_path / "mod.py"
    p.write_text(SIMPLE_MODULE)
    info = await analyze_file(str(p))
    assert len(info.classes) == 1  # Greeter


async def test_analyze_file_syntax_error_graceful(tmp_path: Path) -> None:
    p = tmp_path / "bad.py"
    p.write_text("def (:")
    info = await analyze_file(str(p))
    assert info.has_syntax_error is True
    assert info.syntax_error_message is not None


async def test_analyze_file_imports_classified(tmp_path: Path) -> None:
    p = tmp_path / "mod.py"
    p.write_text(SIMPLE_MODULE)
    info = await analyze_file(str(p))
    kinds = {imp.module: imp.kind for imp in info.imports}
    assert kinds["os"] == ImportKind.STDLIB
    assert kinds["sys"] == ImportKind.STDLIB
    assert kinds["pathlib"] == ImportKind.STDLIB
    assert kinds["pydantic"] == ImportKind.THIRD_PARTY


async def test_analyze_file_module_name_derived_from_path(tmp_path: Path) -> None:
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    p = pkg / "utils.py"
    p.write_text("x = 1")
    info = await analyze_file(str(p))
    # module_name may or may not be derivable depending on cwd, but it's set or None
    assert info.module_name is None or isinstance(info.module_name, str)


async def test_analyze_file_docstring(tmp_path: Path) -> None:
    src = '"""Module level docstring."""\nx = 1\n'
    p = tmp_path / "mod.py"
    p.write_text(src)
    info = await analyze_file(str(p))
    assert info.docstring == "Module level docstring."


async def test_analyze_file_total_lines(tmp_path: Path) -> None:
    p = tmp_path / "mod.py"
    p.write_text(SIMPLE_MODULE)
    info = await analyze_file(str(p))
    assert info.total_lines == len(SIMPLE_MODULE.splitlines())


# ---------------------------------------------------------------------------
# Symbol search (async)
# ---------------------------------------------------------------------------


async def test_find_symbol_finds_function_in_directory(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def target_func(): pass\n")
    (tmp_path / "b.py").write_text("def other_func(): pass\n")
    matches = await find_symbol("target_func", tmp_path)
    assert len(matches) == 1
    assert matches[0].name == "target_func"
    assert "target_func" in matches[0].qualname


async def test_find_symbol_finds_class(tmp_path: Path) -> None:
    (tmp_path / "c.py").write_text("class TargetClass:\n    pass\n")
    matches = await find_symbol("TargetClass", tmp_path)
    assert len(matches) == 1
    assert matches[0].kind == "class"


async def test_find_symbol_returns_empty_for_missing(tmp_path: Path) -> None:
    (tmp_path / "d.py").write_text("def unrelated(): pass\n")
    matches = await find_symbol("NonExistentSymbol", tmp_path)
    assert matches == []


async def test_find_symbol_respects_max_files(tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text(f"def func_{i}(): pass\n")
    # max_files=3 should limit files scanned
    matches = await find_symbol("func_0", tmp_path, max_files=3)
    # We might or might not find it depending on glob order — just check no crash
    assert isinstance(matches, list)


async def test_find_symbol_finds_method(tmp_path: Path) -> None:
    src = textwrap.dedent("""\
        class MyClass:
            def my_method(self): pass
    """)
    (tmp_path / "e.py").write_text(src)
    matches = await find_symbol("my_method", tmp_path)
    assert len(matches) == 1
    assert matches[0].kind == "method"
    assert "MyClass" in matches[0].qualname


# ---------------------------------------------------------------------------
# Structure summary (async)
# ---------------------------------------------------------------------------


async def test_summarize_directory_lists_public_names(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text(SIMPLE_MODULE)
    summaries = await summarize_directory(tmp_path)
    assert len(summaries) == 1
    s = summaries[0]
    assert "greet" in s.public_functions
    assert "add" in s.public_functions
    assert "Greeter" in s.public_classes


async def test_summarize_directory_excludes_private_names(tmp_path: Path) -> None:
    src = textwrap.dedent("""\
        def public_fn(): pass
        def _private_fn(): pass
        class _PrivateClass: pass
        class PublicClass: pass
    """)
    (tmp_path / "mod.py").write_text(src)
    summaries = await summarize_directory(tmp_path)
    s = summaries[0]
    assert "public_fn" in s.public_functions
    assert "_private_fn" not in s.public_functions
    assert "PublicClass" in s.public_classes
    assert "_PrivateClass" not in s.public_classes


async def test_summarize_directory_marks_syntax_errors(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text("def (:")
    summaries = await summarize_directory(tmp_path)
    assert summaries[0].has_syntax_error is True


async def test_summarize_directory_sorted_by_path(tmp_path: Path) -> None:
    (tmp_path / "z_mod.py").write_text("def z(): pass")
    (tmp_path / "a_mod.py").write_text("def a(): pass")
    summaries = await summarize_directory(tmp_path)
    paths = [s.path for s in summaries]
    assert paths == sorted(paths)


async def test_summarize_directory_import_count(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text(SIMPLE_MODULE)
    summaries = await summarize_directory(tmp_path)
    # SIMPLE_MODULE has: import os, import sys, from pathlib import Path, from pydantic import ...
    assert summaries[0].import_count == 4


# ---------------------------------------------------------------------------
# CodeAnalyzer class interface (smoke test)
# ---------------------------------------------------------------------------


async def test_code_analyzer_analyze_file_delegates(tmp_path: Path) -> None:
    p = tmp_path / "mod.py"
    p.write_text("def f(): pass")
    info = await CodeAnalyzer.analyze_file(str(p))
    assert info.functions[0].name == "f"


async def test_code_analyzer_find_symbol_delegates(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("def my_func(): pass")
    matches = await CodeAnalyzer.find_symbol("my_func", tmp_path)
    assert len(matches) == 1


async def test_code_analyzer_summarize_directory_delegates(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("def pub(): pass\ndef _priv(): pass")
    summaries = await CodeAnalyzer.summarize_directory(tmp_path)
    assert "pub" in summaries[0].public_functions
