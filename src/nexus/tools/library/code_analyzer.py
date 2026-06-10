"""H45 AST-based code analysis engine — stdlib only, zero new deps."""

from __future__ import annotations

import ast
import asyncio
import sys
from pathlib import Path

from nexus.tools.library.code_analysis_types import (
    ClassInfo,
    ComplexityRating,
    FunctionInfo,
    ImportInfo,
    ImportKind,
    ModuleInfo,
    ParameterInfo,
    StructureSummary,
    SymbolMatch,
)

# ---------------------------------------------------------------------------
# stdlib name set (Python 3.10+; fallback for older)
# ---------------------------------------------------------------------------

try:
    _STDLIB_NAMES: frozenset[str] = frozenset(sys.stdlib_module_names)
except AttributeError:
    _STDLIB_NAMES = frozenset(
        [
            "abc",
            "ast",
            "asyncio",
            "builtins",
            "collections",
            "contextlib",
            "copy",
            "csv",
            "dataclasses",
            "datetime",
            "enum",
            "functools",
            "gc",
            "hashlib",
            "heapq",
            "http",
            "importlib",
            "inspect",
            "io",
            "itertools",
            "json",
            "logging",
            "math",
            "multiprocessing",
            "operator",
            "os",
            "pathlib",
            "pickle",
            "platform",
            "queue",
            "random",
            "re",
            "shutil",
            "signal",
            "socket",
            "sqlite3",
            "string",
            "struct",
            "subprocess",
            "sys",
            "tempfile",
            "textwrap",
            "threading",
            "time",
            "traceback",
            "typing",
            "unittest",
            "urllib",
            "uuid",
            "warnings",
            "weakref",
            "xml",
            "zipfile",
        ]
    )

_MAX_FILE_SIZE = 500 * 1024  # 500 KB
_MAX_MATCHES = 50


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def parse_source(source: str) -> tuple[ast.Module | None, str | None]:
    """Return (tree, None) on success or (None, error_message) on SyntaxError."""
    try:
        return ast.parse(source), None
    except SyntaxError as exc:
        return None, str(exc)


def classify_import(module_name: str) -> ImportKind:
    """Classify a module name as STDLIB, LOCAL (relative), or THIRD_PARTY."""
    if module_name.startswith("."):
        return ImportKind.LOCAL
    top = module_name.split(".")[0]
    if top in _STDLIB_NAMES:
        return ImportKind.STDLIB
    return ImportKind.THIRD_PARTY


# ---------------------------------------------------------------------------
# Complexity visitor
# ---------------------------------------------------------------------------


class _ComplexityVisitor(ast.NodeVisitor):
    """Count McCabe cyclomatic complexity for a single function body."""

    def __init__(self) -> None:
        self.count = 1

    def visit_If(self, node: ast.If) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # Each additional operand beyond the first adds 1
        self.count += len(node.values) - 1
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.count += 1
        self.generic_visit(node)

    # Do NOT recurse into nested function/class definitions
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        pass


def compute_complexity(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> int:
    """McCabe cyclomatic complexity for a function node.

    Counts branch-inducing nodes in the function body, excluding nested defs.
    Starts at 1 (base path).
    """
    visitor = _ComplexityVisitor()
    for child in func_node.body:
        visitor.visit(child)
    return visitor.count


def _complexity_rating(cc: int) -> ComplexityRating:
    if cc <= 5:
        return ComplexityRating.LOW
    if cc <= 10:
        return ComplexityRating.MEDIUM
    if cc <= 15:
        return ComplexityRating.HIGH
    return ComplexityRating.VERY_HIGH


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------


def extract_parameters(args: ast.arguments) -> list[ParameterInfo]:
    """Extract all parameters with annotations and default flags."""
    params: list[ParameterInfo] = []
    all_args = list(args.args)
    defaults_offset = len(all_args) - len(args.defaults)

    for i, arg in enumerate(all_args):
        params.append(
            ParameterInfo(
                name=arg.arg,
                annotation=ast.unparse(arg.annotation) if arg.annotation else None,
                has_default=i >= defaults_offset,
            )
        )

    if args.vararg:
        params.append(
            ParameterInfo(
                name=args.vararg.arg,
                annotation=ast.unparse(args.vararg.annotation) if args.vararg.annotation else None,
                has_default=False,
            )
        )

    kw_defaults_offset = len(args.kwonlyargs) - len(args.kw_defaults)
    for i, arg in enumerate(args.kwonlyargs):
        params.append(
            ParameterInfo(
                name=arg.arg,
                annotation=ast.unparse(arg.annotation) if arg.annotation else None,
                has_default=args.kw_defaults[i - kw_defaults_offset] is not None
                if i >= kw_defaults_offset
                else False,
            )
        )

    if args.kwarg:
        params.append(
            ParameterInfo(
                name=args.kwarg.arg,
                annotation=ast.unparse(args.kwarg.annotation) if args.kwarg.annotation else None,
                has_default=False,
            )
        )

    return params


# ---------------------------------------------------------------------------
# Function extraction
# ---------------------------------------------------------------------------


def extract_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    parent_class: str | None = None,
    source_lines: list[str],
) -> FunctionInfo:
    """Extract structured information from a function/method AST node."""
    decorators = [ast.unparse(d) for d in node.decorator_list]
    cc = compute_complexity(node)
    return FunctionInfo(
        name=node.name,
        qualname=f"{parent_class}.{node.name}" if parent_class else node.name,
        line_start=node.lineno,
        line_end=node.end_lineno or node.lineno,
        is_async=isinstance(node, ast.AsyncFunctionDef),
        is_method=parent_class is not None,
        is_classmethod="classmethod" in decorators,
        is_staticmethod="staticmethod" in decorators,
        is_property="property" in decorators,
        decorators=decorators,
        parameters=extract_parameters(node.args),
        return_annotation=ast.unparse(node.returns) if node.returns else None,
        docstring=ast.get_docstring(node),
        cyclomatic_complexity=cc,
        complexity_rating=_complexity_rating(cc),
    )


# ---------------------------------------------------------------------------
# Class extraction
# ---------------------------------------------------------------------------


def _collect_class_variables(body: list[ast.stmt]) -> list[str]:
    names: list[str] = []
    for stmt in body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.append(stmt.target.id)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
    return names


def extract_class(node: ast.ClassDef, *, source_lines: list[str]) -> ClassInfo:
    """Extract structured information from a class AST node."""
    methods = [
        extract_function(n, parent_class=node.name, source_lines=source_lines)
        for n in node.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return ClassInfo(
        name=node.name,
        line_start=node.lineno,
        line_end=node.end_lineno or node.lineno,
        bases=[ast.unparse(b) for b in node.bases],
        decorators=[ast.unparse(d) for d in node.decorator_list],
        docstring=ast.get_docstring(node),
        methods=methods,
        class_variables=_collect_class_variables(node.body),
    )


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------


def _extract_imports(body: list[ast.stmt]) -> list[ImportInfo]:
    imports: list[ImportInfo] = []
    for node in body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    ImportInfo(
                        module=alias.name,
                        names=[],
                        kind=classify_import(alias.name),
                        line=node.lineno,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = node.level or 0
            module_key = ("." * level) + module if level else module
            imports.append(
                ImportInfo(
                    module=module_key if module_key else ".",
                    names=[a.name for a in node.names],
                    kind=classify_import(module_key),
                    line=node.lineno,
                )
            )
    return imports


# ---------------------------------------------------------------------------
# Module name derivation
# ---------------------------------------------------------------------------


def _derive_module_name(path: Path) -> str | None:
    try:
        rel = path.relative_to(Path.cwd())
        parts = list(rel.with_suffix("").parts)
        return ".".join(parts)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Async public API
# ---------------------------------------------------------------------------


def _analyze_file_sync(path: Path) -> ModuleInfo:
    source = path.read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines()
    tree, err_msg = parse_source(source)

    if tree is None:
        return ModuleInfo(
            path=str(path),
            module_name=_derive_module_name(path),
            docstring=None,
            imports=[],
            functions=[],
            classes=[],
            total_lines=len(lines),
            has_syntax_error=True,
            syntax_error_message=err_msg,
        )

    imports = _extract_imports(tree.body)
    functions = [
        extract_function(n, source_lines=lines)
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    classes = [
        extract_class(n, source_lines=lines) for n in tree.body if isinstance(n, ast.ClassDef)
    ]

    return ModuleInfo(
        path=str(path),
        module_name=_derive_module_name(path),
        docstring=ast.get_docstring(tree),
        imports=imports,
        functions=functions,
        classes=classes,
        total_lines=len(lines),
        has_syntax_error=False,
        syntax_error_message=None,
    )


async def analyze_file(path: str | Path) -> ModuleInfo:
    """Full structural analysis of a single .py file."""
    return await asyncio.to_thread(_analyze_file_sync, Path(path))


def _build_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    name: str,
) -> str:
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        return f"class {name}({bases})" if bases else f"class {name}"
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    params = ast.unparse(node.args)
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix} {name}({params}){ret}"


def _search_file_sync(
    path: Path,
    name: str,
) -> list[SymbolMatch]:
    if path.stat().st_size > _MAX_FILE_SIZE:
        return []
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree, _ = parse_source(source)
        if tree is None:
            return []
    except OSError:
        return []

    matches: list[SymbolMatch] = []

    # Top-level functions and classes
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            matches.append(
                SymbolMatch(
                    path=str(path),
                    line=node.lineno,
                    kind="function",
                    qualname=node.name,
                    name=node.name,
                    signature=_build_signature(node, node.name),
                )
            )
        elif isinstance(node, ast.ClassDef):
            if node.name == name:
                matches.append(
                    SymbolMatch(
                        path=str(path),
                        line=node.lineno,
                        kind="class",
                        qualname=node.name,
                        name=node.name,
                        signature=_build_signature(node, node.name),
                    )
                )
            # Search methods inside
            for child in node.body:
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == name
                ):
                    qualname = f"{node.name}.{child.name}"
                    matches.append(
                        SymbolMatch(
                            path=str(path),
                            line=child.lineno,
                            kind="method",
                            qualname=qualname,
                            name=child.name,
                            signature=_build_signature(child, child.name),
                        )
                    )

    return matches


async def find_symbol(
    name: str,
    directory: str | Path,
    *,
    max_files: int = 200,
) -> list[SymbolMatch]:
    """Search all .py files under directory for a function or class named name."""
    directory = Path(directory)
    paths = [p for p in directory.rglob("*.py") if "__pycache__" not in p.parts][:max_files]

    results = await asyncio.gather(*[asyncio.to_thread(_search_file_sync, p, name) for p in paths])
    matches: list[SymbolMatch] = []
    for batch in results:
        matches.extend(batch)
        if len(matches) >= _MAX_MATCHES:
            break
    return matches[:_MAX_MATCHES]


def _summarize_file_sync(path: Path) -> StructureSummary:
    if path.stat().st_size > _MAX_FILE_SIZE:
        return StructureSummary(
            path=str(path),
            module_name=_derive_module_name(path),
            public_functions=[],
            public_classes=[],
            import_count=0,
            has_syntax_error=False,
        )
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return StructureSummary(
            path=str(path),
            module_name=None,
            public_functions=[],
            public_classes=[],
            import_count=0,
            has_syntax_error=False,
        )

    tree, err_msg = parse_source(source)
    if tree is None:
        return StructureSummary(
            path=str(path),
            module_name=_derive_module_name(path),
            public_functions=[],
            public_classes=[],
            import_count=0,
            has_syntax_error=True,
        )

    pub_funcs = [
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")
    ]
    pub_classes = [
        n.name for n in tree.body if isinstance(n, ast.ClassDef) and not n.name.startswith("_")
    ]
    import_count = sum(1 for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))

    return StructureSummary(
        path=str(path),
        module_name=_derive_module_name(path),
        public_functions=pub_funcs,
        public_classes=pub_classes,
        import_count=import_count,
        has_syntax_error=False,
    )


async def summarize_directory(
    directory: str | Path,
    *,
    max_files: int = 200,
) -> list[StructureSummary]:
    """Lightweight index: public API surface of every .py file in directory."""
    directory = Path(directory)
    paths = sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)[:max_files]

    results = await asyncio.gather(*[asyncio.to_thread(_summarize_file_sync, p) for p in paths])
    return list(results)


# ---------------------------------------------------------------------------
# CodeAnalyzer — stateless class interface
# ---------------------------------------------------------------------------


class CodeAnalyzer:
    """Stateless facade for the AST analysis engine."""

    @staticmethod
    async def analyze_file(path: str | Path) -> ModuleInfo:
        """Full structural analysis of a single .py file."""
        return await analyze_file(path)

    @staticmethod
    async def find_symbol(
        name: str,
        directory: str | Path,
        *,
        max_files: int = 200,
    ) -> list[SymbolMatch]:
        """Search all .py files under directory for a symbol named name."""
        return await find_symbol(name, directory, max_files=max_files)

    @staticmethod
    async def summarize_directory(
        directory: str | Path,
        *,
        max_files: int = 200,
    ) -> list[StructureSummary]:
        """Lightweight public-API index of every .py file in directory."""
        return await summarize_directory(directory, max_files=max_files)
