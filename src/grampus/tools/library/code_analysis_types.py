"""Pydantic models for H45 code analysis results."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ImportKind(StrEnum):
    """Classification of a Python import statement."""

    STDLIB = "stdlib"
    THIRD_PARTY = "third_party"
    LOCAL = "local"


class ComplexityRating(StrEnum):
    """McCabe cyclomatic complexity band."""

    LOW = "low"  # CC 1-5
    MEDIUM = "medium"  # CC 6-10
    HIGH = "high"  # CC 11-15
    VERY_HIGH = "very_high"  # CC 16+


class ImportInfo(BaseModel):
    """A single import statement extracted from a module."""

    module: str
    names: list[str]
    kind: ImportKind
    line: int


class ParameterInfo(BaseModel):
    """A single function/method parameter."""

    name: str
    annotation: str | None
    has_default: bool


class FunctionInfo(BaseModel):
    """Structural information about a function or method."""

    name: str
    qualname: str
    line_start: int
    line_end: int
    is_async: bool
    is_method: bool
    is_classmethod: bool
    is_staticmethod: bool
    is_property: bool
    decorators: list[str]
    parameters: list[ParameterInfo]
    return_annotation: str | None
    docstring: str | None
    cyclomatic_complexity: int
    complexity_rating: ComplexityRating


class ClassInfo(BaseModel):
    """Structural information about a class definition."""

    name: str
    line_start: int
    line_end: int
    bases: list[str]
    decorators: list[str]
    docstring: str | None
    methods: list[FunctionInfo]
    class_variables: list[str]


class ModuleInfo(BaseModel):
    """Full structural analysis of a single .py file."""

    path: str
    module_name: str | None
    docstring: str | None
    imports: list[ImportInfo]
    functions: list[FunctionInfo]
    classes: list[ClassInfo]
    total_lines: int
    has_syntax_error: bool
    syntax_error_message: str | None


class SymbolMatch(BaseModel):
    """A single match from a cross-file symbol search."""

    path: str
    line: int
    kind: str
    qualname: str
    name: str
    signature: str


class StructureSummary(BaseModel):
    """Lightweight project-level index — one entry per module."""

    path: str
    module_name: str | None
    public_functions: list[str]
    public_classes: list[str]
    import_count: int
    has_syntax_error: bool
