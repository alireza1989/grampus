"""Phase 0 bootstrap tests — verify package structure, imports, and metadata.

These tests form the red phase of TDD for Phase 0. They assert what the
correct package skeleton must look like before any source files are written.
"""

import importlib
import importlib.util
import re

import pytest

LAYER_PACKAGES = [
    "grampus.core",
    "grampus.dapr",
    "grampus.memory",
    "grampus.tools",
    "grampus.orchestration",
    "grampus.safety",
    "grampus.observability",
    "grampus.evaluation",
    "grampus.cli",
]


# ---------------------------------------------------------------------------
# Root package
# ---------------------------------------------------------------------------


def test_grampus_importable() -> None:
    """Root grampus package can be imported without errors."""
    import grampus

    assert grampus is not None


def test_version_attribute_exists() -> None:
    """Package exposes a __version__ string attribute."""
    import grampus

    assert hasattr(grampus, "__version__"), "grampus.__version__ is not defined"
    assert isinstance(grampus.__version__, str), "__version__ must be a str"
    assert grampus.__version__, "__version__ must not be empty"


def test_version_is_semver() -> None:
    """Package version follows MAJOR.MINOR.PATCH semantic versioning."""
    import grampus

    pattern = r"^\d+\.\d+\.\d+$"
    assert re.match(pattern, grampus.__version__), (
        f"__version__ '{grampus.__version__}' does not follow semver (MAJOR.MINOR.PATCH)"
    )


def test_version_matches_pyproject() -> None:
    """Importable __version__ matches the version declared in pyproject.toml."""
    import importlib.metadata

    import grampus

    declared = importlib.metadata.version("grampus-ai")
    assert grampus.__version__ == declared, (
        f"grampus.__version__ ({grampus.__version__}) != pyproject version ({declared})"
    )


# ---------------------------------------------------------------------------
# Architecture layer sub-packages (9 layers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("package", LAYER_PACKAGES)
def test_layer_package_importable(package: str) -> None:
    """Each of the nine architecture-layer sub-packages can be imported."""
    mod = importlib.import_module(package)
    assert mod is not None


@pytest.mark.parametrize("package", LAYER_PACKAGES)
def test_layer_package_has_spec(package: str) -> None:
    """Each layer sub-package resolves to a real filesystem package (has __spec__)."""
    spec = importlib.util.find_spec(package)
    assert spec is not None, f"Module spec not found for '{package}'"
    assert spec.origin is not None or spec.submodule_search_locations is not None, (
        f"'{package}' has no origin or search locations — is it a namespace package?"
    )


@pytest.mark.parametrize("package", LAYER_PACKAGES)
def test_layer_package_is_package_not_module(package: str) -> None:
    """Each layer import is a package (has __path__), not a bare module."""
    mod = importlib.import_module(package)
    assert hasattr(mod, "__path__"), f"'{package}' is a module, not a package"


# ---------------------------------------------------------------------------
# CLI entry point stub
# ---------------------------------------------------------------------------


def test_cli_entry_point_importable() -> None:
    """grampus.cli.main is importable (required for the [project.scripts] entry point)."""
    import grampus.cli.main as cli_main

    assert cli_main is not None


def test_cli_object_exists() -> None:
    """grampus.cli.main.cli is a callable (the Click group registered in pyproject.toml)."""
    from grampus.cli.main import cli

    assert callable(cli)


# ---------------------------------------------------------------------------
# Package completeness sanity checks
# ---------------------------------------------------------------------------


def test_all_nine_layers_present() -> None:
    """Exactly nine architecture layers are importable (catches accidental deletions)."""
    failed = []
    for pkg in LAYER_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError as exc:
            failed.append(f"{pkg}: {exc}")

    assert not failed, "Missing layer packages:\n" + "\n".join(failed)


def test_no_layer_imports_bleed_into_root() -> None:
    """Root __init__.py does not explicitly import or re-export any layer.

    Python automatically binds subpackage names on the parent after any
    `import grampus.X` call — that is unavoidable and correct behaviour.
    This test instead checks the *source* of the root __init__.py to ensure
    it contains no explicit layer imports (those belong in later phases).
    """
    import inspect

    import grampus

    source = inspect.getsource(grampus)
    for layer in (
        "core",
        "dapr",
        "memory",
        "tools",
        "orchestration",
        "safety",
        "observability",
        "evaluation",
        "cli",
    ):
        assert f"import {layer}" not in source, (
            f"Root __init__.py explicitly imports layer '{layer}'. "
            "Phase 0 root package should only define __version__."
        )
        assert f"from grampus.{layer}" not in source, (
            f"Root __init__.py re-exports from 'grampus.{layer}'. "
            "Phase 0 root package should only define __version__."
        )
