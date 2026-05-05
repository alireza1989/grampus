#!/usr/bin/env bash
# Run before tagging a release. All checks must pass.
set -e

echo "=== Lint ==="
uv run ruff check .
uv run ruff format --check .

echo "=== Type check ==="
uv run mypy src/

echo "=== Unit tests ==="
uv run pytest --ignore=tests/dapr/test_integration.py -m "not dapr" -q

echo "=== Integration tests ==="
uv run pytest tests/integration/ -m "integration and not dapr" -q

echo "=== Docs build ==="
uv run mkdocs build --strict

echo "=== Build wheel ==="
uv build

echo ""
echo "All checks passed. Ready to tag and release."
echo "  git tag v0.1.0 && git push origin v0.1.0"
