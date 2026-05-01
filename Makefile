.PHONY: install lint format typecheck test test-integration check dev clean

install:
	uv sync --all-extras --group dev

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src/

test:
	uv run pytest -m "not integration" -v

test-integration:
	docker compose up -d
	uv run pytest -m integration -v
	docker compose down

check: lint typecheck test

dev:
	docker compose up -d
	dapr run \
		--app-id nexus \
		--resources-path ./dapr/components \
		--config ./dapr/config.yaml \
		-- uv run python -m nexus

clean:
	rm -rf .venv .mypy_cache .ruff_cache .pytest_cache dist htmlcov coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
