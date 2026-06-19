"""Pytest configuration and shared fixtures for the Nexus test suite."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests requiring external services (docker, dapr, postgres, redis)",
    )
    config.addinivalue_line(
        "markers",
        "e2e: end-to-end tests requiring full agent loop",
    )
    config.addinivalue_line(
        "markers",
        "real_llm: tests that call real LLM APIs — requires RUN_REAL_LLM_TESTS=true and API keys",
    )
