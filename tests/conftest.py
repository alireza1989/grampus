"""Pytest configuration and shared fixtures for the Nexus test suite."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests requiring external services (docker, dapr, postgres, redis)",
    )
