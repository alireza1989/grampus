"""Tests for grampus.core.errors — the exception hierarchy."""

import pytest

from grampus.core.errors import (
    BudgetExceededError,
    ConfigError,
    GrampusError,
    MemoryError,
    MemorySecurityError,
    ModelError,
    OrchestrationError,
    SafetyError,
    ToolError,
    ToolTimeoutError,
)


class TestGrampusError:
    def test_grampus_error_is_exception(self) -> None:
        err = GrampusError("something broke", code="E001")
        assert isinstance(err, Exception)

    def test_grampus_error_stores_message(self) -> None:
        err = GrampusError("test message", code="E001")
        assert str(err) == "test message"

    def test_grampus_error_stores_code(self) -> None:
        err = GrampusError("msg", code="GRAMPUS_001")
        assert err.code == "GRAMPUS_001"

    def test_grampus_error_details_default_empty(self) -> None:
        err = GrampusError("msg", code="E001")
        assert err.details == {}

    def test_grampus_error_stores_details(self) -> None:
        details = {"key": "value", "count": 42}
        err = GrampusError("msg", code="E001", details=details)
        assert err.details == details

    def test_grampus_error_can_be_raised_and_caught(self) -> None:
        with pytest.raises(GrampusError) as exc_info:
            raise GrampusError("boom", code="E001")
        assert exc_info.value.code == "E001"


class TestErrorHierarchy:
    """All subclasses must be GrampusError instances."""

    def test_config_error_is_grampus_error(self) -> None:
        err = ConfigError("bad config", code="CONFIG_001")
        assert isinstance(err, GrampusError)

    def test_memory_error_is_grampus_error(self) -> None:
        err = MemoryError("memory fail", code="MEM_001")
        assert isinstance(err, GrampusError)

    def test_memory_security_error_is_grampus_error(self) -> None:
        err = MemorySecurityError("poisoning detected", code="MEMSEC_001")
        assert isinstance(err, GrampusError)

    def test_memory_security_error_is_memory_error(self) -> None:
        err = MemorySecurityError("poisoning detected", code="MEMSEC_001")
        assert isinstance(err, MemoryError)

    def test_tool_error_is_grampus_error(self) -> None:
        err = ToolError("tool failed", code="TOOL_001")
        assert isinstance(err, GrampusError)

    def test_tool_timeout_error_is_tool_error(self) -> None:
        err = ToolTimeoutError("timed out", code="TOOL_TIMEOUT")
        assert isinstance(err, ToolError)
        assert isinstance(err, GrampusError)

    def test_orchestration_error_is_grampus_error(self) -> None:
        err = OrchestrationError("graph failed", code="ORCH_001")
        assert isinstance(err, GrampusError)

    def test_budget_exceeded_error_is_orchestration_error(self) -> None:
        err = BudgetExceededError("over budget", code="BUDGET_001")
        assert isinstance(err, OrchestrationError)
        assert isinstance(err, GrampusError)

    def test_safety_error_is_grampus_error(self) -> None:
        err = SafetyError("injection detected", code="SAFETY_001")
        assert isinstance(err, GrampusError)

    def test_model_error_is_grampus_error(self) -> None:
        err = ModelError("api error", code="MODEL_001")
        assert isinstance(err, GrampusError)


class TestErrorDetails:
    def test_tool_error_carries_tool_name(self) -> None:
        err = ToolError("failed", code="TOOL_001", details={"tool_name": "web_search"})
        assert err.details["tool_name"] == "web_search"

    def test_budget_exceeded_carries_budget_info(self) -> None:
        err = BudgetExceededError(
            "exceeded $5 budget",
            code="BUDGET_001",
            details={"spent_usd": 5.12, "limit_usd": 5.0},
        )
        assert err.details["spent_usd"] == 5.12
        assert err.details["limit_usd"] == 5.0

    def test_model_error_carries_provider(self) -> None:
        err = ModelError("rate limited", code="MODEL_RATE_LIMIT", details={"provider": "anthropic"})
        assert err.details["provider"] == "anthropic"


class TestErrorCatching:
    def test_catch_subclass_as_parent(self) -> None:
        with pytest.raises(GrampusError):
            raise ToolTimeoutError("timeout", code="TOOL_TIMEOUT")

    def test_catch_budget_as_grampus(self) -> None:
        with pytest.raises(GrampusError):
            raise BudgetExceededError("budget exceeded", code="BUDGET_001")

    def test_catch_memory_security_as_memory(self) -> None:
        with pytest.raises(MemoryError):
            raise MemorySecurityError("tainted memory", code="MEMSEC_001")
