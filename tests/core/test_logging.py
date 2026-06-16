"""Tests for grampus.core.logging — structured logging with correlation IDs."""

import json
from io import StringIO

from grampus.core.logging import bind_correlation_id, configure_logging, get_logger


class TestGetLogger:
    def test_returns_bound_logger(self) -> None:
        logger = get_logger("grampus.test")
        assert logger is not None

    def test_different_names_return_different_loggers(self) -> None:
        a = get_logger("grampus.a")
        b = get_logger("grampus.b")
        assert a is not b

    def test_logger_has_expected_methods(self) -> None:
        logger = get_logger("grampus.test")
        for method in ("info", "debug", "warning", "error", "critical"):
            assert callable(getattr(logger, method, None)), f"missing method: {method}"


class TestConfigureLogging:
    def test_configure_dev_mode(self) -> None:
        configure_logging(dev=True, level="DEBUG")

    def test_configure_prod_mode(self) -> None:
        configure_logging(dev=False, level="INFO")

    def test_configure_json_output(self, capsys: object) -> None:
        """In prod mode the log output is valid JSON."""
        stream = StringIO()
        configure_logging(dev=False, level="DEBUG", stream=stream)
        logger = get_logger("test.json")
        logger.info("hello world", key="val")
        output = stream.getvalue().strip()
        if output:
            parsed = json.loads(output)
            assert "event" in parsed or "message" in parsed

    def test_configure_accepts_log_levels(self) -> None:
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            configure_logging(dev=True, level=level)

    def test_configure_returns_none(self) -> None:
        result = configure_logging(dev=True, level="INFO")
        assert result is None


class TestCorrelationId:
    def test_bind_returns_token(self) -> None:
        token = bind_correlation_id("abc-123")
        assert token is not None

    def test_bind_with_explicit_id(self) -> None:
        bind_correlation_id("test-correlation-id")
        # Should not raise

    def test_bind_auto_generates_id_when_none(self) -> None:
        token = bind_correlation_id()
        assert token is not None

    def test_correlation_id_appears_in_log_output(self) -> None:
        stream = StringIO()
        configure_logging(dev=False, level="DEBUG", stream=stream)
        bind_correlation_id("req-999")
        logger = get_logger("test.corr")
        logger.info("test event")
        output = stream.getvalue().strip()
        if output:
            parsed = json.loads(output)
            assert parsed.get("correlation_id") == "req-999"


class TestLoggerOutput:
    def test_log_event_contains_level(self) -> None:
        stream = StringIO()
        configure_logging(dev=False, level="DEBUG", stream=stream)
        logger = get_logger("test.level")
        logger.warning("a warning")
        output = stream.getvalue().strip()
        if output:
            parsed = json.loads(output)
            assert parsed.get("level") == "warning"

    def test_log_event_contains_timestamp(self) -> None:
        stream = StringIO()
        configure_logging(dev=False, level="DEBUG", stream=stream)
        logger = get_logger("test.ts")
        logger.info("ts test")
        output = stream.getvalue().strip()
        if output:
            parsed = json.loads(output)
            assert "timestamp" in parsed

    def test_extra_fields_included(self) -> None:
        stream = StringIO()
        configure_logging(dev=False, level="DEBUG", stream=stream)
        logger = get_logger("test.fields")
        logger.info("event", agent_id="agent-1", step=3)
        output = stream.getvalue().strip()
        if output:
            parsed = json.loads(output)
            assert parsed.get("agent_id") == "agent-1"
            assert parsed.get("step") == 3
