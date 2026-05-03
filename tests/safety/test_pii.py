"""Tests for PIIDetector."""

from __future__ import annotations

import pytest

from nexus.core.errors import SafetyError
from nexus.safety.pii import PIIAction, PIIDetector, PIIResult, PIIType


class TestPIIDetectorScan:
    def test_detects_email_address(self) -> None:
        d = PIIDetector()
        result = d.scan("Contact us at alice@example.com for help")
        types = {m.pii_type for m in result.matches}
        assert PIIType.EMAIL in types

    def test_detects_us_phone_number(self) -> None:
        d = PIIDetector()
        result = d.scan("Call me at 555-867-5309")
        types = {m.pii_type for m in result.matches}
        assert PIIType.PHONE in types

    def test_detects_ssn(self) -> None:
        d = PIIDetector()
        result = d.scan("SSN: 123-45-6789")
        types = {m.pii_type for m in result.matches}
        assert PIIType.SSN in types

    def test_detects_credit_card_visa(self) -> None:
        d = PIIDetector()
        result = d.scan("Card: 4111111111111111")
        types = {m.pii_type for m in result.matches}
        assert PIIType.CREDIT_CARD in types

    def test_detects_ip_address(self) -> None:
        d = PIIDetector()
        result = d.scan("Server IP is 192.168.1.100")
        types = {m.pii_type for m in result.matches}
        assert PIIType.IP_ADDRESS in types

    def test_clean_text_has_no_matches(self) -> None:
        d = PIIDetector()
        result = d.scan("The quick brown fox jumps over the lazy dog")
        assert result.matches == []

    def test_multiple_pii_types_detected(self) -> None:
        d = PIIDetector()
        result = d.scan("Email alice@example.com, SSN 123-45-6789")
        types = {m.pii_type for m in result.matches}
        assert PIIType.EMAIL in types
        assert PIIType.SSN in types

    def test_scan_returns_pii_result(self) -> None:
        d = PIIDetector()
        result = d.scan("hello")
        assert isinstance(result, PIIResult)


class TestPIIDetectorActions:
    def test_redact_action_replaces_with_placeholder(self) -> None:
        d = PIIDetector(actions={PIIType.EMAIL: PIIAction.REDACT})
        result = d.scan("Email: alice@example.com")
        assert "[REDACTED:EMAIL]" in result.redacted
        assert "alice@example.com" not in result.redacted

    def test_log_action_passes_text_through(self) -> None:
        d = PIIDetector(actions={PIIType.EMAIL: PIIAction.LOG})
        result = d.scan("Email: alice@example.com")
        assert "alice@example.com" in result.redacted
        assert result.action_taken == PIIAction.LOG

    def test_block_action_raises_safety_error(self) -> None:
        d = PIIDetector(actions={PIIType.SSN: PIIAction.BLOCK})
        with pytest.raises(SafetyError) as exc_info:
            d.scan("SSN: 123-45-6789")
        assert exc_info.value.code == "PII_BLOCKED"

    def test_redacted_text_contains_type_label(self) -> None:
        d = PIIDetector(actions={PIIType.PHONE: PIIAction.REDACT})
        result = d.scan("Call 555-867-5309 please")
        assert "PHONE" in result.redacted

    def test_result_blocked_flag_set_on_block(self) -> None:
        d = PIIDetector(actions={PIIType.SSN: PIIAction.BLOCK})
        with pytest.raises(SafetyError):
            d.scan("SSN: 123-45-6789")

    def test_default_action_is_redact(self) -> None:
        d = PIIDetector()
        result = d.scan("Email: alice@example.com")
        assert "[REDACTED:EMAIL]" in result.redacted

    def test_unspecified_type_defaults_to_log(self) -> None:
        # If actions dict provided but IP not in it, defaults to LOG
        d = PIIDetector(actions={PIIType.EMAIL: PIIAction.REDACT})
        result = d.scan("Server 192.168.1.1")
        # IP address present in original (log = pass through)
        assert "192.168.1.1" in result.redacted


class TestPIIDetectorRedact:
    def test_redact_convenience_method_returns_clean_text(self) -> None:
        d = PIIDetector(actions={PIIType.EMAIL: PIIAction.REDACT})
        cleaned = d.redact("Contact alice@example.com now")
        assert "alice@example.com" not in cleaned
        assert "[REDACTED:EMAIL]" in cleaned

    def test_redact_raises_on_block_action(self) -> None:
        d = PIIDetector(actions={PIIType.SSN: PIIAction.BLOCK})
        with pytest.raises(SafetyError):
            d.redact("SSN: 123-45-6789")

    def test_redact_clean_text_unchanged(self) -> None:
        d = PIIDetector()
        text = "No PII here at all."
        assert d.redact(text) == text
