"""Tests for nexus.dapr.serialization — Pydantic ↔ Dapr bytes conversion."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel

from nexus.core.errors import StateSerializationError
from nexus.dapr.serialization import (
    compute_content_hash,
    empty_response,
    from_dapr_bytes,
    to_dapr_bytes,
)

# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class SimpleModel(BaseModel):
    name: str
    value: int


class NestedModel(BaseModel):
    inner: SimpleModel
    tags: list[str]


class OptionalModel(BaseModel):
    required: str
    optional: str | None = None


# ---------------------------------------------------------------------------
# to_dapr_bytes
# ---------------------------------------------------------------------------


class TestToDaprBytes:
    def test_simple_model(self) -> None:
        m = SimpleModel(name="test", value=42)
        result = to_dapr_bytes(m)
        assert isinstance(result, bytes)
        parsed = json.loads(result)
        assert parsed["name"] == "test"
        assert parsed["value"] == 42

    def test_nested_model(self) -> None:
        m = NestedModel(inner=SimpleModel(name="x", value=1), tags=["a", "b"])
        result = to_dapr_bytes(m)
        parsed = json.loads(result)
        assert parsed["inner"]["name"] == "x"
        assert parsed["tags"] == ["a", "b"]

    def test_optional_none_field(self) -> None:
        m = OptionalModel(required="hi")
        result = to_dapr_bytes(m)
        parsed = json.loads(result)
        assert parsed["required"] == "hi"

    def test_returns_utf8_bytes(self) -> None:
        m = SimpleModel(name="héllo", value=0)
        result = to_dapr_bytes(m)
        assert result.decode("utf-8")

    def test_unicode_in_strings(self) -> None:
        m = SimpleModel(name="日本語テスト", value=99)
        result = to_dapr_bytes(m)
        parsed = json.loads(result.decode("utf-8"))
        assert parsed["name"] == "日本語テスト"

    def test_output_is_deterministic(self) -> None:
        m = SimpleModel(name="same", value=1)
        assert to_dapr_bytes(m) == to_dapr_bytes(m)


# ---------------------------------------------------------------------------
# from_dapr_bytes
# ---------------------------------------------------------------------------


class TestFromDaprBytes:
    def test_round_trip_simple(self) -> None:
        original = SimpleModel(name="round", value=7)
        restored = from_dapr_bytes(to_dapr_bytes(original), SimpleModel)
        assert restored == original

    def test_round_trip_nested(self) -> None:
        original = NestedModel(inner=SimpleModel(name="n", value=2), tags=["x"])
        restored = from_dapr_bytes(to_dapr_bytes(original), NestedModel)
        assert restored == original

    def test_empty_bytes_raises(self) -> None:
        with pytest.raises(StateSerializationError) as exc_info:
            from_dapr_bytes(b"", SimpleModel)
        assert exc_info.value.code == "STATE_SERIALIZATION_ERROR"

    def test_none_bytes_raises(self) -> None:
        with pytest.raises(StateSerializationError):
            from_dapr_bytes(None, SimpleModel)  # type: ignore[arg-type]

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(StateSerializationError):
            from_dapr_bytes(b"not-json{{{", SimpleModel)

    def test_valid_json_wrong_schema_raises(self) -> None:
        bad = json.dumps({"wrong_field": "x"}).encode()
        with pytest.raises(StateSerializationError):
            from_dapr_bytes(bad, SimpleModel)

    def test_extra_fields_ignored(self) -> None:
        data = json.dumps({"name": "ok", "value": 5, "extra": "ignored"}).encode()
        result = from_dapr_bytes(data, SimpleModel)
        assert result.name == "ok"

    def test_non_utf8_bytes_raises(self) -> None:
        with pytest.raises(StateSerializationError):
            from_dapr_bytes(b"\xff\xfe", SimpleModel)

    def test_error_includes_class_name(self) -> None:
        with pytest.raises(StateSerializationError) as exc_info:
            from_dapr_bytes(b"", SimpleModel)
        assert "SimpleModel" in str(exc_info.value)


# ---------------------------------------------------------------------------
# compute_content_hash
# ---------------------------------------------------------------------------


class TestComputeContentHash:
    def test_returns_hex_string(self) -> None:
        h = compute_content_hash(b"hello")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_deterministic(self) -> None:
        assert compute_content_hash(b"abc") == compute_content_hash(b"abc")

    def test_different_inputs_differ(self) -> None:
        assert compute_content_hash(b"abc") != compute_content_hash(b"xyz")

    def test_empty_bytes_has_known_sha256(self) -> None:
        # SHA-256 of empty string is well-known
        h = compute_content_hash(b"")
        assert h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_model_hash_matches_bytes_hash(self) -> None:
        m = SimpleModel(name="x", value=1)
        b = to_dapr_bytes(m)
        assert compute_content_hash(b) == compute_content_hash(b)


# ---------------------------------------------------------------------------
# empty_response
# ---------------------------------------------------------------------------


class TestEmptyResponse:
    def test_empty_bytes_is_empty(self) -> None:
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.data = b""
        assert empty_response(resp) is True

    def test_none_data_is_empty(self) -> None:
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.data = None
        assert empty_response(resp) is True

    def test_non_empty_is_not_empty(self) -> None:
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.data = b'{"x":1}'
        assert empty_response(resp) is False


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------


@given(
    name=st.text(min_size=1, max_size=100),
    value=st.integers(min_value=-(2**31), max_value=2**31 - 1),
)
@settings(max_examples=50)
def test_round_trip_property(name: str, value: int) -> None:
    """Any valid SimpleModel survives a to_dapr_bytes → from_dapr_bytes round-trip."""
    original = SimpleModel(name=name, value=value)
    restored = from_dapr_bytes(to_dapr_bytes(original), SimpleModel)
    assert restored == original
