"""Tests for nexus.memory.provenance — SourceType, Provenance, ProvenanceTracker."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime

import pytest

from nexus.memory.provenance import Provenance, ProvenanceTracker, SourceType


class TestSourceType:
    def test_is_str(self) -> None:
        assert isinstance(SourceType.SYSTEM, str)

    def test_system_trust_level(self) -> None:
        assert SourceType.SYSTEM.trust_level == 1.0

    def test_user_input_trust_level(self) -> None:
        assert SourceType.USER_INPUT.trust_level == 0.9

    def test_llm_generated_trust_level(self) -> None:
        assert SourceType.LLM_GENERATED.trust_level == 0.7

    def test_tool_result_trust_level(self) -> None:
        assert SourceType.TOOL_RESULT.trust_level == 0.6

    def test_external_data_trust_level(self) -> None:
        assert SourceType.EXTERNAL_DATA.trust_level == 0.3

    def test_all_trust_levels_in_unit_interval(self) -> None:
        for st in SourceType:
            assert 0.0 <= st.trust_level <= 1.0


class TestProvenance:
    def test_fields_present(self) -> None:
        p = Provenance(
            id=str(uuid.uuid4()),
            source_type=SourceType.SYSTEM,
            source_id="src",
            trust_level=1.0,
            timestamp=datetime.now(UTC),
            content_hash_sha256="deadbeef",
        )
        assert p.source_type == SourceType.SYSTEM
        assert p.source_id == "src"
        assert p.trust_level == 1.0
        assert p.content_hash_sha256 == "deadbeef"

    def test_json_round_trip(self) -> None:
        p = Provenance(
            id=str(uuid.uuid4()),
            source_type=SourceType.USER_INPUT,
            source_id="user-42",
            trust_level=0.9,
            timestamp=datetime.now(UTC),
            content_hash_sha256="abc123",
        )
        restored = Provenance.model_validate_json(p.model_dump_json())
        assert restored.source_type == SourceType.USER_INPUT
        assert restored.trust_level == 0.9
        assert restored.source_id == "user-42"

    def test_id_has_default(self) -> None:
        p = Provenance(
            source_type=SourceType.SYSTEM,
            source_id="s",
            trust_level=1.0,
            timestamp=datetime.now(UTC),
            content_hash_sha256="x",
        )
        assert len(p.id) == 36  # UUID v4 string length


class TestProvenanceTracker:
    def test_create_returns_provenance(self) -> None:
        tracker = ProvenanceTracker()
        p = tracker.create("hello world", SourceType.USER_INPUT, source_id="u1")
        assert isinstance(p, Provenance)

    def test_create_sets_source_type(self) -> None:
        tracker = ProvenanceTracker()
        p = tracker.create("c", SourceType.LLM_GENERATED, source_id="a")
        assert p.source_type == SourceType.LLM_GENERATED

    def test_create_sets_trust_level_from_source_type(self) -> None:
        tracker = ProvenanceTracker()
        p = tracker.create("c", SourceType.TOOL_RESULT, source_id="t")
        assert p.trust_level == pytest.approx(0.6)

    def test_create_sets_source_id(self) -> None:
        tracker = ProvenanceTracker()
        p = tracker.create("c", SourceType.SYSTEM, source_id="system-42")
        assert p.source_id == "system-42"

    def test_create_computes_sha256_hash(self) -> None:
        tracker = ProvenanceTracker()
        content = "test content for hashing"
        p = tracker.create(content, SourceType.USER_INPUT, source_id="u")
        expected = hashlib.sha256(content.encode()).hexdigest()
        assert p.content_hash_sha256 == expected

    def test_create_assigns_unique_ids(self) -> None:
        tracker = ProvenanceTracker()
        p1 = tracker.create("c", SourceType.SYSTEM, source_id="s")
        p2 = tracker.create("c", SourceType.SYSTEM, source_id="s")
        assert p1.id != p2.id

    def test_create_timestamp_is_utc_aware(self) -> None:
        tracker = ProvenanceTracker()
        p = tracker.create("c", SourceType.SYSTEM, source_id="s")
        assert p.timestamp.tzinfo is not None

    def test_verify_returns_true_for_matching_content(self) -> None:
        tracker = ProvenanceTracker()
        content = "exact match content"
        p = tracker.create(content, SourceType.USER_INPUT, source_id="u")
        assert tracker.verify(content, p) is True

    def test_verify_returns_false_for_tampered_content(self) -> None:
        tracker = ProvenanceTracker()
        p = tracker.create("original", SourceType.USER_INPUT, source_id="u")
        assert tracker.verify("tampered", p) is False

    def test_verify_returns_false_for_empty_vs_nonempty(self) -> None:
        tracker = ProvenanceTracker()
        p = tracker.create("not empty", SourceType.SYSTEM, source_id="s")
        assert tracker.verify("", p) is False

    def test_verify_empty_content_matches_empty_provenance(self) -> None:
        tracker = ProvenanceTracker()
        p = tracker.create("", SourceType.SYSTEM, source_id="s")
        assert tracker.verify("", p) is True

    def test_create_external_data_trust_level(self) -> None:
        tracker = ProvenanceTracker()
        p = tracker.create("c", SourceType.EXTERNAL_DATA, source_id="ext")
        assert p.trust_level == pytest.approx(0.3)