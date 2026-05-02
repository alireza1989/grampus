"""Tests for nexus.memory.types — EpisodicRecord and RetrievedRecord."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nexus.memory.types import EpisodicRecord, RetrievedRecord


def make_record(**kwargs: object) -> EpisodicRecord:
    defaults: dict[str, object] = {
        "id": "rec-001",
        "agent_id": "agent-1",
        "session_id": "session-1",
        "content": "The user prefers dark mode.",
        "trust_score": 0.9,
        "importance_score": 0.5,
        "access_count": 0,
    }
    defaults.update(kwargs)
    return EpisodicRecord(**defaults)  # type: ignore[arg-type]


class TestEpisodicRecord:
    def test_minimal_construction(self) -> None:
        r = make_record()
        assert r.id == "rec-001"
        assert r.agent_id == "agent-1"
        assert r.content == "The user prefers dark mode."

    def test_optional_user_id_defaults_none(self) -> None:
        r = make_record()
        assert r.user_id is None

    def test_optional_embedding_defaults_none(self) -> None:
        r = make_record()
        assert r.embedding is None

    def test_optional_provenance_defaults_none(self) -> None:
        r = make_record()
        assert r.provenance is None

    def test_optional_last_accessed_defaults_none(self) -> None:
        r = make_record()
        assert r.last_accessed is None

    def test_metadata_defaults_empty_dict(self) -> None:
        r = make_record()
        assert r.metadata == {}

    def test_access_count_defaults_zero(self) -> None:
        r = make_record()
        assert r.access_count == 0

    def test_timestamp_auto_set(self) -> None:
        r = make_record()
        assert r.timestamp is not None
        assert r.timestamp.tzinfo is not None

    def test_trust_score_rejects_above_one(self) -> None:
        with pytest.raises(ValidationError):
            make_record(trust_score=1.1)

    def test_trust_score_rejects_below_zero(self) -> None:
        with pytest.raises(ValidationError):
            make_record(trust_score=-0.1)

    def test_importance_score_rejects_above_one(self) -> None:
        with pytest.raises(ValidationError):
            make_record(importance_score=1.5)

    def test_importance_score_rejects_below_zero(self) -> None:
        with pytest.raises(ValidationError):
            make_record(importance_score=-0.5)

    def test_trust_score_boundary_zero(self) -> None:
        r = make_record(trust_score=0.0)
        assert r.trust_score == 0.0

    def test_trust_score_boundary_one(self) -> None:
        r = make_record(trust_score=1.0)
        assert r.trust_score == 1.0

    def test_embedding_stores_floats(self) -> None:
        emb = [0.1, 0.2, 0.3]
        r = make_record(embedding=emb)
        assert r.embedding == emb

    def test_with_user_id(self) -> None:
        r = make_record(user_id="user-42")
        assert r.user_id == "user-42"

    def test_round_trip_json(self) -> None:
        r = make_record(embedding=[0.1, 0.2], user_id="u1", provenance="tool")
        data = r.model_dump_json()
        restored = EpisodicRecord.model_validate_json(data)
        assert restored == r

    def test_metadata_stores_arbitrary_values(self) -> None:
        r = make_record(metadata={"source": "conversation", "turn": 3})
        assert r.metadata["source"] == "conversation"
        assert r.metadata["turn"] == 3


class TestRetrievedRecord:
    def test_construction(self) -> None:
        rec = make_record()
        retrieved = RetrievedRecord(
            record=rec,
            score=0.75,
            recency_score=0.8,
            similarity_score=0.7,
            importance_score=0.6,
        )
        assert retrieved.score == 0.75
        assert retrieved.record is rec

    def test_score_rejects_above_one(self) -> None:
        rec = make_record()
        with pytest.raises(ValidationError):
            RetrievedRecord(
                record=rec,
                score=1.5,
                recency_score=0.8,
                similarity_score=0.7,
                importance_score=0.6,
            )

    def test_score_rejects_below_zero(self) -> None:
        rec = make_record()
        with pytest.raises(ValidationError):
            RetrievedRecord(
                record=rec,
                score=-0.1,
                recency_score=0.8,
                similarity_score=0.7,
                importance_score=0.6,
            )

    def test_round_trip_json(self) -> None:
        rec = make_record()
        retrieved = RetrievedRecord(
            record=rec,
            score=0.5,
            recency_score=0.6,
            similarity_score=0.4,
            importance_score=0.5,
        )
        data = retrieved.model_dump_json()
        restored = RetrievedRecord.model_validate_json(data)
        assert restored == retrieved
