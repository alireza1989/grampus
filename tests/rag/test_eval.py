"""Tests for demos.rag.eval — pure unit tests, no LLM calls."""

from __future__ import annotations

import json
from pathlib import Path


def test_sample_qa_file_is_valid_json() -> None:
    qa_path = Path(__file__).parents[2] / "demos" / "rag" / "eval" / "sample_qa.json"
    data = json.loads(qa_path.read_text())
    assert isinstance(data, list)
    assert len(data) > 0
    for item in data:
        assert "id" in item
        assert "question" in item
        assert "expected_topics" in item


def test_faithfulness_prompt_contains_context() -> None:
    from demos.rag.eval.rag_eval import _build_faithfulness_prompt

    result = _build_faithfulness_prompt(context="ctx value", answer="ans value")
    assert "ctx value" in result
    assert "ans value" in result


def test_relevancy_prompt_contains_question() -> None:
    from demos.rag.eval.rag_eval import _build_relevancy_prompt

    result = _build_relevancy_prompt(question="q?", answer="ans value")
    assert "q?" in result
    assert "ans value" in result


def test_parse_score_valid() -> None:
    from demos.rag.eval.rag_eval import _parse_score

    assert _parse_score("Score: 4/5") == 4.0


def test_parse_score_fallback() -> None:
    from demos.rag.eval.rag_eval import _parse_score

    result = _parse_score("I would say 3")
    assert isinstance(result, float)
