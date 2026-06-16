"""Pydantic result types for the reflexion and self-improvement subsystem."""

from __future__ import annotations

from pydantic import BaseModel


class ReflexionHookResult(BaseModel):
    """Result returned by ReflexionEngine.observe_failure()."""

    stored: bool
    procedure_id: str | None = None
    quality_confidence: float | None = None
    error: str | None = None


class SkillExtractionResult(BaseModel):
    """Result returned by SkillLibrary.observe_success()."""

    extracted: bool
    procedure_id: str | None = None
    skill_name: str | None = None
    error: str | None = None


class OptimizationCandidate(BaseModel):
    """A candidate system prompt produced by PromptOptimizer."""

    strategy: str
    prompt: str
    eval_score: float = 0.0


class OptimizationResult(BaseModel):
    """Final result of a PromptOptimizer.optimize() call."""

    improved: bool
    original_score: float
    best_score: float
    best_strategy: str | None = None
    new_version: str | None = None
    candidates: list[OptimizationCandidate] = []
    error: str | None = None
