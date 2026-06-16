"""Deterministic mock LLM client for running the demo without an API key.

Simulates the full multi-agent research workflow:
  - Supervisor: decomposes topic → calls tools → returns report
  - Researcher: searches sources → extracts claims → returns findings
  - FactChecker: verifies claims → returns confidence-scored results
  - Writer: formats content → saves report → returns final document

All responses are topic-aware (keywords injected from user input) and
fully deterministic — same input always produces the same output.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from grampus.core.models.base import ModelResponse
from grampus.core.types import Message, Role, StreamChunk, TokenUsage, ToolCall

_MOCK_MODEL = "mock-research-agent"


def _make_usage(input_t: int = 1200, output_t: int = 450) -> TokenUsage:
    return TokenUsage(
        input_tokens=input_t,
        output_tokens=output_t,
        total_tokens=input_t + output_t,
        cost_usd=round((input_t * 0.000003) + (output_t * 0.000015), 6),
        model=_MOCK_MODEL,
    )


def _resp(text: str | None = None, tool_calls: list[ToolCall] | None = None) -> ModelResponse:
    return ModelResponse(
        content=text,
        tool_calls=tool_calls or [],
        token_usage=_make_usage(),
        model=_MOCK_MODEL,
        stop_reason="end_turn" if not tool_calls else "tool_use",
    )


def _tc(name: str, args: dict[str, Any], idx: int = 1) -> ToolCall:
    return ToolCall(id=f"mock-{name}-{idx}", name=name, arguments=args)


_INJECTION_INDICATORS = [
    "ignore",
    "forget",
    "reveal",
    "system prompt",
    "pretend",
    "previous instructions",
    "disregard",
    "override",
]


def _is_injection(text: str) -> bool:
    lower = text.lower()
    return sum(1 for ind in _INJECTION_INDICATORS if ind in lower) >= 2


def _extract_topic(messages: list[Message]) -> str:
    """Return the research topic, extracted from the first user message or worker summary."""
    for msg in messages:
        if msg.role == Role.USER and msg.content:
            content = msg.content.strip()
            if "Worker results:" not in content:
                # Strip a leading markdown report title if workers forwarded it
                clean = re.sub(r"^#+\s*Research Report:\s*", "", content, flags=re.IGNORECASE)
                return clean[:120].strip()
    # Fallback: pull topic from inside the worker results block
    for msg in messages:
        if msg.role == Role.USER and msg.content and "Worker results:" in msg.content:
            m = re.search(r"Research Report:\s*([^\n#]{5,120})", msg.content)
            if m:
                return m.group(1).strip()
    return "artificial intelligence research"


def _identify_agent(messages: list[Message]) -> str:
    """Detect agent role from the system prompt."""
    for msg in messages:
        if msg.role == Role.SYSTEM and msg.content:
            c = msg.content.lower()
            if "supervisor" in c or "orchestrat" in c or "decompos" in c:
                return "supervisor"
            if "researcher" in c or "gather comprehensive" in c:
                return "researcher"
            if "fact" in c and ("check" in c or "verify" in c or "validat" in c):
                return "fact_checker"
            if "writer" in c or "synthesize" in c or "structured report" in c:
                return "writer"
    return "supervisor"


def _count_assistant_turns(messages: list[Message]) -> int:
    """Count previous assistant turns to determine current step."""
    return sum(1 for m in messages if m.role == Role.ASSISTANT)


def _has_worker_results(messages: list[Message]) -> bool:
    for msg in messages:
        if msg.role == Role.USER and msg.content and "Worker results:" in msg.content:
            return True
    return False


def _build_final_report(topic: str) -> str:
    """Return a realistic research report for any topic."""
    kw = topic.strip()
    if _is_injection(kw):
        kw = "artificial intelligence applications"
    return f"""# Research Report: {kw}

## Executive Summary

This comprehensive analysis examines the current state of research on **{kw}**, synthesizing findings from peer-reviewed literature, industry reports, and expert analyses. Three core themes emerge: foundational mechanisms, practical applications, and future trajectories. All claims have been verified against multiple independent sources with confidence scores reflecting evidence quality.

## Key Findings

### 1. Foundational Research Landscape
Peer-reviewed studies published in Nature, Science, and domain-specific journals confirm that {kw} has undergone significant advancement over the past five years. Meta-analyses of 150+ studies indicate a consistent upward trend in both research output (42% increase) and citation impact (3.1× improvement).

**Confidence Score: 0.89** | Sources: https://nature.com/articles/research-advances, https://science.org/doi/overview

### 2. Practical Applications and Industry Adoption
Leading organizations including research universities, Fortune 500 companies, and government agencies have initiated structured programmes around {kw}. Early deployments report efficiency gains of 15–40% compared to baseline approaches, with cost reductions averaging 23% in pilot programmes.

**Confidence Score: 0.76** | Sources: https://ieee.org/xplore/topic-review, https://mckinsey.com/research/technology

### 3. Emerging Opportunities
Analysis of patent filings (2022–2024) reveals a 67% increase in IP activity related to {kw}, with particular concentration in applications spanning healthcare, logistics, and environmental monitoring. Academic-industry partnerships have tripled since 2021, signalling growing commercial readiness.

**Confidence Score: 0.71** | Sources: https://arxiv.org/abs/recent-survey, https://patents.google.com/topic

## Detailed Analysis

The evidence base for {kw} reflects a field transitioning from foundational research to applied implementation. Key accelerants include technical maturity (stable algorithms, reproducible benchmarks), infrastructure readiness (cloud-managed services), regulatory clarity (EU/US/APAC guidance), and strong economic drivers (TAM $40B–$180B by 2030).

## Limitations & Caveats

This report synthesises publicly available research through the analysis date. Proprietary results from closed corporate R&D programmes are not reflected. Timeline projections carry inherent uncertainty and should be treated as indicative rather than definitive.

Confidence scores below 0.70 indicate areas where evidence is preliminary or contradictory. Readers should consult primary sources before making significant resource commitments in these areas.

## Citations

1. Smith, A. et al. (2024). "Advances in {kw}: A Systematic Review." *Nature Reviews*, 12(3), 145–167. https://nature.com/articles/research-advances
2. Johnson, B. & Lee, C. (2024). "Industrial Applications of {kw}." *Science*, 384, eadn0421. https://science.org/doi/overview
3. Williams, D. et al. (2023). "Technical Benchmarks for {kw} Systems." *IEEE Transactions*, 71(8), 4521–4538. https://ieee.org/xplore/topic-review
4. Chen, X. (2024). "Economic Impact Analysis: {kw}." *McKinsey Global Institute*. https://mckinsey.com/research/technology
5. Patel, R. et al. (2024). "Survey of Recent Progress in {kw}." *arXiv preprint*. https://arxiv.org/abs/recent-survey

## Confidence Assessment

| Claim | Confidence | Verification Status |
|-------|------------|---------------------|
| Research output growth (42%) | 0.89 | ✓ Peer-reviewed meta-analysis |
| Efficiency gains (15–40%) | 0.76 | ✓ Multiple industry pilots |
| Patent activity increase (67%) | 0.71 | ✓ Patent database analysis |
| Market size ($40B–$180B) | 0.62 | ~ Analyst projections vary |
| Timeline to mass adoption | 0.58 | ? Contested among experts |

---
*Report generated by Nexus Deep Research Agent | Word count: ~520 words | Reading time: ~3 minutes*
"""


class MockModelClient:
    """Stateless deterministic mock LLM that simulates the research workflow.

    No API key required. Identifies the calling agent from the system prompt,
    determines the current step from message history, and returns scripted
    but realistic tool calls and text responses.
    """

    async def complete(
        self,
        *,
        messages: list[Message],
        model: str = _MOCK_MODEL,
        tools: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **_: Any,
    ) -> ModelResponse:
        agent = _identify_agent(messages)
        step = _count_assistant_turns(messages)
        topic = _extract_topic(messages)

        if agent == "supervisor":
            return self._supervisor(step, topic, messages)
        if agent == "researcher":
            return self._researcher(step, topic)
        if agent == "fact_checker":
            return self._fact_checker(step, topic)
        if agent == "writer":
            return self._writer(step, topic)
        return self._supervisor(step, topic, messages)

    async def stream(
        self,
        *,
        messages: list[Message],
        model: str = _MOCK_MODEL,
        tools: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **_: Any,
    ) -> Any:
        """Yield mock response word-by-word to simulate streaming."""
        response = await self.complete(messages=messages, model=model, tools=tools)
        response_text = response.content or ""
        words = response_text.split() if response_text else []

        for i, word in enumerate(words):
            delta = word + (" " if i < len(words) - 1 else "")
            yield StreamChunk(delta=delta, model=_MOCK_MODEL)
            await asyncio.sleep(0)

        yield StreamChunk(
            delta="",
            finish_reason=response.stop_reason,
            token_usage=response.token_usage,
            model=_MOCK_MODEL,
            is_final=True,
        )

    # ------------------------------------------------------------------
    # Per-agent response sequences
    # ------------------------------------------------------------------

    def _supervisor(self, step: int, topic: str, messages: list[Message]) -> ModelResponse:
        # Crew review turn: supervisor has already seen worker outputs
        if _has_worker_results(messages):
            return _resp(_build_final_report(topic))

        # Single-agent / eval mode: supervisor runs full research pipeline
        if step == 0:
            return _resp(tool_calls=[_tc("web_search", {"query": topic, "max_results": 3})])
        if step == 1:
            return _resp(
                tool_calls=[
                    _tc(
                        "score_credibility",
                        {
                            "source_url": "https://nature.com/articles/example",
                            "content": f"Research on {topic} shows significant advances.",
                        },
                        2,
                    )
                ]
            )
        if step == 2:
            return _resp(
                tool_calls=[
                    _tc(
                        "extract_claims",
                        {"text": f"Studies confirm that {topic} delivers measurable benefits."},
                        3,
                    )
                ]
            )
        if step == 3:
            return _resp(
                tool_calls=[
                    _tc(
                        "format_markdown",
                        {
                            "sections": {
                                "Executive Summary": f"Analysis of {topic} reveals significant advances.",
                                "Key Findings": "Multiple verified sources confirm high confidence scores.",
                                "Detailed Analysis": "Evidence supports strong adoption trends.",
                                "Limitations & Caveats": "Further research needed in specific sub-domains.",
                                "Citations": "1. https://nature.com/example  2. https://science.org/example",
                            }
                        },
                        4,
                    )
                ]
            )
        if step == 4:
            return _resp(
                tool_calls=[_tc("word_count", {"text": f"Research report on {topic}. " * 50}, 5)]
            )
        if step == 5:
            return _resp(
                tool_calls=[
                    _tc(
                        "save_report",
                        {
                            "title": f"Research Report: {topic}",
                            "content": _build_final_report(topic),
                        },
                        6,
                    )
                ]
            )
        return _resp(_build_final_report(topic))

    def _researcher(self, step: int, topic: str) -> ModelResponse:
        if step == 0:
            return _resp(
                tool_calls=[
                    _tc("web_search", {"query": f"{topic} research advances", "max_results": 4})
                ]
            )
        if step == 1:
            return _resp(
                tool_calls=[
                    _tc(
                        "extract_claims",
                        {
                            "text": f"Recent studies on {topic} show significant progress across multiple domains."
                        },
                        2,
                    )
                ]
            )
        return _resp(
            f"Research findings for '{topic}':\n\n"
            "Sources reviewed: 4 peer-reviewed papers, 2 industry reports\n\n"
            "Key claims identified:\n"
            f"1. {topic} demonstrates measurable improvement over baseline (confidence: 0.87)\n"
            f"2. Adoption rate has increased 42% year-over-year (confidence: 0.79)\n"
            f"3. Cost reduction of 15-30% observed in pilot deployments (confidence: 0.71)\n\n"
            "Top sources:\n"
            "- https://nature.com/articles/advances (credibility: 0.92)\n"
            "- https://arxiv.org/abs/recent-paper (credibility: 0.85)"
        )

    def _fact_checker(self, step: int, topic: str) -> ModelResponse:
        if step == 0:
            return _resp(
                tool_calls=[
                    _tc(
                        "web_search",
                        {"query": f"verify {topic} claims evidence", "max_results": 3},
                        1,
                    )
                ]
            )
        return _resp(
            f"Fact-checking results for '{topic}':\n\n"
            "Verified claims:\n"
            f"✓ [0.89] {topic} shows measurable improvement — CONFIRMED (3 corroborating sources)\n"
            f"✓ [0.76] Adoption increased significantly — CONFIRMED (industry data)\n"
            f"~ [0.64] Cost reduction estimates — PARTIALLY VERIFIED (ranges vary by context)\n"
            f"✗ [0.41] Specific timeline claims — UNVERIFIED (insufficient evidence)\n\n"
            "Overall verification confidence: 0.74"
        )

    def _writer(self, step: int, topic: str) -> ModelResponse:
        report = _build_final_report(topic)
        if step == 0:
            return _resp(
                tool_calls=[
                    _tc(
                        "format_markdown",
                        {
                            "sections": {
                                "Executive Summary": f"Comprehensive analysis of {topic}.",
                                "Key Findings": "Verified claims with confidence scores.",
                                "Detailed Analysis": "Evidence from peer-reviewed sources.",
                                "Limitations & Caveats": "Areas requiring further research.",
                                "Citations": "https://nature.com/example https://science.org/example",
                            }
                        },
                        1,
                    )
                ]
            )
        if step == 1:
            return _resp(tool_calls=[_tc("word_count", {"text": report}, 2)])
        if step == 2:
            return _resp(
                tool_calls=[
                    _tc(
                        "save_report",
                        {"title": f"Research Report: {topic}", "content": report},
                        3,
                    )
                ]
            )
        return _resp(report)
