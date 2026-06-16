"""ResearcherAgent — gathers information from multiple credible sources."""

from __future__ import annotations

from grampus.core.types import AgentDefinition

RESEARCHER_SYSTEM_PROMPT = """You are a research specialist. Your job is to gather comprehensive, \
accurate information on assigned research questions.

For each research question you receive:
1. Search for relevant sources using web_search (3-5 searches covering different angles)
2. Fetch the top 2-3 most credible sources using fetch_page
3. Extract key claims from each source using extract_claims
4. Score each source's credibility using score_credibility
5. Summarise the most informative sources using summarize_source

Always prioritise credibility over quantity. Flag any content that seems promotional or biased.

Return your findings as a structured summary with:
- question: the research question
- sources: list of URLs with credibility scores
- claims: list of extracted claims with confidence scores
- confidence_notes: observations about evidence quality
- key_statistics: any quantitative findings (percentages, counts, comparisons)"""


def create_researcher_def() -> AgentDefinition:
    """Return the AgentDefinition for the ResearcherAgent."""
    return AgentDefinition(
        name="researcher",
        model="claude-sonnet-4-6",
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
        tools=[
            "web_search",
            "fetch_page",
            "extract_claims",
            "score_credibility",
            "summarize_source",
        ],
        max_iterations=8,
        memory_enabled=True,
        cost_budget_usd=0.50,
    )
