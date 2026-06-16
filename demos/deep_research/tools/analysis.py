"""Analysis tools: extract_claims, score_credibility, summarize_source.

All implementations are self-contained (no external ML dependencies).
- extract_claims uses regex + linguistic heuristics
- score_credibility uses domain authority + content signals
- summarize_source uses TF-IDF-style extractive summarisation
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any
from urllib.parse import urlparse

from grampus.core.types import ToolParameter
from grampus.tools.registry import ToolRegistry

_registry = ToolRegistry()

# ---------------------------------------------------------------------------
# Claim extraction heuristics
# ---------------------------------------------------------------------------

_CLAIM_SIGNALS = [
    r"studies? (?:show|suggest|indicate|demonstrate|confirm|reveal)",
    r"research (?:shows?|suggests?|indicates?|demonstrates?|confirms?|reveals?)",
    r"according to",
    r"data (?:shows?|suggests?|indicates?|demonstrates?)",
    r"results? (?:show|suggest|indicate|demonstrate)",
    r"\b(?:first|largest?|fastest?|most|highest?|lowest?)\b",
    r"\b\d+(?:\.\d+)?%\b",
    r"\b\d+(?:\.\d+)?×\b",
    r"\bproven?\b",
    r"\bconfirmed?\b",
    r"\bmeasur(?:ed?|able)\b",
    r"\bsignificant(?:ly)?\b",
    r"\bimprove(?:ment|d|s)?\b.*\b\d",
]
_CLAIM_RE = re.compile("|".join(_CLAIM_SIGNALS), re.IGNORECASE)

_UNCERTAINTY_WORDS = {"may", "might", "could", "possibly", "perhaps", "likely", "suggest"}
_STRONG_WORDS = {"confirm", "demonstrate", "prove", "establish", "show"}

# ---------------------------------------------------------------------------
# Domain credibility tiers
# ---------------------------------------------------------------------------

_HIGH_CRED_DOMAINS = {
    "nature.com",
    "science.org",
    "thelancet.com",
    "nejm.org",
    "cell.com",
    "pubmed.ncbi.nlm.nih.gov",
    "who.int",
    "cdc.gov",
    "nih.gov",
    "nsf.gov",
    "ieee.org",
    "acm.org",
    "arxiv.org",
    "biorxiv.org",
}
_MEDIUM_CRED_DOMAINS = {
    "nytimes.com",
    "theguardian.com",
    "bbc.com",
    "reuters.com",
    "apnews.com",
    "wired.com",
    "techcrunch.com",
    "ibm.com",
    "google.com",
    "microsoft.com",
    "mckinsey.com",
    "gartner.com",
    "deloitte.com",
    "hbr.org",
}
_BLOG_INDICATORS = {"blog", "medium.com", "substack", "wordpress", "blogspot"}


def _domain_score(url: str) -> float:
    try:
        netloc = urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return 0.3
    if any(netloc.endswith(d) or netloc == d for d in _HIGH_CRED_DOMAINS):
        return 0.92
    if any(netloc.endswith(d) or netloc == d for d in _MEDIUM_CRED_DOMAINS):
        return 0.74
    if netloc.endswith(".edu") or netloc.endswith(".gov") or netloc.endswith(".org"):
        return 0.85
    if any(ind in netloc for ind in _BLOG_INDICATORS):
        return 0.42
    return 0.55


def _content_score(content: str) -> tuple[float, list[str]]:
    """Return (score, factors) based on content signals."""
    factors: list[str] = []
    score = 0.5

    # Citation density
    citations = len(re.findall(r"(?:doi|https?://|et al\.|References?)", content, re.IGNORECASE))
    if citations >= 5:
        score += 0.12
        factors.append("high citation density")
    elif citations >= 2:
        score += 0.06
        factors.append("moderate citation density")

    # Methodology mention
    if re.search(
        r"\b(?:method|study design|participants?|sample size|p-value|CI)\b", content, re.IGNORECASE
    ):
        score += 0.08
        factors.append("methodology described")

    # Hedging language ratio
    words = content.lower().split()
    hedge_count = sum(1 for w in words if w in {"may", "might", "could", "possibly", "perhaps"})
    hedge_ratio = hedge_count / max(len(words), 1)
    if hedge_ratio > 0.02:
        score -= 0.05
        factors.append("high hedging language")

    # Superlative / promotional language
    promos = len(
        re.findall(
            r"\b(?:revolutionary|groundbreaking|unprecedented|game.changing)\b",
            content,
            re.IGNORECASE,
        )
    )
    if promos >= 3:
        score -= 0.1
        factors.append("promotional language detected")

    # Author credentials
    if re.search(r"\bPh\.?D\.?\b|\bM\.?D\.?\b|\bDr\.?\b|\bProfessor\b", content):
        score += 0.04
        factors.append("author credentials mentioned")

    return min(max(score, 0.1), 1.0), factors


# ---------------------------------------------------------------------------
# TF-IDF extractive summariser (no external deps)
# ---------------------------------------------------------------------------


def _tokenise(text: str) -> list[str]:
    return re.findall(r"\b[a-z]{3,}\b", text.lower())


def _compute_tfidf(sentences: list[str]) -> list[tuple[int, float]]:
    """Return (sentence_index, score) pairs sorted by score descending."""
    if not sentences:
        return []
    n_docs = len(sentences)
    term_doc_freq: Counter[str] = Counter()
    sent_terms = [_tokenise(s) for s in sentences]

    for terms in sent_terms:
        for t in set(terms):
            term_doc_freq[t] += 1

    scores: list[tuple[int, float]] = []
    for idx, terms in enumerate(sent_terms):
        tf = Counter(terms)
        score = sum(
            (tf[t] / max(len(terms), 1)) * math.log(n_docs / (term_doc_freq[t] + 1)) for t in tf
        )
        scores.append((idx, score))

    return sorted(scores, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


@_registry.tool(
    name="extract_claims",
    description="Extract factual claims from text, assigning confidence scores to each.",
    parameters=[
        ToolParameter(
            name="text", type="string", description="Source text to analyse", required=True
        ),
        ToolParameter(
            name="min_confidence",
            type="number",
            description="Minimum confidence threshold (0.0–1.0)",
            required=False,
            default=0.5,
        ),
    ],
)
async def extract_claims(text: str, min_confidence: float = 0.5) -> list[dict[str, Any]]:
    """Extract factual claims using regex + linguistic heuristics."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims: list[dict[str, Any]] = []

    for sent in sentences:
        if len(sent.split()) < 6:
            continue
        if not _CLAIM_RE.search(sent):
            continue

        words_lower = sent.lower().split()
        uncertainty = sum(1 for w in words_lower if w in _UNCERTAINTY_WORDS)
        strength = sum(1 for w in words_lower if w in _STRONG_WORDS)
        has_number = bool(re.search(r"\d", sent))

        base = 0.55
        base += strength * 0.08
        base -= uncertainty * 0.06
        base += 0.10 if has_number else 0.0
        confidence = min(max(round(base, 2), 0.1), 0.99)

        if confidence < min_confidence:
            continue

        quote = sent[:120] + ("..." if len(sent) > 120 else "")
        claims.append(
            {
                "claim": sent,
                "confidence": confidence,
                "requires_verification": confidence < 0.75,
                "source_quote": quote,
            }
        )

    if not claims:
        claims.append(
            {
                "claim": text[:200],
                "confidence": 0.50,
                "requires_verification": True,
                "source_quote": text[:100],
            }
        )

    return claims[:10]


@_registry.tool(
    name="score_credibility",
    description="Score the credibility of a source based on its URL and content.",
    parameters=[
        ToolParameter(
            name="source_url", type="string", description="URL of the source", required=True
        ),
        ToolParameter(
            name="content", type="string", description="Page content to analyse", required=True
        ),
    ],
)
async def score_credibility(source_url: str, content: str) -> dict[str, Any]:
    """Compute credibility score from domain authority and content signals."""
    dom_score = _domain_score(source_url)
    cont_score, factors = _content_score(content)

    overall = round((dom_score * 0.6 + cont_score * 0.4), 3)
    return {
        "overall_score": overall,
        "domain_score": round(dom_score, 3),
        "content_score": round(cont_score, 3),
        "factors": factors or ["standard content quality"],
    }


@_registry.tool(
    name="summarize_source",
    description="Produce an extractive summary of page content with key points and sentiment.",
    parameters=[
        ToolParameter(
            name="content", type="string", description="Text content to summarise", required=True
        ),
        ToolParameter(
            name="max_words",
            type="number",
            description="Target word count for the summary",
            required=False,
            default=150,
        ),
    ],
)
async def summarize_source(content: str, max_words: int = 150) -> dict[str, Any]:
    """Extractive summarisation using TF-IDF sentence scoring."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if len(s.split()) >= 5]
    if not sentences:
        return {"summary": content[:300], "key_points": [], "sentiment": "neutral"}

    ranked = _compute_tfidf(sentences)
    top_n = max(3, min(8, max_words // 30))
    selected_indices = sorted(idx for idx, _ in ranked[:top_n])
    summary_sentences = [sentences[i] for i in selected_indices if i < len(sentences)]

    word_limit = max_words
    summary_words: list[str] = []
    for sent in summary_sentences:
        words = sent.split()
        if len(summary_words) + len(words) > word_limit:
            break
        summary_words.extend(words)
    summary = " ".join(summary_words)

    key_points = [sentences[idx] for idx, _ in ranked[:3] if idx < len(sentences)]

    # Simple sentiment: count positive vs negative signal words
    pos = len(
        re.findall(
            r"\b(?:improve|advance|benefit|success|achieve|breakthrough|progress)\b",
            content,
            re.IGNORECASE,
        )
    )
    neg = len(
        re.findall(
            r"\b(?:fail|limit|challenge|barrier|problem|issue|risk)\b", content, re.IGNORECASE
        )
    )
    if pos > neg * 1.5:
        sentiment = "positive"
    elif neg > pos * 1.5:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {"summary": summary, "key_points": key_points[:3], "sentiment": sentiment}
