"""Search tools: web_search, fetch_page, trending_topics.

All tools return deterministic mock data based on query keywords so that
the same input always produces the same output (safe for CI and evals).
fetch_page runs page content through the SafetyPipeline before returning
because web content is the primary prompt-injection attack surface.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from nexus.core.types import ToolParameter, ToolResult
from nexus.safety.injection import DetectionLevel, PromptInjectionDetector
from nexus.safety.pipeline import SafetyPipeline
from nexus.tools.registry import ToolRegistry

_registry = ToolRegistry()

# Module-level safety pipeline for fetch_page content scanning
_safety = SafetyPipeline(injection_detector=PromptInjectionDetector(level=DetectionLevel.BALANCED))

# ---------------------------------------------------------------------------
# Mock knowledge base — topic → results
# ---------------------------------------------------------------------------

_TOPIC_RESULTS: dict[str, list[dict[str, Any]]] = {
    "quantum": [
        {
            "title": "Quantum Computing Advances in Drug Discovery",
            "url": "https://nature.com/articles/quantum-drug-discovery-2024",
            "snippet": "Researchers demonstrate quantum algorithms that outperform classical simulations for molecular docking problems, reducing computation time by 100×.",
            "date": "2024-03-15",
            "source_credibility": 0.94,
        },
        {
            "title": "IBM Quantum Achieves 99.5% Gate Fidelity",
            "url": "https://ibm.com/research/quantum-fidelity-2024",
            "snippet": "IBM's latest 433-qubit system reaches unprecedented fidelity levels, enabling practical quantum advantage for chemistry simulations.",
            "date": "2024-02-20",
            "source_credibility": 0.91,
        },
        {
            "title": "Variational Quantum Eigensolvers for Pharmaceutical Research",
            "url": "https://arxiv.org/abs/2401.09872",
            "snippet": "VQE algorithms applied to protein folding problems show 3× improvement in binding affinity prediction accuracy over classical methods.",
            "date": "2024-01-28",
            "source_credibility": 0.88,
        },
        {
            "title": "Quantum Machine Learning in ADMET Prediction",
            "url": "https://pubmed.ncbi.nlm.nih.gov/quantum-admet-2024",
            "snippet": "Quantum neural networks demonstrate 15-40% accuracy improvement for absorption, distribution, metabolism prediction tasks.",
            "date": "2024-04-10",
            "source_credibility": 0.87,
        },
        {
            "title": "Challenges in Near-Term Quantum Computing",
            "url": "https://science.org/doi/quantum-nisq-challenges",
            "snippet": "NISQ-era devices remain limited by decoherence and error rates. Fault-tolerant quantum computing likely 8-15 years away.",
            "date": "2024-01-05",
            "source_credibility": 0.93,
        },
    ],
    "crispr": [
        {
            "title": "CRISPR-Cas9 Gene Editing Reaches Clinical Milestone",
            "url": "https://nature.com/articles/crispr-clinical-2024",
            "snippet": "First FDA-approved CRISPR therapy for sickle cell disease shows 97% efficacy in trial of 45 patients over 18 months.",
            "date": "2024-03-22",
            "source_credibility": 0.96,
        },
        {
            "title": "Base Editing vs Prime Editing: Comparative Analysis",
            "url": "https://science.org/doi/crispr-comparison-2024",
            "snippet": "Prime editing achieves 4× lower off-target rate than base editing while maintaining comparable on-target efficiency.",
            "date": "2024-02-14",
            "source_credibility": 0.94,
        },
        {
            "title": "CRISPR in Agriculture: Drought-Resistant Crops",
            "url": "https://cell.com/articles/crispr-agriculture-2024",
            "snippet": "Gene-edited wheat varieties show 30% improved yield under water stress conditions without transgenic modification.",
            "date": "2024-01-30",
            "source_credibility": 0.89,
        },
        {
            "title": "Ethical Frameworks for Germline Gene Editing",
            "url": "https://who.int/publications/crispr-ethics-2024",
            "snippet": "WHO expert panel publishes updated guidelines restricting germline editing to therapeutic applications with strict oversight.",
            "date": "2024-04-02",
            "source_credibility": 0.97,
        },
        {
            "title": "CRISPR Delivery Systems: Lipid Nanoparticles Advance",
            "url": "https://nejm.org/doi/crispr-delivery-2024",
            "snippet": "Improved LNP formulations enable liver-targeted delivery with 85% editing efficiency at clinically relevant doses.",
            "date": "2024-03-08",
            "source_credibility": 0.95,
        },
    ],
    "machine learning": [
        {
            "title": "Large Language Models in Scientific Discovery",
            "url": "https://science.org/doi/llm-science-2024",
            "snippet": "AI systems now generate novel hypotheses validated experimentally in materials science and drug discovery at scale.",
            "date": "2024-04-15",
            "source_credibility": 0.93,
        },
        {
            "title": "Graph Neural Networks for Climate Modeling",
            "url": "https://nature.com/articles/gnn-climate-2024",
            "snippet": "GraphCast achieves 10-day weather predictions 1.5× more accurate than traditional numerical models at 1/100 the compute cost.",
            "date": "2024-02-28",
            "source_credibility": 0.92,
        },
        {
            "title": "Federated Learning Preserves Privacy in Medical AI",
            "url": "https://nejm.org/doi/federated-medical-ai",
            "snippet": "Multi-hospital FL system trained on 2.3M patient records achieves diagnostic accuracy comparable to centralised training.",
            "date": "2024-03-18",
            "source_credibility": 0.91,
        },
        {
            "title": "Scaling Laws and Emergent Capabilities in Foundation Models",
            "url": "https://arxiv.org/abs/2401.15678",
            "snippet": "New theoretical framework predicts capability thresholds based on parameter count and training data distribution.",
            "date": "2024-01-22",
            "source_credibility": 0.86,
        },
    ],
    "battery": [
        {
            "title": "Solid-State Batteries Reach Commercial Viability",
            "url": "https://nature.com/articles/solid-state-battery-2024",
            "snippet": "Toyota solid-state batteries achieve 1,200 Wh/L energy density with 10-minute fast charging capability and 10-year lifetime.",
            "date": "2024-03-05",
            "source_credibility": 0.93,
        },
        {
            "title": "Sodium-Ion Batteries: Low-Cost Grid Storage",
            "url": "https://science.org/doi/sodium-ion-2024",
            "snippet": "CATL's sodium-ion cells reach 160 Wh/kg at $40/kWh, enabling economical large-scale energy storage without lithium.",
            "date": "2024-02-12",
            "source_credibility": 0.91,
        },
        {
            "title": "Lithium-Sulfur Battery Cycle Life Breakthrough",
            "url": "https://arxiv.org/abs/2402.08934",
            "snippet": "Novel polysulfide barrier coating extends Li-S battery cycle life from 200 to 1,500 cycles, closing gap with Li-ion.",
            "date": "2024-02-25",
            "source_credibility": 0.85,
        },
    ],
    "blockchain": [
        {
            "title": "Blockchain Supply Chain Transparency: Real-World Results",
            "url": "https://hbr.org/blockchain-supply-chain-2024",
            "snippet": "Walmart's blockchain food traceability system reduces contamination response time from 7 days to 2.2 seconds.",
            "date": "2024-04-08",
            "source_credibility": 0.84,
        },
        {
            "title": "Zero-Knowledge Proofs Enable Privacy-Preserving Verification",
            "url": "https://ieee.org/xplore/zkp-supply-chain",
            "snippet": "ZK-SNARK based systems allow suppliers to prove compliance without revealing sensitive business information.",
            "date": "2024-01-19",
            "source_credibility": 0.88,
        },
    ],
    "renewable": [
        {
            "title": "Long-Duration Energy Storage: Iron-Air Batteries Scale Up",
            "url": "https://nature.com/articles/iron-air-storage-2024",
            "snippet": "Form Energy's iron-air batteries achieve 100-hour storage at $20/kWh, enabling fully renewable grid operation.",
            "date": "2024-03-28",
            "source_credibility": 0.92,
        },
        {
            "title": "Green Hydrogen Production Reaches Cost Parity",
            "url": "https://science.org/doi/green-hydrogen-2024",
            "snippet": "Proton-exchange membrane electrolyzers achieve $1.50/kg H₂ production cost, competitive with grey hydrogen.",
            "date": "2024-02-07",
            "source_credibility": 0.90,
        },
        {
            "title": "Grid-Scale Gravitational Storage: Commercial Deployment",
            "url": "https://ieee.org/xplore/gravity-storage",
            "snippet": "ARES Nevada system demonstrates 400 MWh gravitational storage with 80% round-trip efficiency at $50/kWh.",
            "date": "2024-01-14",
            "source_credibility": 0.87,
        },
    ],
}

_DEFAULT_RESULTS = [
    {
        "title": "Recent Advances and Applications: A Comprehensive Review",
        "url": "https://nature.com/articles/comprehensive-review-2024",
        "snippet": "Systematic review of 200+ studies reveals consistent performance improvements and expanding application domains.",
        "date": "2024-03-20",
        "source_credibility": 0.90,
    },
    {
        "title": "Industry Adoption Accelerates: Market Analysis 2024",
        "url": "https://mckinsey.com/research/technology-adoption-2024",
        "snippet": "Enterprise adoption rate reaches 34% with projected 67% penetration by 2027 across Fortune 500 companies.",
        "date": "2024-02-18",
        "source_credibility": 0.78,
    },
    {
        "title": "Technical Challenges and Solutions: Expert Roundtable",
        "url": "https://ieee.org/xplore/technical-challenges-2024",
        "snippet": "Panel of 45 domain experts identifies top 10 remaining technical barriers and evaluates solution pathways.",
        "date": "2024-01-09",
        "source_credibility": 0.88,
    },
    {
        "title": "Regulatory Landscape: Compliance Guide 2024",
        "url": "https://gov.uk/research/regulatory-framework",
        "snippet": "Updated frameworks across EU, US, and APAC jurisdictions provide clearer compliance pathways for practitioners.",
        "date": "2024-04-01",
        "source_credibility": 0.92,
    },
    {
        "title": "Future Outlook: Predictions and Timelines",
        "url": "https://gartner.com/research/technology-forecast-2024",
        "snippet": "Gartner Hype Cycle positions technology at 'Slope of Enlightenment' with mainstream adoption expected 2026-2028.",
        "date": "2024-03-31",
        "source_credibility": 0.76,
    },
]

_TRENDING: dict[str, list[dict[str, Any]]] = {
    "science": [
        {
            "topic": "Quantum error correction",
            "momentum_score": 0.92,
            "related_queries": ["surface codes", "logical qubits", "fault tolerance"],
        },
        {
            "topic": "mRNA therapeutics beyond COVID",
            "momentum_score": 0.88,
            "related_queries": ["cancer vaccines", "rare diseases", "protein replacement"],
        },
        {
            "topic": "Neuromorphic computing",
            "momentum_score": 0.81,
            "related_queries": ["Intel Loihi", "spike neural networks", "brain-inspired chips"],
        },
    ],
    "technology": [
        {
            "topic": "Agentic AI frameworks",
            "momentum_score": 0.95,
            "related_queries": ["multi-agent systems", "autonomous agents", "tool use"],
        },
        {
            "topic": "Photonic computing",
            "momentum_score": 0.84,
            "related_queries": [
                "optical neural networks",
                "silicon photonics",
                "light-speed computation",
            ],
        },
        {
            "topic": "Post-quantum cryptography",
            "momentum_score": 0.87,
            "related_queries": [
                "NIST standards",
                "lattice cryptography",
                "quantum-resistant algorithms",
            ],
        },
    ],
    "medicine": [
        {
            "topic": "GLP-1 receptor agonists",
            "momentum_score": 0.94,
            "related_queries": ["Ozempic", "weight loss", "cardiometabolic benefits"],
        },
        {
            "topic": "Cell therapy manufacturing",
            "momentum_score": 0.86,
            "related_queries": [
                "CAR-T scale-up",
                "allogeneic cells",
                "Good Manufacturing Practice",
            ],
        },
        {
            "topic": "Digital biomarkers",
            "momentum_score": 0.80,
            "related_queries": ["wearable sensors", "passive monitoring", "clinical validation"],
        },
    ],
    "default": [
        {
            "topic": "Large language models",
            "momentum_score": 0.97,
            "related_queries": ["GPT-4", "Claude", "Gemini"],
        },
        {
            "topic": "Climate tech investment",
            "momentum_score": 0.85,
            "related_queries": ["carbon capture", "clean energy", "ESG"],
        },
        {
            "topic": "Biosecurity and synthetic biology",
            "momentum_score": 0.79,
            "related_queries": ["dual-use research", "biosafety", "pandemic preparedness"],
        },
    ],
}


def _get_results_for_query(query: str, max_results: int) -> list[dict[str, Any]]:
    """Return deterministic results based on keyword matching."""
    q = query.lower()
    best_key = None
    best_score = 0
    for key in _TOPIC_RESULTS:
        score = sum(1 for word in key.split() if word in q)
        if score > best_score:
            best_score = score
            best_key = key

    base = list(_TOPIC_RESULTS[best_key]) if best_key else list(_DEFAULT_RESULTS)

    # Deterministic seeded shuffle based on query hash
    seed = int(hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()[:8], 16)
    for i in range(len(base) - 1, 0, -1):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        j = seed % (i + 1)
        base[i], base[j] = base[j], base[i]

    return base[:max_results]


def _generate_page_content(url: str) -> str:
    """Generate realistic page content from URL keywords."""
    domain = url.split("/")[2] if "//" in url else "unknown.com"
    path_words = url.replace("-", " ").replace("_", " ").split("/")[-1].split()[:4]
    topic_hint = " ".join(path_words) if path_words else "research topic"

    return f"""
    {topic_hint.title()} — Research Overview

    Published by {domain} | Peer-reviewed | Open Access

    Abstract

    This article presents a comprehensive analysis of {topic_hint}, drawing on data from
    237 independent studies conducted between 2020 and 2024. Our findings confirm significant
    performance improvements averaging 34% over baseline approaches, with particular strength
    in practical deployment scenarios.

    Introduction

    The field of {topic_hint} has undergone rapid transformation over the past four years.
    Advances in computational infrastructure, data availability, and methodological refinement
    have collectively enabled capabilities that were considered aspirational as recently as 2021.
    This review synthesises the current state of knowledge and identifies key directions for
    future investigation.

    Methods

    We conducted a systematic literature search across PubMed, Scopus, and Web of Science
    using standardised search terms. Inclusion criteria required peer review, empirical evaluation,
    and quantitative results. Of 1,847 papers identified, 237 met inclusion criteria after
    full-text screening by two independent reviewers.

    Results

    Primary outcomes demonstrate consistent improvement across all evaluated metrics:
    - Accuracy improvement: +34% (95% CI: 28–40%)
    - Computational efficiency: +67% reduction in processing time
    - Cost reduction: 23% average across deployment contexts
    - Reliability: 99.2% uptime in production systems over 12-month follow-up

    Subgroup analysis reveals strongest effects in high-stakes applications (healthcare, finance)
    where precision requirements drive adoption of more sophisticated approaches.

    Discussion

    These findings support a positive trajectory for {topic_hint} with meaningful real-world
    impact. Several limitations warrant acknowledgement: publication bias may inflate positive
    results; industry-funded studies show modestly higher effect sizes; long-term stability
    data beyond 24 months remains limited.

    The regulatory landscape continues to evolve. Recent guidance from major jurisdictions
    provides clearer pathways while maintaining appropriate safeguards. Practitioners are
    advised to monitor ongoing regulatory developments.

    Conclusion

    Evidence strongly supports continued investment in {topic_hint} research and application.
    The field has matured sufficiently for production deployment in many contexts, though
    domain-specific validation remains essential before safety-critical applications.

    References: [1] Smith et al. 2024, Nature; [2] Chen et al. 2024, Science;
    [3] Williams et al. 2023, IEEE; [4] Johnson 2024, NEJM; [5] Patel et al. 2024, arXiv
    """


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


@_registry.tool(
    name="web_search",
    description="Search the web for information on a topic and return a ranked list of results.",
    parameters=[
        ToolParameter(
            name="query", type="string", description="Search query string", required=True
        ),
        ToolParameter(
            name="max_results",
            type="number",
            description="Maximum number of results to return",
            required=False,
            default=5,
        ),
    ],
)
async def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Return deterministic mock search results ranked by credibility."""
    return _get_results_for_query(query, max_results)


@_registry.tool(
    name="fetch_page",
    description="Fetch and return the full text content of a web page. Content is safety-checked.",
    parameters=[
        ToolParameter(
            name="url", type="string", description="URL of the page to fetch", required=True
        ),
    ],
)
async def fetch_page(url: str) -> dict[str, Any]:
    """Fetch page content and run it through the safety pipeline before returning."""
    content = _generate_page_content(url)
    word_count = len(content.split())
    fetched_at = datetime.now(UTC).isoformat()

    # Safety check — web content is the primary injection vector
    raw_result = ToolResult(tool_call_id="fetch-safety-check", output=content)
    safe_result, _ = await _safety.check_tool_result(raw_result)
    safe_content = str(safe_result.output) if safe_result.output is not None else ""

    # Derive title from URL path
    path = url.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ")
    title = path.title() if path else url

    return {
        "url": url,
        "title": title,
        "content": safe_content.strip(),
        "word_count": word_count,
        "fetched_at": fetched_at,
    }


@_registry.tool(
    name="trending_topics",
    description="Return trending research topics and their momentum scores for a domain.",
    parameters=[
        ToolParameter(
            name="domain",
            type="string",
            description="Domain to query: 'science', 'technology', or 'medicine'",
            required=True,
        ),
    ],
)
async def trending_topics(domain: str) -> list[dict[str, Any]]:
    """Return trending topics for the specified domain."""
    return _trending_for_domain(domain)


def _trending_for_domain(domain: str) -> list[dict[str, Any]]:
    key = domain.lower()
    if key in _TRENDING:
        return _TRENDING[key]
    return _TRENDING["default"]
