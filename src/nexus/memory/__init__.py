"""Memory layer: working, episodic, semantic, and procedural memory with security."""

from nexus.memory.consolidation import ConsolidationPipeline, ConsolidationResult
from nexus.memory.embeddings import EmbeddingService, cosine_similarity
from nexus.memory.episodic import EpisodicMemory
from nexus.memory.procedural import ProceduralMemory
from nexus.memory.procedure_extractor import ProcedureExtractor
from nexus.memory.procedure_matcher import ProcedureMatcher
from nexus.memory.retriever import EpisodicRetriever
from nexus.memory.semantic import SemanticMemory
from nexus.memory.semantic_retriever import ScoredFact, SemanticRetriever
from nexus.memory.summarizer import SummarizationStrategy, Summarizer
from nexus.memory.token_counter import TokenCounter
from nexus.memory.types import (
    EpisodicRecord,
    Procedure,
    ProcedureStep,
    RetrievedRecord,
    SemanticFact,
)
from nexus.memory.working import WorkingMemory

__all__ = [
    "ConsolidationPipeline",
    "ConsolidationResult",
    "EmbeddingService",
    "EpisodicMemory",
    "EpisodicRecord",
    "EpisodicRetriever",
    "Procedure",
    "ProcedureExtractor",
    "ProcedureMatcher",
    "ProcedureStep",
    "ProceduralMemory",
    "RetrievedRecord",
    "ScoredFact",
    "SemanticFact",
    "SemanticMemory",
    "SemanticRetriever",
    "Summarizer",
    "SummarizationStrategy",
    "TokenCounter",
    "WorkingMemory",
    "cosine_similarity",
]
