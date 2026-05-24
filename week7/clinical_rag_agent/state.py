"""State schema for the Week 7 clinical RAG graph."""

from typing import Optional, TypedDict


class RAGChunk(TypedDict):
    """A retrieved evidence chunk used for grounded generation."""

    source_id: str
    text: str
    score: float


class ClinicalRAGState(TypedDict):
    """Shared state passed across graph nodes."""

    # Input
    user_query: str

    # Retrieval
    retrieved_chunks: list[RAGChunk]
    retrieval_confidence: float

    # Control and safety
    risk_level: str  # low | medium | high
    needs_human: bool
    retry_count: int
    max_retries: int
    last_error: Optional[str]

    # Output
    citations: list[str]
    final_response: Optional[str]
    disclaimer: str
