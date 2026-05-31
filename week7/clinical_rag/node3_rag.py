"""RAG graph node for the Week 7 clinical agent."""

from __future__ import annotations

from pathlib import Path

from rag_pipeline import (
    ClinicalRAGPipeline,
    DISCLAIMER,
    build_response_with_citations,
)
from state import ClinicalRAGState


def node3_rag(state: ClinicalRAGState) -> ClinicalRAGState:
    """Retrieve evidence and generate a grounded response.

    Behavior:
    - Uses local guideline files from data/guidelines/*.txt
    - Produces citations and confidence score
    - Escalates to human review when confidence is low
    """
    try:
        query = state.get("user_query", "").strip()
        if not query:
            return {
                **state,
                "final_response": "Please provide a clinical question.",
                "citations": [],
                "retrieved_chunks": [],
                "retrieval_confidence": 0.0,
                "risk_level": "low",
                "needs_human": False,
                "last_error": "empty_query",
                "disclaimer": DISCLAIMER,
            }

        pipeline = ClinicalRAGPipeline(base_dir=Path(__file__).resolve().parent)
        retrieval = pipeline.retrieve(query=query)
        response, citations, confidence = build_response_with_citations(chunks=retrieval.chunks)

        risk_level = "high" if confidence < 0.2 else "medium" if confidence < 0.5 else "low"
        needs_human = risk_level in {"high", "medium"}

        return {
            **state,
            "retrieved_chunks": [
                {"source_id": c.source_id, "text": c.text, "score": c.score}
                for c in retrieval.chunks
            ],
            "retrieval_confidence": confidence,
            "citations": citations,
            "final_response": response,
            "risk_level": risk_level,
            "needs_human": needs_human,
            "last_error": None,
            "disclaimer": DISCLAIMER,
        }

    except Exception as exc:
        return {
            **state,
            "last_error": str(exc),
            "retry_count": state.get("retry_count", 0) + 1,
            "final_response": (
                "The RAG pipeline failed for this request. "
                "Please retry or consult a clinician.\n\n"
                f"Disclaimer: {DISCLAIMER}."
            ),
            "needs_human": True,
        }
