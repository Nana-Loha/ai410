"""RAG graph node for the Week 7 clinical agent."""

from __future__ import annotations

import logging
from pathlib import Path

from rag_pipeline import (
    ClinicalRAGPipeline,
    DISCLAIMER,
    build_response_with_citations,
)
from state import ClinicalRAGState

logger = logging.getLogger(__name__)

# Aligned with Lab 7.3 corpus confidence threshold (0.35)
_CONF_OUT_OF_CORPUS = 0.35  # below → not in corpus → HITL
_CONF_LOW_RISK = 0.60       # above → strong retrieval → no escalation

_pipeline: ClinicalRAGPipeline | None = None


def _get_pipeline() -> ClinicalRAGPipeline:
    """Return a module-level singleton to avoid reloading the index on every call."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ClinicalRAGPipeline(base_dir=Path(__file__).resolve().parent)
    return _pipeline


def _classify_risk(confidence: float) -> tuple[str, bool]:
    """Return (risk_level, needs_human) based on retrieval confidence."""
    if confidence < _CONF_OUT_OF_CORPUS:
        return "high", True    # out-of-corpus → escalate to HITL
    if confidence < _CONF_LOW_RISK:
        return "medium", False  # borderline → warn but do not block
    return "low", False


def node3_rag(state: ClinicalRAGState) -> ClinicalRAGState:
    """Retrieve guideline evidence and produce a grounded response."""
    query = state.get("user_query", "").strip()

    if not query:
        logger.warning("node3_rag: empty user_query received")
        return {
            **state,
            "final_response": "Please provide a clinical question.",
            "citations": [],
            "retrieved_chunks": [],
            "retrieval_confidence": 0.0,
            "risk_level": "medium",
            "needs_human": False,
            "last_error": "empty_query",
            "disclaimer": DISCLAIMER,
        }

    try:
        retrieval = _get_pipeline().retrieve(query=query)
        response, citations, confidence = build_response_with_citations(chunks=retrieval.chunks)
        risk_level, needs_human = _classify_risk(confidence)

        logger.info(
            "node3_rag: confidence=%.3f risk=%s needs_human=%s citations=%s query=%r",
            confidence, risk_level, needs_human, citations, query[:80],
        )

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
            "retry_count": 0,
            "disclaimer": DISCLAIMER,
        }

    except Exception as exc:
        logger.error("node3_rag: pipeline error: %s", exc, exc_info=True)
        return {
            **state,
            "last_error": str(exc),
            "retry_count": state.get("retry_count", 0) + 1,
            "final_response": (
                "The RAG pipeline failed. Please retry or consult a clinician.\n\n"
                f"Disclaimer: {DISCLAIMER}."
            ),
            "risk_level": "high",
            "needs_human": True,
        }
