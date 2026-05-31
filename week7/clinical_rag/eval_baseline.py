"""
Lab 7.2 — Retrieval Evaluation (Baseline)
==========================================
Course: AI410 Spring 2026 | Bellevue College
Student: Pitchanan Lohavanichbutr

Negative test logic:
  - Positive query: hit if expected_source in retrieved citations
  - Negative query (expected=NONE): hit if confidence < CONFIDENCE_THRESHOLD
    Rationale: RAG always returns chunks — low confidence = system correctly
    uncertain about out-of-corpus topics.

Run:
    uv run python eval_baseline.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from rag_pipeline import ClinicalRAGPipeline, PipelineConfig

BASE_DIR             = Path(__file__).resolve().parent
EVAL_PATH            = BASE_DIR / "data" / "eval_queries.jsonl"
CONFIDENCE_THRESHOLD = 0.35


def get_config() -> PipelineConfig:
    use_hf   = os.environ.get("USE_HF_EMBEDDINGS", "").lower() in {"1", "true", "yes"}
    hf_model = os.environ.get("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return PipelineConfig(
        chunk_size=180, chunk_overlap=36, similarity_top_k=2,
        enable_metadata_filter=False, enable_lexical_rerank=False,
        embedding_backend="huggingface_local" if use_hf else "mock",
        embedding_model_name=hf_model,
    )


def load_eval_set() -> list[dict]:
    return [json.loads(l) for l in EVAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def reciprocal_rank(citations: list[str], expected: str) -> float:
    if expected == "NONE":
        return 0.0
    for i, src in enumerate(citations, 1):
        if src == expected:
            return 1.0 / i
    return 0.0


def evaluate(config: PipelineConfig, eval_set: list[dict]) -> dict:
    pipeline = ClinicalRAGPipeline(base_dir=BASE_DIR, config=config)
    pipeline.ingest()

    total = hit_count = 0
    rr_sum = 0.0
    pos_total = neg_total = pos_hit = neg_hit = 0
    failures = []

    for row in eval_set:
        total    += 1
        expected  = row["expected_source"]
        result    = pipeline.retrieve(row["query"])
        citations = [c.source_id for c in result.chunks]

        if expected == "NONE":
            neg_total += 1
            is_hit = result.confidence < CONFIDENCE_THRESHOLD
            if is_hit:
                neg_hit += 1
        else:
            pos_total += 1
            is_hit = expected in citations
            if is_hit:
                pos_hit += 1

        if is_hit:
            hit_count += 1
        else:
            failures.append({
                "id": row["id"], "query": row["query"],
                "expected": expected, "returned": citations,
                "confidence": round(result.confidence, 4),
            })

        rr_sum += reciprocal_rank(citations, expected)

    return {
        "total":        total,
        "hit_rate":     hit_count / max(total, 1),
        "mrr":          rr_sum / max(total, 1),
        "pos_hit_rate": pos_hit / max(pos_total, 1),
        "neg_hit_rate": neg_hit / max(neg_total, 1),
        "failures":     failures,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("LAB 7.2 — Baseline Evaluation")
    print("=" * 60)

    result = evaluate(get_config(), load_eval_set())

    print(f"\nHit Rate        : {result['hit_rate']:.3f}")
    print(f"MRR             : {result['mrr']:.3f}")
    print(f"Positive Hit    : {result['pos_hit_rate']:.3f}")
    print(f"Negative Hit    : {result['neg_hit_rate']:.3f} (threshold={CONFIDENCE_THRESHOLD})")
    print(f"Failures        : {len(result['failures'])}")

    if result["failures"]:
        print("\n── Failures ──")
        for f in result["failures"]:
            print(f"  [{f['id']}] expected={f['expected']} | returned={f['returned']} | confidence={f['confidence']}")

    print("\n✅ Lab 7.2 complete.")