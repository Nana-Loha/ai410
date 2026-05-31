"""
generate_answers.py — RAG Answer Generation + Evaluation
=========================================================
Course: AI410 Spring 2026 | Bellevue College
Student: Pitchanan Lohavanichbutr

Loads best config from Lab 7.3 (chunk=320, overlap=64, top_k=4)
Runs all 50 eval queries through RAG pipeline
Shows per-query: hit/miss, rank, retrieved source, generated answer
Shows summary: Hit Rate, MRR

Run:
    uv run python generate_answers.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from rag_pipeline import (
    ClinicalRAGPipeline,
    PipelineConfig,
    build_response_with_citations,
)

BASE_DIR             = Path(__file__).resolve().parent
EVAL_PATH            = BASE_DIR / "data" / "eval_queries.jsonl"
CONFIDENCE_THRESHOLD = 0.35

BEST_CONFIG = PipelineConfig(
    chunk_size=320,
    chunk_overlap=64,
    similarity_top_k=4,
    enable_metadata_filter=False,
    enable_lexical_rerank=False,
    embedding_backend="huggingface_local",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
)


def load_eval_set() -> list[dict]:
    return [
        json.loads(l)
        for l in EVAL_PATH.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def reciprocal_rank(citations: list[str], expected: str) -> float:
    if expected == "NONE":
        return 0.0
    for i, src in enumerate(citations, 1):
        if src == expected:
            return 1.0 / i
    return 0.0


def run() -> None:
    eval_set = load_eval_set()
    pipeline = ClinicalRAGPipeline(base_dir=BASE_DIR, config=BEST_CONFIG)
    pipeline.ingest()

    # Pre-load index once before loop
    pipeline._load_or_build_index()

    total = hit_count = 0
    rr_sum = 0.0
    SEP = "─" * 70

    print(f"\n{'='*70}")
    print("RAG ANSWER GENERATION — Best Config (chunk=320, top_k=4)")
    print(f"{'='*70}\n")

    for row in eval_set:
        total    += 1
        qid      = row["id"]
        query    = row["query"]
        expected = row["expected_source"]
        is_neg   = expected == "NONE"

        result    = pipeline.retrieve(query)
        citations = [c.source_id for c in result.chunks]

        # Hit/Miss
        if is_neg:
            is_hit = result.confidence < CONFIDENCE_THRESHOLD
            rank   = None
            rr     = 0.0
        else:
            is_hit = expected in citations
            rank   = next((i + 1 for i, s in enumerate(citations) if s == expected), None)
            rr     = 1.0 / rank if rank else 0.0

        if is_hit:
            hit_count += 1
        rr_sum += rr

        # Generate answer from retrieved chunks
        if result.chunks:
            answer, _, _ = build_response_with_citations(result.chunks)
            answer_short = answer.split("\n\n")[0][:120].strip()
        else:
            answer_short = "This topic is not covered in the provided clinical guidelines."

        # Print per-query result
        status   = "✅ HIT " if is_hit else "❌ MISS"
        rank_str = f"rank={rank}, RR={rr:.3f}" if rank else ("confidence below threshold" if is_neg and is_hit else "not in top-k")

        print(f"{status} [{qid}] {row.get('topic','')}")
        print(f"   Query    : {query[:75]}")
        print(f"   Expected : {expected}")
        print(f"   Retrieved: {citations} ({rank_str})")
        print(f"   Conf     : {result.confidence:.3f}")
        print(f"   Answer   : {answer_short}...")
        print(SEP)

    # Summary
    hit_rate = hit_count / max(total, 1)
    mrr      = rr_sum / max(total, 1)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total Queries : {total} (45 positive + 5 negative)")
    print(f"Hits          : {hit_count}")
    print(f"Hit Rate      : {hit_rate:.3f}")
    print(f"MRR           : {mrr:.3f}")
    print(f"{'='*70}")
    print("\n✅ Done.")


if __name__ == "__main__":
    run()