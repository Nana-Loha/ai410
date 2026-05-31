"""
Lab 7.3 — Retrieval Tuning (Ablation Study)
=============================================
Course: AI410 Spring 2026 | Bellevue College
Student: Pitchanan Lohavanichbutr

Tuning Strategy:
  Step 1 — Find best chunk_size (180 vs 256 vs 320)
  Step 2 — Use best chunk → find best top_k (2 vs 4)
  Step 3 — Use best chunk + best k → enable metadata_filter

Overlap rationale: chunk_size x 20% (industry standard 10-25%)
Chunk rationale: 256 tokens ~ 1 paragraph, 320 ~ 2 paragraphs (clinical text)

Run:
    uv run python eval_tuned.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from rag_pipeline import ClinicalRAGPipeline, PipelineConfig

BASE_DIR             = Path(__file__).resolve().parent
EVAL_PATH            = BASE_DIR / "data" / "eval_queries.jsonl"
CONFIDENCE_THRESHOLD = 0.35


def get_embed_settings() -> tuple[str, str]:
    use_hf   = os.environ.get("USE_HF_EMBEDDINGS", "").lower() in {"1", "true", "yes"}
    hf_model = os.environ.get("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return "huggingface_local" if use_hf else "mock", hf_model


def load_eval_set() -> list[dict]:
    return [json.loads(l) for l in EVAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def reciprocal_rank(citations: list[str], expected: str) -> float:
    if expected == "NONE":
        return 0.0
    for i, src in enumerate(citations, 1):
        if src == expected:
            return 1.0 / i
    return 0.0


def evaluate(config: PipelineConfig, eval_set: list[dict], label: str) -> dict:
    print(f"\n[{label}] chunk={config.chunk_size}, overlap={config.chunk_overlap}, "
          f"top_k={config.similarity_top_k}, filter={config.enable_metadata_filter}")

    pipeline = ClinicalRAGPipeline(base_dir=BASE_DIR, config=config)
    pipeline.ingest()

    total = hit_count = 0
    rr_sum = 0.0
    failures = []

    for row in eval_set:
        total    += 1
        expected  = row["expected_source"]
        result    = pipeline.retrieve(row["query"])
        citations = [c.source_id for c in result.chunks]

        if expected == "NONE":
            is_hit = result.confidence < CONFIDENCE_THRESHOLD
        else:
            is_hit = expected in citations

        if is_hit:
            hit_count += 1
        else:
            failures.append({
                "id": row["id"], "query": row["query"],
                "expected": expected, "returned": citations,
                "confidence": round(result.confidence, 4),
            })

        rr_sum += reciprocal_rank(citations, expected)

    hit_rate = hit_count / max(total, 1)
    mrr      = rr_sum / max(total, 1)
    print(f"  Hit Rate: {hit_rate:.3f} | MRR: {mrr:.3f} | Failures: {len(failures)}")

    return {
        "label":    label,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k":    config.similarity_top_k,
        "metadata_filter": config.enable_metadata_filter,
        "hit_rate": hit_rate,
        "mrr":      mrr,
        "failures": failures,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("LAB 7.3 — Retrieval Tuning (Ablation Study)")
    print("=" * 60)

    eval_set          = load_eval_set()
    backend, hf_model = get_embed_settings()

    # ── Load baseline metrics from Lab 7.2 ──
    baseline_report = BASE_DIR / "lab72_baseline_report.md"
    print("\n[Baseline] Reading from Lab 7.2 results...")
    print("  (Run eval_baseline.py first if not done)")

    # ── Step 1: Test chunk_size ──
    print("\n── Step 1: Chunking ──")
    r1 = evaluate(PipelineConfig(chunk_size=256, chunk_overlap=51, similarity_top_k=2,
                                  enable_metadata_filter=False, enable_lexical_rerank=False,
                                  embedding_backend=backend, embedding_model_name=hf_model),
                  eval_set, "Round 1 (chunk=256)")

    r2 = evaluate(PipelineConfig(chunk_size=320, chunk_overlap=64, similarity_top_k=2,
                                  enable_metadata_filter=False, enable_lexical_rerank=False,
                                  embedding_backend=backend, embedding_model_name=hf_model),
                  eval_set, "Round 2 (chunk=320)")

    best_chunk_result = max([r1, r2], key=lambda x: x["hit_rate"])
    best_chunk   = best_chunk_result["chunk_size"]
    best_overlap = best_chunk_result["chunk_overlap"]
    print(f"\n→ Best chunk_size: {best_chunk} (overlap={best_overlap})")

    # ── Step 2: Test top_k ──
    print("\n── Step 2: top_k ──")
    r3 = evaluate(PipelineConfig(chunk_size=best_chunk, chunk_overlap=best_overlap, similarity_top_k=4,
                                  enable_metadata_filter=False, enable_lexical_rerank=False,
                                  embedding_backend=backend, embedding_model_name=hf_model),
                  eval_set, "Round 3 (top_k=4)")

    best_k = 4 if r3["hit_rate"] >= r2["hit_rate"] else 2
    print(f"\n→ Best top_k: {best_k}")

    # ── Step 3: metadata_filter ──
    print("\n── Step 3: metadata_filter ──")
    r4 = evaluate(PipelineConfig(chunk_size=best_chunk, chunk_overlap=best_overlap, similarity_top_k=best_k,
                                  enable_metadata_filter=True, enable_lexical_rerank=False,
                                  embedding_backend=backend, embedding_model_name=hf_model),
                  eval_set, "Round 4 (metadata_filter=True)")

    # ── Summary ──
    all_rounds = [r1, r2, r3, r4]
    best = max(all_rounds, key=lambda x: x["hit_rate"])

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Round':<30} {'Hit Rate':>10} {'MRR':>8} {'Failures':>10}")
    print("-" * 60)
    for r in all_rounds:
        marker = " ⭐" if r["label"] == best["label"] else ""
        print(f"{r['label']:<30} {r['hit_rate']:>10.3f} {r['mrr']:>8.3f} {len(r['failures']):>10}{marker}")
    print("=" * 60)
    print(f"\nBest: {best['label']} — Hit Rate={best['hit_rate']:.3f} | MRR={best['mrr']:.3f}")

    if best["failures"]:
        print("\n── Remaining Failures ──")
        for f in best["failures"]:
            print(f"  [{f['id']}] expected={f['expected']} | confidence={f['confidence']}")

    print("\n✅ Lab 7.3 complete.")