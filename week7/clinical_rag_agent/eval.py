"""Lab 7 retrieval evaluation and tuning runner.

Computes Hit Rate and MRR for baseline vs tuned retrieval configs.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from rag_pipeline import ClinicalRAGPipeline, PipelineConfig


BASE_DIR = Path(__file__).resolve().parent
EVAL_SET_PATH = BASE_DIR / "data" / "eval_queries.jsonl"
REPORT_PATH = BASE_DIR / "evaluation_report.md"


def load_eval_set(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def reciprocal_rank(citations: list[str], expected: str) -> float:
    if expected == "NONE":
        return 0.0
    for idx, source in enumerate(citations, start=1):
        if source == expected:
            return 1.0 / idx
    return 0.0


def evaluate(config: PipelineConfig, eval_set: list[dict]) -> dict:
    pipeline = ClinicalRAGPipeline(base_dir=BASE_DIR, config=config)
    pipeline.ingest()

    total = 0
    hit_count = 0
    rr_sum = 0.0
    failures: list[dict] = []

    for row in eval_set:
        total += 1
        query = row["query"]
        expected = row["expected_source"]

        result = pipeline.retrieve(query)
        citations = [c.source_id for c in result.chunks]

        is_hit = expected in citations if expected != "NONE" else len(citations) == 0
        if is_hit:
            hit_count += 1
        else:
            failures.append(
                {
                    "id": row["id"],
                    "query": query,
                    "expected": expected,
                    "returned": citations,
                }
            )

        rr_sum += reciprocal_rank(citations, expected)

    return {
        "config": asdict(config),
        "total": total,
        "hit_rate": hit_count / max(total, 1),
        "mrr": rr_sum / max(total, 1),
        "failures": failures,
    }


def write_report(baseline: dict, tuned: dict) -> None:
    baseline_failures = baseline["failures"][:3]
    tuned_failures = tuned["failures"][:3]

    report = [
        "# Week 7 Retrieval Evaluation Report",
        "",
        "## Metrics",
        "",
        "| Run | Hit Rate | MRR |",
        "|---|---:|---:|",
        f"| Baseline | {baseline['hit_rate']:.3f} | {baseline['mrr']:.3f} |",
        f"| Tuned | {tuned['hit_rate']:.3f} | {tuned['mrr']:.3f} |",
        "",
        "## Baseline Config",
        "",
        f"- {baseline['config']}",
        "",
        "## Tuned Config",
        "",
        f"- {tuned['config']}",
        "",
        "## Observed Retrieval Failures (2-3)",
        "",
    ]

    combined = baseline_failures + tuned_failures
    if not combined:
        report.append("- No failures observed on this fixed test set.")
    else:
        for failure in combined[:3]:
            report.append(
                "- "
                + f"{failure['id']}: expected={failure['expected']}, returned={failure['returned']}"
            )
            report.append(f"  - Query: {failure['query']}")

    report.extend(
        [
            "",
            "## Tuning Summary",
            "",
            "- Increased chunk size and overlap to preserve context in guideline snippets.",
            "- Enabled topic metadata filtering to narrow retrieval to relevant clinical domain.",
            "- Enabled lexical reranking and increased similarity_top_k to improve recall and ranking.",
            "- Keep deterministic corpus and fixed eval set for reproducible comparisons.",
        ]
    )

    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    eval_set = load_eval_set(EVAL_SET_PATH)

    use_hf = os.environ.get("USE_HF_EMBEDDINGS", "").lower() in {"1", "true", "yes"}
    hf_model = os.environ.get("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    baseline_cfg = PipelineConfig(
        chunk_size=180,
        chunk_overlap=20,
        similarity_top_k=2,
        enable_metadata_filter=False,
        enable_lexical_rerank=False,
        embedding_backend="huggingface_api" if use_hf else "mock",
        embedding_model_name=hf_model,
    )
    tuned_cfg = PipelineConfig(
        chunk_size=320,
        chunk_overlap=80,
        similarity_top_k=4,
        enable_metadata_filter=True,
        enable_lexical_rerank=True,
        embedding_backend="huggingface_api" if use_hf else "mock",
        embedding_model_name=hf_model,
    )

    baseline = evaluate(baseline_cfg, eval_set)
    tuned = evaluate(tuned_cfg, eval_set)

    write_report(baseline, tuned)

    print("Evaluation complete")
    print(f"Baseline: hit_rate={baseline['hit_rate']:.3f}, mrr={baseline['mrr']:.3f}")
    print(f"Tuned: hit_rate={tuned['hit_rate']:.3f}, mrr={tuned['mrr']:.3f}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
