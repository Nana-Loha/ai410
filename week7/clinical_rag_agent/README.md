# Week 7 Clinical RAG Agent

This folder contains Week 7 lab artifacts for retrieval-augmented generation (RAG) and final project scope preparation.

## Lab Coverage
- Lab 7.1: Reproducible LlamaIndex ingestion over local guideline corpus
- Lab 7.2: Retrieval evaluation with Hit Rate and MRR on fixed test set
- Lab 7.3: Retrieval tuning via chunking and retrieval parameters
- Lab 7.4: Finalized architecture/spec/backlog artifacts for Week 8

## Files
- `rag_pipeline.py`: ingestion + retrieval pipeline
- `rag_config.json`: baseline and tuned retrieval parameters
- `eval.py`: fixed-set retrieval evaluation and report generation
- `data/guidelines/*.txt`: course-relevant corpus
- `data/eval_queries.jsonl`: fixed retrieval evaluation set
- `evaluation_report.md`: generated metrics + failures
- `architecture_diagram.md`: final project architecture diagram
- `SPEC.md`: updated scope/finalization draft
- `tasks.md`: updated implementation backlog

## Setup
From repo root:

```powershell
uv sync
```

Optional: use Hugging Face API embeddings (instead of mock embeddings):

```powershell
uv add llama-index-embeddings-huggingface-api
$env:HF_TOKEN = "hf_xxx"
$env:USE_HF_EMBEDDINGS = "true"
$env:HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

## Run Lab 7.1 (Ingestion)

```powershell
uv run python week7/clinical_rag_agent/rag_pipeline.py
```

Note: if running as script, ingestion is called from evaluation flow. You can also call ingestion in Python directly.

## Run Lab 7.2 and 7.3 (Evaluation + Tuning)

```powershell
uv run python week7/clinical_rag_agent/eval.py
```

To run eval with Hugging Face embeddings, keep `USE_HF_EMBEDDINGS=true` in your environment.

This writes:
- `week7/clinical_rag_agent/evaluation_report.md`
- `week7/clinical_rag_agent/storage/index/ingestion_manifest.json`

## Safety Constraints
- Do not store patient queries or records in local files.
- Include disclaimer in every final answer:
  - Not a substitute for professional medical advice
- Escalate high-risk or low-confidence outputs for human review.
