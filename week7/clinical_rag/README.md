# Week 7 — Clinical RAG Pipeline
**Course:** AI410 Spring 2026 | Bellevue College  
**Student:** Pitchanan Lohavanichbutr  
**Sprint:** Sprint 3 — Healthcare AI Agent  
**Repository:** https://github.com/Nana-Loha/ai410  

---

## Submission Contents

| Required | File | Lab |
|----------|------|-----|
| LlamaIndex RAG implementation | `rag_pipeline.py`, `ingest.py` | 7.1 |
| Setup notes | This README | 7.1 |
| Retrieval evaluation (Hit Rate, MRR, failures) | `eval_baseline.py`, `eval_tuned.py`, `Evaluation_report.md` | 7.2–7.3 |
| Architecture diagram | `Architecture_diagram.md` | 7.4 |
| SPEC.md draft | `spec.md` | 7.4 |
| Team task backlog | `tasks.md` | 7.4 |

---

## Lab 7.1 — LlamaIndex RAG Implementation

### Pipeline Overview

```
data/guidelines/*.txt
        ↓
SimpleDirectoryReader     ← load .txt files
        ↓
Metadata Enrichment       ← topic, guideline_source per file
        ↓
SentenceSplitter          ← chunk_size=320, overlap=64
        ↓
HuggingFaceEmbedding      ← all-MiniLM-L6-v2 (local, no API cost)
        ↓
VectorStoreIndex          ← persisted to storage/index/
        ↓
ingestion_manifest.json   ← reproducibility record
```

### Corpus (data/guidelines/)

| File | Source | Topic |
|------|--------|-------|
| `chest_pain.txt` | ACC/AHA 2021 Chest Pain Guideline | Cardiology |
| `warfarin_aspirin.txt` | FDA Drug Label + ISMP + ACC/AHA 2019 | Pharmacology |
| `hypertension_followup.txt` | ACC/AHA 2017 + JNC 8 + ADA 2023 | Cardiology |
| `diabetes_a1c_management.txt` | ADA Standards of Care 2023 | Endocrinology |
| `asthma_controller_stepup.txt` | GINA 2023 + NAEPP EPR-3 | Pulmonology |

### Setup

**Prerequisites:** Python 3.12+, uv

```powershell
# Install dependencies
uv add llama-index llama-index-embeddings-huggingface

# Run Lab 7.1 — build index
uv run python week7/clinical_rag/ingest.py --rebuild
```

Expected output:
```
[Embed] HuggingFace Local: sentence-transformers/all-MiniLM-L6-v2
[Ingest] Building index from 5 files...
[Ingest] 72 chunks (size=320, overlap=64)
[Ingest] Done — manifest: storage/index/ingestion_manifest.json
```

---

## Lab 7.2 — Retrieval Evaluation

### Evaluation Set

`data/eval_queries.jsonl` — 50 queries:
- 45 positive queries (mapped to specific guideline files)
- 5 negative queries (out-of-corpus topics: dengue, migraine, pneumonia, Alzheimer's, oncology)

**Negative test logic:**  
Positive: hit if `expected_source` found in top-k retrieved chunks.  
Negative (NONE): hit if `confidence < 0.35` — RAG always returns chunks, so low confidence indicates the system correctly recognizes out-of-corpus topics.

### Baseline Metrics

| Metric | Value |
|--------|-------|
| Hit Rate @ 2 | 0.880 |
| MRR @ 2 | 0.830 |
| Positive Hit Rate | 0.956 |
| Negative Hit Rate | 0.200 |

```powershell
uv run python week7/clinical_rag/eval_baseline.py
```

### Observed Retrieval Failures

**Failure 1 — Chunk boundary fragmentation (q13)**  
Query: *"What is the therapeutic INR range for most warfarin indications?"*  
Root cause: chunk_size=180 split INR range sentence from surrounding drug-specific context. Numeric ranges in other files scored higher.

**Failure 2 — Cross-topic semantic overlap (q24)**  
Query: *"Why should ACE inhibitor and ARB not be combined in hypertension treatment?"*  
Root cause: ACEi/ARB terminology appears in both hypertension and diabetes files. Smaller chunks lost the ONTARGET trial context needed to distinguish the sources.

**Failure 3 — False positive on out-of-corpus query (q46)**  
Query: *"What is the first-line treatment for acute dengue fever?"*  
Root cause: No confidence threshold — system returned unrelated chunks with confidence ≥ 0.35, producing a clinically dangerous false positive.

---

## Lab 7.3 — Retrieval Tuning (Ablation Study)

Tested one variable at a time to isolate each parameter's effect.

| Round | chunk_size | overlap | top_k | metadata_filter | Hit Rate | MRR |
|-------|-----------|---------|-------|----------------|----------|-----|
| Baseline | 180 | 36 | 2 | ❌ | 0.880 | 0.830 |
| Round 1 | 256 | 51 | 2 | ❌ | 0.900 | 0.850 |
| Round 2 | 320 | 64 | 2 | ❌ | 0.920 | 0.840 |
| **Round 3 ⭐** | **320** | **64** | **4** | ❌ | **0.940** | **0.845** |
| Round 4 | 320 | 64 | 4 | ✅ | 0.900 | 0.820 |

**Best config: Round 3** (chunk=320, overlap=64, top_k=4, metadata_filter=False)

**Key findings:**
- Larger chunks preserve multi-sentence clinical decision logic (+0.040 Hit Rate)
- Higher top_k increases recall (+0.020 Hit Rate)
- Metadata filtering **reduced** performance — clinical guidelines have cross-topic overlap (e.g., aspirin appears in both cardiology and pharmacology guidelines)

**Overlap rationale:** `overlap = chunk_size × 20%` — industry standard (LlamaIndex docs, LangChain best practices, BEIR Benchmark — Thakur et al. 2021)

```powershell
uv run python week7/clinical_rag/eval_tuned.py
```

---

## Lab 7.4 — Final Project Scope

**Framework:** LangGraph + LlamaIndex Hybrid

| Layer | Framework |
|-------|-----------|
| Orchestration | LangGraph |
| Retrieval | LlamaIndex AgentWorkflow |
| Embedding | all-MiniLM-L6-v2 (local) |
| LLM | Claude Sonnet / Claude Opus |

See `spec.md`, `Architecture_diagram.md`, and `tasks.md` for full details.

`node3_rag.py` implements the LlamaIndex AgentWorkflow pattern — ready for integration into the LangGraph pipeline in Week 8.

---

## File Reference

| File | Purpose |
|------|---------|
| `rag_pipeline.py` | Core RAG pipeline (ingestion + retrieval) |
| `ingest.py` | Lab 7.1 — reproducible ingestion |
| `eval_baseline.py` | Lab 7.2 — baseline evaluation |
| `eval_tuned.py` | Lab 7.3 — ablation tuning |
| `generate_answers.py` | QA generation with per-query results |
| `Evaluation_report.md` | Full evaluation report |
| `node3_rag.py` | AgentWorkflow pattern (Lab 7.4) |
| `state.py` | ClinicalRAGState schema |
| `spec.md` | Final project specification |
| `Architecture_diagram.md` | System architecture |
| `tasks.md` | Implementation backlog |
| `data/eval_queries.jsonl` | 50-query evaluation set |
| `data/guidelines/` | Clinical guideline corpus (5 files) |

---

## Safety

- No patient data stored in any file
- Every response includes: *"Not a substitute for professional medical advice"*
- High-risk outputs escalated to HITL (confidence < 0.35 → human review)

---

*Disclaimer: This system is for clinical decision support only. Not a substitute for professional medical advice.*