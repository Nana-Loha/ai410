# Week 7 — Retrieval Evaluation Report

**Course:** AI410 Spring 2026 | Bellevue College  
**Student:** Pitchanan Lohavanichbutr  
**Date:** 2026-05-31  
**Framework:** LlamaIndex + sentence-transformers/all-MiniLM-L6-v2 (local)  
**Corpus:** 5 clinical guideline files (ACC/AHA, ADA, GINA, FDA, ISMP)  
**Eval Set:** 50 queries (45 positive + 5 negative)  

---

## Lab 7.1 — Ingestion Summary

| Parameter | Value |
|-----------|-------|
| Documents | 5 guideline files |
| Chunks | 72 nodes |
| chunk_size | 256 tokens (baseline) |
| chunk_overlap | 40 tokens |
| Embedding | sentence-transformers/all-MiniLM-L6-v2 (local) |
| Storage | storage/index/ (persisted) |

Corpus covers: cardiology (chest pain, hypertension), pharmacology (warfarin + aspirin), endocrinology (diabetes A1C), pulmonology (asthma).

---

## Lab 7.2 — Baseline Evaluation

### Baseline Config

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| chunk_size | 180 | ~2-3 sentences per chunk |
| chunk_overlap | 36 | chunk_size × 20% |
| similarity_top_k | 2 | minimum retrieval candidates |
| metadata_filter | False | disabled for baseline |
| embedding | huggingface_local | all-MiniLM-L6-v2 |

### Baseline Metrics

| Metric | Value |
|--------|-------|
| Hit Rate @ 2 | **0.880** |
| MRR @ 2 | **0.830** |
| Positive Hit Rate | 0.956 (43/45) |
| Negative Hit Rate | 0.200 (1/5) |
| Failures | 6 |

**Negative test logic:**  
Positive query: hit if `expected_source` found in top-k retrieved chunks.  
Negative query (NONE): hit if `confidence < 0.35` — RAG always returns chunks, so low confidence score indicates the system is correctly uncertain about out-of-corpus topics.

### Observed Retrieval Failures (2-3)

**Failure 1 — Cross-topic semantic overlap**  
Query: *"Why should ACE inhibitor and ARB not be combined in hypertension treatment?"*  
Expected: `hypertension_followup.txt` | Retrieved: `diabetes_a1c_management.txt`  
Root cause: ACEi/ARB terminology appears in both hypertension and diabetes files (renal protection context). With chunk_size=180, the ONTARGET trial sentence was split from its surrounding context, reducing its retrieval score. The diabetes file's ACEi-related chunks scored higher due to random vector proximity.  
Fix: Increase chunk_size to preserve full clinical decision sections.

**Failure 2 — Chunk boundary fragmentation**  
Query: *"What is the therapeutic INR range for most warfarin indications?"*  
Expected: `warfarin_aspirin.txt` | Retrieved: `chest_pain.txt`, `hypertension_followup.txt`  
Root cause: INR monitoring section sits at a paragraph boundary in `warfarin_aspirin.txt`. With chunk_size=180, the splitter created a boundary immediately before the INR range sentence, separating it from drug-specific context. Other files containing numeric ranges (BP, troponin) scored higher.  
Fix: Increase chunk_size and overlap to keep drug-specific sections intact.

**Failure 3 — False positive on out-of-corpus query**  
Query: *"What chemotherapy regimen is used for stage III colon cancer?"*  
Expected: NONE | Confidence: 0.41 (above threshold 0.35)  
Root cause: No confidence threshold caused the system to return unrelated chunks from chest pain guidelines with confidence >= 0.35, producing a false positive. The system should have returned "not covered in corpus" instead.  
Fix: Confidence threshold = 0.35 correctly handles most negative cases, but threshold may need tuning for edge cases.

---

## Lab 7.3 — Retrieval Tuning (Ablation Study)

### Tuning Strategy

Tested one variable at a time to isolate the effect of each parameter:

1. **Step 1** — Find best `chunk_size`: tested 256 and 320 (baseline=180)
2. **Step 2** — Use best chunk → find best `top_k`: tested 4 (baseline=2)
3. **Step 3** — Use best chunk + k → enable `metadata_filter`

**Overlap rationale:** `overlap = chunk_size × 20%` — industry standard range is 10-25% (LlamaIndex docs, LangChain best practices). Preserves context at boundaries without excessive duplication.

**Chunk size rationale:** Clinical guidelines are written in paragraphs. 256 tokens ≈ 1 paragraph, 320 tokens ≈ 1-2 paragraphs. Larger chunks preserve multi-sentence clinical decision logic in one retrieval unit (Anthropic RAG research, BEIR Benchmark — Thakur et al. 2021).

### Results

| Round | chunk_size | overlap | top_k | metadata_filter | Hit Rate | MRR | Failures |
|-------|-----------|---------|-------|----------------|----------|-----|----------|
| Baseline | 180 | 36 | 2 | ❌ | 0.880 | 0.830 | 6 |
| Round 1 | 256 | 51 | 2 | ❌ | 0.900 | 0.850 | 5 |
| Round 2 | 320 | 64 | 2 | ❌ | 0.920 | 0.840 | 4 |
| Round 3 ⭐ | 320 | 64 | 4 | ❌ | **0.940** | **0.845** | **3** |
| Round 4 | 320 | 64 | 4 | ✅ | 0.900 | 0.820 | 5 |

**Best config: Round 3** (chunk=320, overlap=64, top_k=4, metadata_filter=False)

### Tuning Observations

**Step 1 — Chunking (chunk_size 180 → 320):**  
Increasing chunk_size improved Hit Rate from 0.880 to 0.920 (+0.040). Larger chunks preserved multi-sentence clinical decision logic — e.g. INR ranges and ONTARGET trial findings stayed within one chunk with surrounding context. chunk=320 outperformed chunk=256, confirming clinical guidelines benefit from paragraph-level chunking.

**Step 2 — Retrieval parameter (top_k 2 → 4):**  
Increasing top_k from 2 to 4 improved Hit Rate from 0.920 to 0.940 (+0.020). More candidates increased recall without introducing significant noise, as semantic embeddings already ranked relevant chunks near the top.

**Step 3 — Metadata filter:**  
Enabling metadata_filter reduced Hit Rate from 0.940 to 0.900 (-0.040). Clinical guidelines have significant cross-topic overlap — aspirin appears in both `chest_pain.txt` (cardiology) and `warfarin_aspirin.txt` (pharmacology). The topic classifier routed some queries to the wrong domain, filtering out the correct source. Metadata filtering is appropriate only when topic boundaries are well-defined and non-overlapping.

### Key Insight

> Metadata filtering reduced retrieval quality in this corpus because clinical guidelines are inherently cross-topic. A cardiology query about aspirin in ACS should retrieve pharmacology content about aspirin interactions. Rigid topic filtering breaks this cross-domain reasoning. Future improvement: replace hard metadata filtering with soft re-ranking that boosts topic-relevant chunks without excluding others.

---

## Summary

| | Baseline | Best (Round 3) | Delta |
|--|---------|---------------|-------|
| Hit Rate | 0.880 | **0.940** | +0.060 |
| MRR | 0.830 | **0.845** | +0.015 |
| Failures | 6 | **3** | -3 |

Retrieval quality improved meaningfully through chunking and top_k tuning. The best configuration uses chunk_size=320, overlap=64, top_k=4 without metadata filtering.

Negative test Hit Rate (0.200 baseline) indicates the system still returns unrelated chunks with high confidence for 4/5 out-of-corpus queries. A lower confidence threshold or corpus expansion would improve this in production.

---

*Disclaimer: This system is for clinical decision support only. Not a substitute for professional medical advice.*