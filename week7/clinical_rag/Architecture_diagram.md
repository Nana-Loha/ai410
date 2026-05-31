# Architecture Diagram — Clinical Multi-Agent AI Assistant
**Course:** AI410 Spring 2026 | Bellevue College  
**Developer:** Pitchanan Lohavanichbutr (Solo)  
**Date:** 2026-05-31  

---

## Framework Choice

**LangGraph + LlamaIndex Hybrid**

| Layer | Framework | Role |
|-------|-----------|------|
| Orchestration | LangGraph | Stateful workflow, conditional routing, HITL |
| Retrieval | LlamaIndex | RAG pipeline, VectorStoreIndex, AgentWorkflow |
| Embedding | all-MiniLM-L6-v2 (local) | Semantic search, no API cost |
| LLM | Claude Sonnet / Claude Opus | Clinical reasoning |

---

## System Architecture

```
[User Input]
     │
     ▼
┌─────────────────────────────────────────────────────┐
│              LangGraph Pipeline                      │
│                                                      │
│  ┌──────────────────────────────────────────┐       │
│  │  Node 1 — Planner                        │       │
│  │  Classify intent: symptom/soap/drug_check│       │
│  └──────────────────┬───────────────────────┘       │
│                     │                               │
│                     ▼                               │
│  ┌──────────────────────────────────────────┐       │
│  │  Node 2 — Tool                           │       │
│  │  Claude Sonnet / Claude Opus            │       │
│  └──────────────────┬───────────────────────┘       │
│                     │                               │
│                     ▼                               │
│  ┌──────────────────────────────────────────┐       │
│  │  Node 3 — RAG (LlamaIndex)               │◄──────┼── Corpus
│  │  AgentWorkflow + VectorStoreIndex        │       │   (ACC/AHA, ADA,
│  │  chunk=320, top_k=4, HR=0.940            │       │    GINA, FDA, ISMP)
│  └──────────────────┬───────────────────────┘       │
│                     │                               │
│                     ▼                               │
│  ┌──────────────────────────────────────────┐       │
│  │  Node 4 — Evaluator                      │       │
│  │  Validate quality, retry loop (max 3)    │───────┼── retry → Node 2
│  └──────────────────┬───────────────────────┘       │
│                     │                               │
│                     ▼                               │
│  ┌──────────────────────────────────────────┐       │
│  │  Node 5 — HITL                           │       │
│  │  Human checkpoint (high-risk only)       │       │
│  └──────────────────┬───────────────────────┘       │
│                     │                               │
└─────────────────────┼───────────────────────────────┘
                      │
                      ▼
             [Response + Citations]
```

---

## Node Descriptions

| Node | Framework | Responsibility |
|------|-----------|----------------|
| Planner | LangGraph | Classifies user intent into task type |
| Tool | LangGraph + Claude | Calls clinical tools (symptom check, SOAP, drug check) |
| RAG | LlamaIndex AgentWorkflow | Retrieves evidence from guideline corpus |
| Evaluator | LangGraph | Validates output quality, triggers retry (max 3) |
| HITL | LangGraph | Human checkpoint for high-risk clinical outputs |

---

## RAG Pipeline Detail (Node 3)

```
[Clinical Query]
     │
     ▼
[Query Embedding]          ← all-MiniLM-L6-v2 (local)
     │
     ▼
[VectorStoreIndex]         ← LlamaIndex, persisted storage/index/
     │
     ▼
[Top-K Retrieval]          ← similarity_top_k=4
     │
     ▼
[Confidence Check]         ← threshold=0.35
     │
     ├── conf < 0.35 → "Not covered in corpus"
     │
     └── conf ≥ 0.35 → [Response with Citations]
```

### Retrieval Configuration (Best — Lab 7.3 Round 3)

| Parameter | Value |
|-----------|-------|
| chunk_size | 320 tokens |
| chunk_overlap | 64 tokens (20%) |
| similarity_top_k | 4 |
| metadata_filter | False |
| confidence_threshold | 0.35 |
| Hit Rate | 0.940 |
| MRR | 0.845 |

---

## Corpus

| File | Source |
|------|--------|
| chest_pain.txt | ACC/AHA 2021 Chest Pain Guideline |
| warfarin_aspirin.txt | FDA Drug Label + ISMP + ACC/AHA 2019 |
| hypertension_followup.txt | ACC/AHA 2017 + JNC 8 + ADA 2023 |
| diabetes_a1c_management.txt | ADA Standards of Care 2023 |
| asthma_controller_stepup.txt | GINA 2023 + NAEPP EPR-3 |

---

## Integration Points

```
Week 6 (LangGraph agent)
     +
Week 7 (LlamaIndex RAG)
     ↓
Node 3 connects both:
  - LangGraph passes AgentState to node3_rag.py
  - LlamaIndex AgentWorkflow retrieves guidelines
  - Citations returned to LangGraph state
```

---

*Disclaimer: Not a substitute for professional medical advice.*