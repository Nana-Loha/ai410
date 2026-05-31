# SPEC.md — Clinical Multi-Agent AI Assistant
**Course:** AI410 Spring 2026 | Bellevue College  
**Sprint:** Sprint 3 — Week 7 Gate  
**Developer:** Pitchanan Lohavanichbutr (Solo)  
**Date:** 2026-05-31  

---

## Project Overview

A clinical decision support system that processes free-text patient queries through a sequential multi-node pipeline, retrieving evidence-based guidelines via LlamaIndex RAG before generating structured clinical responses.

**Extends Week 6** healthcare agent (LangGraph + HITL) by adding RAG-powered guideline retrieval as Node 3 in the pipeline.

---

## Framework Choice

| Layer | Framework | Rationale |
|-------|-----------|-----------|
| Orchestration | **LangGraph** | Stateful multi-node workflow, conditional routing, HITL support |
| Retrieval | **LlamaIndex** | Production RAG pipeline, VectorStoreIndex, AgentWorkflow pattern |
| Embedding | sentence-transformers/all-MiniLM-L6-v2 | Local, no API cost, reproducible |
| LLM | Claude Sonnet / Claude Opus | Anthropic API, task-specific model allocation |

**Architecture: LangGraph + LlamaIndex Hybrid**

LangGraph handles orchestration and state management. LlamaIndex handles retrieval and evidence grounding. They connect at Node 3 (RAG node) where LangGraph passes clinical context to LlamaIndex AgentWorkflow.

---

## System Architecture

```
[User Input]
     ↓
[Node 1: Planner]       — classify intent (symptom / soap / drug_check)
     ↓
[Node 2: Tool]          — call clinical tools (Claude Sonnet)
     ↓
[Node 3: RAG]           — LlamaIndex AgentWorkflow retrieves guidelines
     ↓                    corpus: ACC/AHA, ADA, GINA, FDA, ISMP
[Node 4: Evaluator]     — validate output, trigger retry loop (max 3)
     ↓
[Node 5: HITL]          — human checkpoint for high-risk outputs
     ↓
[Node 6: Response]      — final grounded response with citations
     ↓
[User Output]
```

---

## Team Roles (Solo Developer)

| Role | Responsibility |
|------|---------------|
| Architect | System design, framework selection, integration |
| Backend Engineer | LangGraph nodes, LlamaIndex pipeline, RAG |
| ML Engineer | Embedding, chunking strategy, retrieval tuning |
| QA Engineer | Evaluation framework, hit rate/MRR, failure analysis |
| DevOps | Environment setup, reproducibility, CI |

---

## Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-001 | Ingest clinical guidelines corpus (≥5 documents) | High |
| FR-002 | Retrieve relevant guidelines for any clinical query | High |
| FR-003 | Hit Rate ≥ 0.85 on fixed 50-query evaluation set | High |
| FR-004 | MRR ≥ 0.75 on fixed evaluation set | High |
| FR-005 | HITL checkpoint for high-risk clinical outputs | High |
| FR-006 | Structured logging for all pipeline steps | Medium |
| FR-007 | Confidence threshold for out-of-corpus queries | Medium |
| FR-008 | Reproducible ingestion with manifest tracking | Medium |

---

## Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-001 | No patient data stored in local files (HIPAA) |
| NFR-002 | Every response includes medical disclaimer |
| NFR-003 | Reproducible environment via uv + pyproject.toml |
| NFR-004 | All retrieval decisions traceable to source documents |

---

## Corpus

| File | Source | Topic |
|------|--------|-------|
| chest_pain.txt | ACC/AHA 2021 Chest Pain Guideline | Cardiology |
| warfarin_aspirin.txt | FDA Drug Label + ISMP + ACC/AHA 2019 | Pharmacology |
| hypertension_followup.txt | ACC/AHA 2017 + JNC 8 + ADA 2023 | Cardiology |
| diabetes_a1c_management.txt | ADA Standards of Care 2023 | Endocrinology |
| asthma_controller_stepup.txt | GINA 2023 + NAEPP EPR-3 | Pulmonology |

---

## Retrieval Configuration (Best — Lab 7.3 Round 3)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| chunk_size | 320 | ~2 paragraphs — preserves clinical decision context |
| chunk_overlap | 64 | chunk_size × 20% (industry standard) |
| similarity_top_k | 4 | Increases recall without noise |
| metadata_filter | False | Cross-topic overlap in clinical guidelines |
| confidence_threshold | 0.35 | Selected through retrieval evaluation experiments |

### Evaluation Summary

| Run | Hit Rate | MRR |
|-----|----------|-----|
| Baseline | 0.880 | 0.830 |
| Tuned (best) | **0.940** | **0.845** |
| Improvement | +0.060 | +0.015 |

---

## Constraints

- Solo developer
- Python 3.12+ required
- Local embedding only (no API rate limits)
- No patient data in any file
- uv as package manager

---

*Disclaimer: Not a substitute for professional medical advice.*