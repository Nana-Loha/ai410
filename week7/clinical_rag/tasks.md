# tasks.md — Final Project Implementation Backlog
**Course:** AI410 Spring 2026 | Bellevue College  
**Developer:** Pitchanan Lohavanichbutr (Solo)  
**Project:** Clinical Multi-Agent AI Assistant  
**Sprint:** Sprint 3 — Final Project Gate  

---

## Status Legend
- 🟢 Done
- 🟡 In Progress
- 🔴 TODO

---

## Completed

| Task | Status |
|------|--------|
| LangGraph 4-node pipeline (Planner, Tool, Evaluator, HITL) | 🟢 Done |
| Symptom check, SOAP, drug interaction tools | 🟢 Done |
| Self-correction retry loop (max 3) | 🟢 Done |
| HITL checkpoint for high-risk outputs | 🟢 Done |
| LlamaIndex ingestion pipeline (5 guideline files) | 🟢 Done |
| Retrieval evaluation (Hit Rate=0.940, MRR=0.845) | 🟢 Done |
| Ablation study — chunking, top_k, metadata filter | 🟢 Done |
| AgentWorkflow pattern (node3_rag.py) | 🟢 Done |
| SPEC.md + architecture diagram | 🟢 Done |

---

## Environment Hardening

| Task | Priority | Status |
|------|----------|--------|
| Create `.env.example` with all required env vars | High | 🔴 TODO |
| Add `setup.sh` for reproducible environment setup | High | 🔴 TODO |
| Add `.gitignore` for storage/, __pycache__, .env | High | 🔴 TODO |
| GitHub Actions CI pipeline | High | 🔴 TODO |
| Validate MCP server connectivity | High | 🔴 TODO |

---

## Prototype Slice Build

| Task | Priority | Status |
|------|----------|--------|
| Integrate Node 3 RAG into LangGraph graph.py | High | 🔴 TODO |
| Connect node3_rag.py to Week 6 pipeline | High | 🔴 TODO |
| Run one complete user flow: input → RAG → response | High | 🔴 TODO |
| Validate citations in final response | High | 🔴 TODO |
| Test all 3 task types (symptom, SOAP, drug_check) | High | 🔴 TODO |

---

## HITL and Logging Validation

| Task | Priority | Status |
|------|----------|--------|
| Add structured JSON logging to all nodes | High | 🔴 TODO |
| Demonstrate one HITL interrupt for high-risk output | High | 🔴 TODO |
| Capture structured log file for submission | High | 🔴 TODO |

---

## Architecture (Finalized)

| Decision | Choice |
|----------|--------|
| Orchestration | LangGraph |
| Retrieval | LlamaIndex AgentWorkflow |
| Embedding | all-MiniLM-L6-v2 (local) |
| LLM (clinical) | Claude Sonnet |
| LLM (drug check) | Claude Opus |
| chunk_size | 320 |
| top_k | 4 |

---

*Last updated: 2026-05-31*