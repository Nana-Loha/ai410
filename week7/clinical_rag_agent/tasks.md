# Week 7/8 Task Backlog

## Retrieval Pipeline
- [ ] Add metadata tags (topic, urgency) during ingestion for filterable retrieval.
- [ ] Add corpus validation checks for empty/duplicate guideline files.
- [ ] Add deterministic ingestion smoke test.

## Evaluation
- [ ] Expand eval set from 6 to 20+ queries across symptom, meds, and chronic care topics.
- [ ] Add per-topic metric breakdown (Hit Rate and MRR by category).
- [ ] Add regression gate to fail CI if tuned metrics drop below baseline.

## Tuning
- [ ] Compare chunk sizes 128/256/384 with overlap sweep.
- [ ] Add metadata filtering experiment (e.g., cardiology-only queries).
- [ ] Evaluate similarity_top_k sweep (2, 4, 6).

## Agent Integration (Week 8)
- [ ] Connect RAG node outputs to evaluator node policy checks.
- [ ] Add HITL prompt path for high-risk or low-confidence responses.
- [ ] Add structured output contract for final clinical answer + citations + disclaimer.

## Safety and Compliance
- [ ] Add automated check ensuring disclaimer appears in all final outputs.
- [ ] Add no-persistence audit test for user inputs and patient-like strings.
- [ ] Add incident playbook section for retrieval failure handling.
