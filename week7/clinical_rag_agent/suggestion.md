# Week 7 Clinical RAG Agent Suggestions

## Current Status Summary
- `week7/clinical_rag_agent` is mostly empty (only `data/` and `storage/` directories currently visible).
- Week 6 already has a working LangGraph healthcare agent pattern (`planner -> tool -> evaluator -> HITL`).
- Existing project requirements emphasize:
  - no patient data persistence
  - mandatory medical disclaimer
  - confirmation before critical actions
  - graceful provider error handling

## Suggestions (Prioritized)

1. Reuse Week 6 graph architecture for RAG
- Keep the same node separation and safety gates.
- Add a dedicated RAG retrieval/generation node (`node3_rag.py`) after intent/tool routing.
- Keep evaluator + HITL checkpoints for high-risk outputs.

2. Define strict RAG state schema first
- Add a clear state contract in `state.py` before node logic.
- Include fields for query, retrieved chunks, citations, confidence, risk level, and final response.
- Track retry count and last error in state for self-correction.

3. Enforce citation-grounded output
- In `rag_pipeline.py`, require each clinical recommendation to map to retrieved evidence.
- If evidence is weak/missing, response must say "insufficient evidence" instead of guessing.

4. Add safety and privacy checks in evaluator
- In `eval.py`, validate:
  - disclaimer present
  - at least one citation for clinical claims
  - no direct PHI leakage in logs/outputs
  - escalation trigger for emergency-risk guidance

5. Start with deterministic local fixtures before live model calls
- Place a small guideline corpus in `data/guidelines/`.
- Build a deterministic retrieval test path first to de-risk integration.

6. Make README and SPEC executable
- `README.md` should include exact run/test commands.
- `SPEC.md` should define measurable acceptance criteria (latency, citation coverage, error behavior).

## To-Do List

### A. Foundation
- [ ] Create `week7/clinical_rag_agent/state.py` with TypedDict/dataclass for full graph state.
- [ ] Create `week7/clinical_rag_agent/rag_pipeline.py` with:
  - index/load function
  - retrieve(query) function
  - generate_with_citations(query, contexts) function
- [ ] Create `week7/clinical_rag_agent/node3_rag.py` with a pure node function that reads/writes state only.

### B. Evaluation & Safety
- [ ] Create `week7/clinical_rag_agent/eval.py` checks:
  - disclaimer coverage
  - citation presence/format
  - high-risk trigger detection
  - empty retrieval fallback
- [ ] Add a helper that blocks unsupported medical certainty language when confidence is low.

### C. Data & Storage
- [ ] Add at least 5 guideline source files under `week7/clinical_rag_agent/data/guidelines/`.
- [ ] Define storage strategy in `week7/clinical_rag_agent/storage/` (local index artifacts only, no patient logs).
- [ ] Document retention policy: no user query or patient text persisted.

### D. Docs
- [ ] Write `week7/clinical_rag_agent/SPEC.md` with:
  - user stories
  - FR/NFR
  - acceptance tests
  - safety constraints
- [ ] Write `week7/clinical_rag_agent/README.md` with setup, run, and test steps.

### E. Tests (Recommended Next)
- [ ] Add `tests/test_week7_rag_retrieval.py` for top-k retrieval behavior.
- [ ] Add `tests/test_week7_rag_response_format.py` for disclaimer + citation structure.
- [ ] Add `tests/test_week7_rag_safety.py` for emergency/escalation paths.
- [ ] Add `tests/test_week7_rag_no_persistence.py` to verify no patient data is stored.

## Suggested Execution Order
1. `state.py` -> `rag_pipeline.py` -> `node3_rag.py`
2. `eval.py` safety checks
3. `SPEC.md` and `README.md`
4. tests
5. integration with live provider(s)

## Definition of Done (Week 7)
- RAG responses include citations and disclaimer in 100% of successful runs.
- High-risk outputs trigger human confirmation path.
- Empty/failed retrieval produces safe fallback (no hallucinated clinical claims).
- No patient data written to disk/logs.
- Core Week 7 tests pass.
