# Week 7 Clinical RAG Agent SPEC (Updated Draft)

## Scope Finalization (Lab 7.4)

### Framework Choice
- Workflow orchestration: LangGraph (Week 8 implementation target)
- Retrieval/indexing: LlamaIndex over local guideline corpus
- Runtime language: Python 3.12+

### Team Roles
- Retrieval + data pipeline owner: builds ingestion, indexing, and corpus QA
- Evaluation owner: maintains fixed test set, metrics, and failure analysis
- Agent integration owner: wires planner, RAG node, evaluator, and HITL flow
- Documentation owner: keeps SPEC, backlog, architecture, and reports current

### Architecture Decisions
- Keep local guideline corpus under data/guidelines for deterministic iteration.
- Persist only index artifacts in storage/index; never persist patient prompts.
- Require citation-grounded outputs for clinical suggestions.
- Escalate to human review when retrieval confidence is low or risk is high.

## Week 8 Functional Requirements
- FR-01: System must ingest guideline text corpus reproducibly and write an ingestion manifest.
- FR-02: System must retrieve top-k evidence chunks with source citations.
- FR-03: System must report retrieval metrics (Hit Rate and MRR) on a fixed query set.
- FR-04: System must support tuning via chunking and retrieval parameter changes.
- FR-05: System must include disclaimer on every final answer.
- FR-06: System must never persist patient query or patient note content to disk.

## Non-Functional Requirements
- NFR-01: Evaluation run should complete in under 2 minutes on local machine.
- NFR-02: Retrieval evaluation must be reproducible from committed corpus and eval set.
- NFR-03: Fail-safe output must be shown when no evidence is found.

## Deliverables Mapping
- Lab 7.1: rag_pipeline.py + rag_config.json + corpus files + ingestion manifest
- Lab 7.2: eval.py + evaluation_report.md (Hit Rate, MRR, failures)
- Lab 7.3: baseline vs tuned config comparison in evaluation report
- Lab 7.4: this SPEC draft + tasks backlog + architecture_diagram.md
