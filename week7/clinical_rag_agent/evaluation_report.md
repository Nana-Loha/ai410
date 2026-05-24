# Week 7 Retrieval Evaluation Report

## Metrics

| Run | Hit Rate | MRR |
|---|---:|---:|
| Baseline | 0.333 | 0.250 |
| Tuned | 0.667 | 0.667 |

## Baseline Config

- {'chunk_size': 180, 'chunk_overlap': 20, 'similarity_top_k': 2, 'enable_metadata_filter': False, 'enable_lexical_rerank': False, 'embedding_backend': 'mock', 'embedding_model_name': 'sentence-transformers/all-MiniLM-L6-v2'}

## Tuned Config

- {'chunk_size': 320, 'chunk_overlap': 80, 'similarity_top_k': 4, 'enable_metadata_filter': True, 'enable_lexical_rerank': True, 'embedding_backend': 'mock', 'embedding_model_name': 'sentence-transformers/all-MiniLM-L6-v2'}

## Observed Retrieval Failures (2-3)

- q2: expected=warfarin_aspirin_interaction.txt, returned=['chest_pain_triage.txt', 'asthma_controller_stepup.txt']
  - Query: What should be monitored when a patient takes warfarin with aspirin?
- q3: expected=hypertension_followup.txt, returned=['chest_pain_triage.txt', 'asthma_controller_stepup.txt']
  - Query: What follow-up should be done for hypertension management?
- q4: expected=diabetes_a1c_management.txt, returned=['chest_pain_triage.txt', 'asthma_controller_stepup.txt']
  - Query: How should A1c management be individualized in type 2 diabetes?

## Tuning Summary

- Increased chunk size and overlap to preserve context in guideline snippets.
- Enabled topic metadata filtering to narrow retrieval to relevant clinical domain.
- Enabled lexical reranking and increased similarity_top_k to improve recall and ranking.
- Keep deterministic corpus and fixed eval set for reproducible comparisons.
