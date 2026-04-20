# TruthGit: Version-Controlled Belief Memory for LLM Agents

## Draft Status

This draft freezes the Benchmark v3 phase 2 result generated with `gpt-4o-mini` as the backbone label. The benchmark is deterministic and synthetic, but it now stresses dynamic truth maintenance rather than simple fact recall.

## Abstract

Long-running LLM agents operate in changing worlds: facts become stale, sources conflict, hypotheses should remain isolated, and bad memory writes must be reversible. Retrieval-augmented generation and chat-history memory can recall old text, but they do not directly represent which belief is current, which source governs it, or why a prior version was superseded. TruthGit is a research prototype that stores memory as version-controlled atomic beliefs. Each belief version records provenance, branch, commit lineage, status, temporal validity, supersession, contradiction groups, and audit events. In a synthetic changing-world benchmark with 86 cases and 161 structured questions, TruthGit reaches 1.0 across current truth, ordered history, provenance, rollback recovery, branch isolation, merge conflict resolution, and low-trust warning metrics, while naive history, simple RAG, and TF-IDF embedding RAG fail on columns requiring explicit memory structure.

## Hypothesis

LLM agents will answer changing-world memory questions more reliably when durable memory is represented as version-controlled belief state rather than flat chat history or unversioned retrieval chunks.

More specifically:

- branch metadata should improve hypothetical isolation;
- rollback metadata should improve recovery from bad writes and rollback-aware history;
- provenance metadata should improve exact current-source citation;
- contradiction groups and temporal windows should improve merge conflict handling.

## Related Work

Recent memory evaluation work has moved beyond single-shot recall. StoryBench argues for dynamic long-term-memory evaluation with multi-turn sequential reasoning. MEMTRACK evaluates long-term memory and state tracking in dynamic multi-platform agent environments. StructMemEval, introduced in "Evaluating Memory Structure in LLM Agents," directly argues that memory benchmarks should test how agents organize memory rather than only whether they recall stored facts.

TruthGit follows the same direction but focuses on a different research question: whether explicit version-control semantics for atomic beliefs improve state tracking under supersession, rollback, branching, provenance, and merge conflict.

References:

- Luanbo Wan and Weizhi Ma. 2025. "StoryBench: A Dynamic Benchmark for Evaluating Long-Term Memory with Multi Turns." arXiv:2506.13356. https://arxiv.org/abs/2506.13356
- Darshan Deshpande et al. 2025. "MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments." arXiv:2510.01353. https://arxiv.org/abs/2510.01353
- Alina Shutova, Alexandra Olenina, Ivan Vinogradov, and Anton Sinitsin. 2026. "Evaluating Memory Structure in LLM Agents." arXiv:2602.11243. https://arxiv.org/abs/2602.11243

## System

TruthGit stores long-term memory in SQLite through deterministic Python services. The LLM may extract candidate claims and explain results, but the database and commit engine decide whether memory changes are staged, approved, superseded, merged, rolled back, or flagged for review.

Core tables:

- `Source`: provenance and trust score.
- `Branch`: main or hypothetical belief branch.
- `Commit`: add, update, merge, retract, rollback operation.
- `Belief`: stable subject+predicate identity.
- `BeliefVersion`: object value, status, source, confidence, temporal window, branch, supersession, contradiction group.
- `StagedCommit`: durable review queue for proposed writes.
- `AuditEvent`: append-only operational audit log.

## Benchmark V3 Phase 2

Benchmark v3 phase 2 contains 86 generated worlds and 161 structured questions.

Question categories:

- current truth after supersession;
- exact ordered historical timeline;
- date-aware time-slice history;
- provenance for the exact current governing source;
- rollback recovery after bad commits;
- branch-only hypothetical isolation;
- resolved and unresolved merge conflict;
- low-trust warning behavior.

New phase 2 hard cases:

- multiple sources mention the same winning fact;
- a rolled-back source mentions the same object and should no longer be cited;
- main and branch have the same object but different governing sources;
- a merge has a winning branch source and a weaker later conflicting branch;
- main changes after branch fork;
- two competing branches are merged into main;
- merge values coexist temporally instead of overwriting.

## Systems Compared

- `naive_chat_history`: flat append-only memory.
- `simple_rag`: subject/predicate retrieval with source-trust sorting.
- `embedding_rag`: local TF-IDF cosine retrieval over memory chunks.
- `truthgit`: real TruthGit branch, commit, rollback, conflict, provenance, and audit engines.

Ablations:

- `truthgit_no_branches`
- `truthgit_no_rollback`
- `truthgit_no_review_gate`
- `truthgit_no_trust_scoring`

## Main Result

| System | Current | History | Provenance | Rollback | Branch | Merge | Low-trust |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive chat history | 0.545 | 0.545 | 0.661 | 0.000 | 0.500 | 0.400 | 0.000 |
| simple RAG | 1.000 | 0.545 | 0.729 | 0.000 | 0.500 | 0.350 | 0.000 |
| embedding RAG | 1.000 | 0.273 | 0.729 | 0.000 | 0.500 | 0.400 | 0.000 |
| TruthGit | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Interpretation

Retrieval baselines can often recover the current object when the benchmark only asks "what is true now?" Simple RAG and embedding RAG both reach 1.0 on current truth because the current object is usually present in a high-trust or recent chunk.

The gap appears when the question requires memory structure:

- History requires exact ordered timelines and date-aware slices. Retrieval returns relevant chunks but does not maintain a canonical timeline.
- Provenance requires the exact source that currently justifies an answer, not merely any source that mentions the object.
- Rollback requires retraction semantics. Flat baselines keep the bad chunk.
- Branch isolation requires branch-local state. Flat baselines leak hypothetical facts into main.
- Merge conflict requires contradiction groups, branch lineage, and temporal coexistence. Retrieval can return text but cannot decide whether a merge should remain unresolved.
- Low-trust warning requires a review gate or trust-aware write policy.

The ablations support the structural claim. Removing branches hurts branch isolation and branch-specific provenance. Removing rollback destroys rollback recovery and rollback-aware history/provenance. Removing the review gate destroys low-trust warnings. Removing trust scoring weakens current-truth behavior under poisoning.

## Limitations

The benchmark is synthetic and deterministic. It does not measure full natural-language answer quality or human preference. The embedding baseline is TF-IDF rather than a neural retriever with learned reranking. TruthGit's merge policy is hand-written and deterministic. Same-object corroboration is represented as a new governing belief version; a production system should also preserve all corroborating sources as a support set. The benchmark should be expanded with larger stochastic worlds, noisier source text, and live LLM extraction errors.

## Future Work

- Add neural embedding and reranking baselines.
- Add stochastic world generation with hidden source reliability.
- Add adversarial poisoning and delayed rollback scenarios.
- Evaluate human-readable conflict explanations.
- Learn branch creation and merge policies from feedback.
- Add support-set provenance so multiple sources can jointly justify one belief version.
