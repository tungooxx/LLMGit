# Qualitative Case Study: Project Assistant Memory

This case study is not an additional benchmark and does not change the architecture. It is a realistic narrative example for explaining why version-controlled belief memory matters.

## Scenario

A project assistant tracks a product launch schedule.

## Event Sequence

1. Initial schedule:

   Source: `launch-plan-v1`

   Claim: "The launch review is scheduled for May 12."

   TruthGit state: active belief version.

2. Later superseding update:

   Source: `calendar-update-product-lead`

   Claim: "The launch review moved to May 19."

   TruthGit state: May 19 becomes active and supersedes May 12. May 12 remains historical.

3. Conflicting low-trust update:

   Source: `unverified-slack-forward`

   Claim: "The launch review is canceled."

   TruthGit state: staged or unresolved because the source is low trust and the predicate is important.

4. Rollback:

   Source: `ops-correction`

   Claim: "The cancellation forward was wrong."

   TruthGit state: the bad cancellation update is retracted. May 19 remains the active answer.

5. Hypothetical branch:

   Branch: `contingency-plan`

   Claim: "If legal approval slips, the launch review will move to May 26."

   TruthGit state: May 26 is visible only on `contingency-plan`; it does not overwrite main.

## Questions TruthGit Can Answer

- Current truth: "When is the launch review?" -> May 19.
- Historical truth: "What was the original date?" -> May 12.
- Provenance: "Which source currently justifies May 19?" -> `calendar-update-product-lead`.
- Rollback: "Why is the cancellation not active?" -> it came from `unverified-slack-forward` and was retracted by `ops-correction`.
- Branch: "What date is planned under the contingency branch?" -> May 26 on `contingency-plan`, not on `main`.

## Why A Flat Memory Baseline Fails

A flat chat history or RAG store can retrieve all messages, but it has no durable representation of which claim is active, which claim is historical, which source governs the answer, which bad update was rolled back, or which future plan belongs only to a branch.
