# LongMemEval Public Benchmark Track

TruthGit's frozen synthetic benchmark remains the main table. LongMemEval is the first public benchmark track and should be reported as a supplementary external evaluation once hypotheses are generated and scored.

## Why LongMemEval

LongMemEval is an ICLR 2025 benchmark for long-term interactive memory in chat assistants. It provides 500 questions and tests information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention. Those categories are close enough to TruthGit's motivation to be useful, while still being public and independently maintained.

Official resources:

- Paper: https://arxiv.org/abs/2410.10813
- Code: https://github.com/xiaowu0162/LongMemEval
- Cleaned dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

## Position In The Paper

Use the frozen TruthGit benchmark as the primary result table because it directly tests version-controlled belief operations: supersession, rollback, branch isolation, provenance, merge conflict, temporal coexistence, and review gates.

Use LongMemEval as the first public benchmark because it gives an external check that the memory system can operate on public long-term chat-memory data. It should not replace the main table because LongMemEval is broader conversational QA and does not directly score all TruthGit-specific version-control operations.

## Data Setup

Download the cleaned small split:

```bash
mkdir -p data
curl -L -o data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

Optional official files:

```bash
curl -L -o data/longmemeval_oracle.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
curl -L -o data/longmemeval_m_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json
```

## TruthGit Adapter

Inspect the dataset and create prompt JSONL:

```bash
LONGMEMEVAL_DATA=data/longmemeval_s_cleaned.json \
  bash scripts/run_longmemeval.sh
```

This writes:

- `experiments/public_results/longmemeval/longmemeval_s_cleaned_manifest.json`
- `experiments/public_results/longmemeval/longmemeval_s_cleaned_prompts.jsonl`

The prompt JSONL is an intermediate artifact. A later answer-generation step should read each prompt, query the memory system, and write official LongMemEval hypothesis JSONL:

```json
{"question_id": "example-id", "hypothesis": "the system answer"}
```

Then use the official LongMemEval evaluator from the upstream repository.

## Reporting Rule

Report two separate tables:

1. Main table: frozen TruthGit Benchmark v3 phase 2.
2. Public benchmark table: LongMemEval.

Do not merge LongMemEval into the frozen TruthGit benchmark or change the synthetic benchmark after freeze.
