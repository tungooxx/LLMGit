# LongMemEval Public Benchmark Track

TruthGit's frozen synthetic benchmark remains the main table. LongMemEval is the first public benchmark track and should be reported as a supplementary external evaluation.

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

## Prompt-Only Adapter

Inspect the dataset and create prompt JSONL:

```bash
LONGMEMEVAL_DATA=data/longmemeval_s_cleaned.json \
  bash scripts/run_longmemeval.sh
```

This writes:

- `experiments/public_results/longmemeval/longmemeval_s_cleaned_manifest.json`
- `experiments/public_results/longmemeval/longmemeval_s_cleaned_prompts.jsonl`

The prompt JSONL is an intermediate artifact. It intentionally does not expose `answer`, `answer_session_ids`, or `has_answer` evidence labels unless `--include-evidence-markers` is explicitly used for debugging.

## Real Full Benchmark Run

The full-history runner measures a non-leaking `gpt-4o-mini` baseline over the LongMemEval chat history. It does not ingest memory into TruthGit.

Run a small prompt-only smoke test first:

```powershell
.\scripts\run_longmemeval_full.ps1 -Limit 3
```

Run the full LongMemEval-S cleaned benchmark with `gpt-4o-mini` as the answer model and `gpt-4o` as the official-style judge:

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENAI_MODEL = "gpt-4o-mini"
.\scripts\run_longmemeval_full.ps1
```

The full run downloads the 277 MB `longmemeval_s_cleaned.json` file if needed, generates official hypothesis JSONL, evaluates with the LongMemEval answer-check prompt, and writes a summary JSON.

Outputs:

- `experiments/public_results/longmemeval/*prompts.jsonl`
- `experiments/public_results/longmemeval/*hypotheses.jsonl`
- `experiments/public_results/longmemeval/*eval-results-gpt-4o.jsonl`
- `experiments/public_results/longmemeval/*summary.json`

The hypothesis file follows the official contract:

```json
{"question_id": "example-id", "hypothesis": "the system answer"}
```

The local evaluator mirrors the official QA judge prompts. For a final paper number, keep the generated hypotheses and also run the upstream evaluator from the official repository:

```bash
cd LongMemEval/src/evaluation
python3 evaluate_qa.py gpt-4o ../../../experiments/public_results/longmemeval/your.hypotheses.jsonl ../../data/longmemeval_s_cleaned.json
python3 print_qa_metrics.py ../../../experiments/public_results/longmemeval/your.hypotheses.jsonl.eval-results-gpt-4o ../../data/longmemeval_s_cleaned.json
```

## TruthGit On LongMemEval

Use this path for the real public benchmark comparison against systems such as Hindsight, TiMem, LightMem, and Zep. It is different from the prompt-only runner:

```text
LongMemEval sessions
-> OpenAI structured claim extraction
-> TruthGit staged commits
-> approval into commits
-> Source / Belief / BeliefVersion / AuditEvent records
-> TruthGit memory context
-> gpt-4o-mini answer
-> official-style LongMemEval judge
```

Smoke test:

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENAI_MODEL = "gpt-4o-mini"
.\scripts\run_longmemeval_truthgit.ps1 -Limit 3 -Trace
```

Full TruthGit LongMemEval-S run:

```powershell
.\scripts\run_longmemeval_truthgit.ps1
```

Default settings:

- answer model: `gpt-4o-mini`
- extraction model: `gpt-4o-mini`
- extraction mode: `per_session`
- selected sessions per question: `12`
- judge model: `gpt-4o`

Useful options:

```powershell
# Cheaper but less version-control faithful: one extraction call per selected record.
.\scripts\run_longmemeval_truthgit.ps1 -ExtractionMode record_batch -Limit 10

# More sessions, more cost, usually better recall.
.\scripts\run_longmemeval_truthgit.ps1 -MaxSessions 24 -Limit 10

# Generate hypotheses without judge evaluation.
.\scripts\run_longmemeval_truthgit.ps1 -Limit 10 -SkipEvaluation

# Held-out smoke test after tuning the first examples.
.\scripts\run_longmemeval_truthgit.ps1 -SampleSize 50 -SampleSeed 20260420 -NoResume
```

Outputs:

- `experiments/public_results/longmemeval/*truthgit*.hypotheses.jsonl`
- `experiments/public_results/longmemeval/*truthgit*.eval-results-gpt-4o.jsonl`
- `experiments/public_results/longmemeval/*truthgit*.summary.json`
- optional trace files under `*.traces/`

## Development Honesty Rule

Small `-Limit 10` runs are smoke tests for adapter debugging only. They are not publishable LongMemEval scores, especially after inspecting failures and changing extraction or retrieval logic.

For a fair public number:

1. freeze the adapter code and prompts;
2. run on a held-out random sample or the full LongMemEval-S split;
3. do not inspect gold answers before changing the code again;
4. report the exact command, model, split, limit, and commit SHA;
5. keep development-smoke results separate from final evaluation results.

The TruthGit adapter may use deterministic, source-text-only normalization rules, but those rules must be general and documented. It must not use `answer`, `answer_session_ids`, `has_answer`, or judge labels during generation.

Predicate labels in TruthGit's LongMemEval path are generated by the claim extractor model as open snake-case relation names. Python may normalize obvious aliases for stable keys, but it does not use a closed predicate ontology or benchmark-specific predicate labels.

## Competitive Plan

The first honest target is a full-history, non-leaking `gpt-4o-mini` run. LongMemEval-S fits in a 128k context model, so this establishes a strong public baseline.

The second target is the TruthGit run above. Report it separately from the prompt-only baseline:

| System | LongMemEval-S |
| --- | ---: |
| full-history gpt-4o-mini | measured locally |
| TruthGit + gpt-4o-mini | measured locally |
| Hindsight / TiMem / LightMem / Zep | cite reported numbers with exact split caveats |

To improve toward published systems:

- compare `reader_mode=con` against `reader_mode=direct`
- compare JSON history against natural-language history
- run `gpt-4o-mini` first, then test a stronger answer model if budget allows
- inspect errors by `question_type`
- compare TruthGit `per_session` against cheaper `record_batch`
- tune `MaxSessions` and trace failed records to improve extraction/retrieval

## Reporting Rule

Report two separate tables:

1. Main table: frozen TruthGit Benchmark v3 phase 2.
2. Public benchmark table: LongMemEval.

Do not merge LongMemEval into the frozen TruthGit benchmark or change the synthetic benchmark after freeze.
