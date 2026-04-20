# TruthGit Final Reproducibility Pack

This file freezes the final experiment setup for the TruthGit submission draft.

## Locked Setup

| Item | Value |
| --- | --- |
| Benchmark version | `truthgit-benchmark-v3-phase2-final` |
| Structural table backbone label | `gpt-4o-mini` metadata only |
| Benchmark logic commit | `6b5e6d5478baa67eb0c767fcfad162d3a2b4919f` |
| Prompt template | `experiments/prompt_templates/final_answer_prompt.txt` |
| Manifest | `experiments/results/final_manifest.json` |
| Paper draft | `docs/paper_draft.md` |
| Qualitative figure | `docs/figures/truthgit_qualitative_lineage.svg` |
| Qualitative case study | `docs/case_study_project_assistant.md` |
| Metric figure | `experiments/results/metric_summary.png` |

The benchmark generator and scoring rules are frozen at this version. Do not change them unless an implementation bug is found. The main frozen table is a structural memory correctness benchmark, not a live LLM reasoning benchmark.

## Exact Commands

Run the structural reproduction script:

```bash
bash scripts/reproduce_final.sh
```

Equivalent individual commands:

```bash
python -m compileall app experiments tests
python -m pytest -q
python -m experiments.run_benchmark --output-dir experiments/results --backbone gpt-4o-mini --include-ablations
python -m experiments.plot_results --summary-csv experiments/results/metric_summary.csv --output-png experiments/results/metric_summary.png
python scripts/write_final_manifest.py
```

Optional model-in-the-loop table, requiring `OPENAI_API_KEY`:

```bash
python -m experiments.reader_benchmark --output-dir experiments/results --reader openai --reader-model gpt-4o-mini
```

## Expected Output Files

- `experiments/results/benchmark_results.json`
- `experiments/results/metric_summary.csv`
- `experiments/results/question_scores.csv`
- `experiments/results/predictions.csv`
- `experiments/results/reader_benchmark_results.json`
- `experiments/results/reader_metric_summary.csv`
- `experiments/results/reader_question_scores.csv`
- `experiments/results/reader_predictions.csv`
- `experiments/results/metric_summary.png`
- `experiments/results/final_manifest.json`
- `docs/figures/truthgit_qualitative_lineage.svg`

The manifest records SHA-256 checksums for the prompt template, qualitative figure, and exported result files.

## Public Benchmark Track

The frozen TruthGit benchmark remains the main table. LongMemEval is the first public benchmark track and should be reported separately as an external supplementary table.

LongMemEval setup:

- adapter: `experiments/public_benchmarks/longmemeval.py`
- instructions: `docs/public_benchmarks/longmemeval.md`
- runner: `scripts/run_longmemeval.sh`
- full generator/evaluator: `scripts/run_longmemeval_full.ps1` or `scripts/run_longmemeval_full.sh`
- TruthGit generator/evaluator: `scripts/run_longmemeval_truthgit.ps1` or `scripts/run_longmemeval_truthgit.sh`
- default data file: `data/longmemeval_s_cleaned.json`

Run after downloading the official cleaned split:

```bash
LONGMEMEVAL_DATA=data/longmemeval_s_cleaned.json bash scripts/run_longmemeval.sh
```

This produces prompt JSONL and a dataset manifest under `experiments/public_results/longmemeval/`. It does not alter the frozen TruthGit result files.

Real public benchmark run:

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENAI_MODEL = "gpt-4o-mini"
.\scripts\run_longmemeval_full.ps1
```

Run `.\scripts\run_longmemeval_full.ps1 -Limit 3` first as a smoke test. The full run generates non-leaking hypotheses, evaluates with the official-style GPT-4o judge prompt, and writes `*.summary.json` under `experiments/public_results/longmemeval/`.

Real TruthGit public benchmark run:

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENAI_MODEL = "gpt-4o-mini"
.\scripts\run_longmemeval_truthgit.ps1 -Limit 3 -Trace
.\scripts\run_longmemeval_truthgit.ps1
```

This path creates a fresh TruthGit store per LongMemEval record, extracts claims from selected sessions, persists staged commits, approves them into commits, stores belief versions/sources/audit events, answers from TruthGit memory, and evaluates with the same LongMemEval judge path.

LongMemEval reporting rule: `-Limit` runs are adapter smoke tests, not final public scores. If failures are inspected and code is changed, rerun on a fresh held-out sample or the full split before reporting. Generation must not use `answer`, `answer_session_ids`, `has_answer`, or judge labels.

Predicate labels are open model-generated relation names. The deterministic layer only normalizes obvious aliases for stable keys; it does not impose a closed predicate ontology on LongMemEval extraction.

## Structural Memory Correctness Table

| System | Current | History | Provenance | Rollback | Branch | Merge | Low-trust |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive chat history | 0.545 | 0.545 | 0.661 | 0.000 | 0.500 | 0.400 | 0.000 |
| simple RAG | 1.000 | 0.545 | 0.729 | 0.000 | 0.500 | 0.350 | 0.000 |
| embedding RAG | 1.000 | 0.273 | 0.729 | 0.000 | 0.500 | 0.400 | 0.000 |
| TruthGit | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Final Interpretation

The retrieval baselines can often recover the current fact, but they fail when the question requires changing-world state management: exact history, exact current provenance, rollback recovery, branch isolation, merge conflict state, and low-trust review warnings.

Report this as a structural systems result. The deterministic runner calls `system.ingest_event(event)` and `system.answer(question)`; it does not call `gpt-4o-mini` in the scoring loop. The separate model-in-the-loop runner is `python -m experiments.reader_benchmark`, which feeds each system's retrieved memory context to the same reader model and writes `reader_*` result files.

The ablations show that each TruthGit mechanism protects a separate capability:

- branches protect hypothetical isolation and branch-specific provenance;
- rollback protects bad-commit recovery and invalidated source cleanup;
- review gates protect low-trust warning behavior;
- trust scoring protects current truth under poisoning and source selection.
