# TruthGit Final Reproducibility Pack

This file freezes the final experiment setup for the TruthGit submission draft.

## Locked Setup

| Item | Value |
| --- | --- |
| Benchmark version | `truthgit-benchmark-v3-phase2-final` |
| Backbone label | `gpt-4o-mini` |
| Benchmark logic commit | `6b5e6d5478baa67eb0c767fcfad162d3a2b4919f` |
| Prompt template | `experiments/prompt_templates/final_answer_prompt.txt` |
| Manifest | `experiments/results/final_manifest.json` |
| Paper draft | `docs/paper_draft.md` |
| Qualitative figure | `docs/figures/truthgit_qualitative_lineage.svg` |
| Qualitative case study | `docs/case_study_project_assistant.md` |
| Metric figure | `experiments/results/metric_summary.png` |

The benchmark generator and scoring rules are frozen at this version. Do not change them unless an implementation bug is found.

## Exact Commands

Run the full reproduction script:

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

## Expected Output Files

- `experiments/results/benchmark_results.json`
- `experiments/results/metric_summary.csv`
- `experiments/results/question_scores.csv`
- `experiments/results/predictions.csv`
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
- default data file: `data/longmemeval_s_cleaned.json`

Run after downloading the official cleaned split:

```bash
LONGMEMEVAL_DATA=data/longmemeval_s_cleaned.json bash scripts/run_longmemeval.sh
```

This produces prompt JSONL and a dataset manifest under `experiments/public_results/longmemeval/`. It does not alter the frozen TruthGit result files.

## Final Result Table

| System | Current | History | Provenance | Rollback | Branch | Merge | Low-trust |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive chat history | 0.545 | 0.545 | 0.661 | 0.000 | 0.500 | 0.400 | 0.000 |
| simple RAG | 1.000 | 0.545 | 0.729 | 0.000 | 0.500 | 0.350 | 0.000 |
| embedding RAG | 1.000 | 0.273 | 0.729 | 0.000 | 0.500 | 0.400 | 0.000 |
| TruthGit | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Final Interpretation

The retrieval baselines can often recover the current fact, but they fail when the question requires changing-world state management: exact history, exact current provenance, rollback recovery, branch isolation, merge conflict state, and low-trust review warnings.

The ablations show that each TruthGit mechanism protects a separate capability:

- branches protect hypothetical isolation and branch-specific provenance;
- rollback protects bad-commit recovery and invalidated source cleanup;
- review gates protect low-trust warning behavior;
- trust scoring protects current truth under poisoning and source selection.
