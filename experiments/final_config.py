"""Frozen final benchmark configuration for the TruthGit paper run."""

from __future__ import annotations

from pathlib import Path

BENCHMARK_VERSION = "truthgit-benchmark-v3-phase2-final"
BACKBONE = "gpt-4o-mini"
BENCHMARK_LOGIC_COMMIT_SHA = "6b5e6d5478baa67eb0c767fcfad162d3a2b4919f"
PROMPT_TEMPLATE_PATH = Path("experiments/prompt_templates/final_answer_prompt.txt")
QUALITATIVE_FIGURE_PATH = Path("docs/figures/truthgit_qualitative_lineage.svg")
RESULT_FILES = [
    Path("experiments/results/benchmark_results.json"),
    Path("experiments/results/metric_summary.csv"),
    Path("experiments/results/question_scores.csv"),
    Path("experiments/results/predictions.csv"),
    Path("experiments/results/metric_summary.png"),
    Path("experiments/results/final_manifest.json"),
]
