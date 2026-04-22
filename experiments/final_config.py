"""Frozen final benchmark configuration for the TruthGit paper run."""

from __future__ import annotations

from pathlib import Path

BENCHMARK_VERSION = "truthgit-benchmark-v4-support-ci-final"
BACKBONE = "gpt-4o-mini"
BENCHMARK_LOGIC_COMMIT_SHA = "pending-next-commit"
PROMPT_TEMPLATE_PATH = Path("experiments/prompt_templates/final_answer_prompt.txt")
QUALITATIVE_FIGURE_PATH = Path("docs/figures/truthgit_qualitative_lineage.svg")
RESULT_FILES = [
    Path("experiments/results/benchmark_results.json"),
    Path("experiments/results/metric_summary.csv"),
    Path("experiments/results/question_scores.csv"),
    Path("experiments/results/predictions.csv"),
    Path("experiments/results/governance_benchmark_results.json"),
    Path("experiments/results/governance_metric_summary.csv"),
    Path("experiments/results/governance_case_results.csv"),
    Path("experiments/results/governance_routing_counts.csv"),
    Path("experiments/results/governance_quarantine_metrics.png"),
    Path("experiments/results/memory_ci_case_study.json"),
    Path("experiments/results/metric_summary.png"),
    Path("experiments/results/final_manifest.json"),
]
