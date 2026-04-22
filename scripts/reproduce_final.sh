#!/usr/bin/env bash
set -euo pipefail

if command -v py >/dev/null 2>&1; then
  PYTHON_CMD=(py -3)
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_CMD=(python.exe)
else
  PYTHON_CMD=(python)
fi

"${PYTHON_CMD[@]}" -m compileall app experiments tests
"${PYTHON_CMD[@]}" -m pytest -q
"${PYTHON_CMD[@]}" -m experiments.run_benchmark \
  --output-dir experiments/results \
  --backbone gpt-4o-mini \
  --include-ablations
"${PYTHON_CMD[@]}" -m experiments.governance_benchmark \
  --output-dir experiments/results
"${PYTHON_CMD[@]}" -m experiments.memory_ci_case_study \
  --output experiments/results/memory_ci_case_study.json
"${PYTHON_CMD[@]}" -m experiments.plot_results \
  --summary-csv experiments/results/metric_summary.csv \
  --output-png experiments/results/metric_summary.png
"${PYTHON_CMD[@]}" scripts/write_final_manifest.py

echo "Final TruthGit benchmark reproduction complete."
echo "Expected outputs:"
echo "  experiments/results/benchmark_results.json"
echo "  experiments/results/metric_summary.csv"
echo "  experiments/results/question_scores.csv"
echo "  experiments/results/predictions.csv"
echo "  experiments/results/governance_benchmark_results.json"
echo "  experiments/results/governance_metric_summary.csv"
echo "  experiments/results/governance_case_results.csv"
echo "  experiments/results/governance_routing_counts.csv"
echo "  experiments/results/governance_quarantine_metrics.png"
echo "  experiments/results/memory_ci_case_study.json"
echo "  experiments/results/metric_summary.png"
echo "  experiments/results/final_manifest.json"
