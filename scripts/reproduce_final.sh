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
echo "  experiments/results/metric_summary.png"
echo "  experiments/results/final_manifest.json"
