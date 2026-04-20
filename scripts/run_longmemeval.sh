#!/usr/bin/env bash
set -euo pipefail

if command -v py >/dev/null 2>&1; then
  PYTHON_CMD=(py -3)
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_CMD=(python.exe)
else
  PYTHON_CMD=(python)
fi

DATA_FILE="${LONGMEMEVAL_DATA:-data/longmemeval_s_cleaned.json}"
OUTPUT_DIR="${LONGMEMEVAL_OUTPUT_DIR:-experiments/public_results/longmemeval}"
SPLIT_LABEL="${LONGMEMEVAL_SPLIT_LABEL:-longmemeval_s_cleaned}"

if [[ ! -f "$DATA_FILE" ]]; then
  cat <<EOF
LongMemEval data file not found: $DATA_FILE

Download the official cleaned split first:
  mkdir -p data
  curl -L -o data/longmemeval_s_cleaned.json \\
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

Then rerun:
  LONGMEMEVAL_DATA=data/longmemeval_s_cleaned.json bash scripts/run_longmemeval.sh
EOF
  exit 2
fi

"${PYTHON_CMD[@]}" -m experiments.public_benchmarks.longmemeval inspect \
  --data "$DATA_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --split-label "$SPLIT_LABEL"

"${PYTHON_CMD[@]}" -m experiments.public_benchmarks.longmemeval make-prompts \
  --data "$DATA_FILE" \
  --output-jsonl "$OUTPUT_DIR/${SPLIT_LABEL}_prompts.jsonl"

cat <<EOF
LongMemEval public-benchmark inputs are ready.

Next:
  1. Smoke test the real generator/evaluator:
     LONGMEMEVAL_LIMIT=3 bash scripts/run_longmemeval_full.sh
  2. Run the full generator/evaluator:
     bash scripts/run_longmemeval_full.sh
  3. Keep the generated hypothesis JSONL for the official upstream evaluator:
     https://github.com/xiaowu0162/LongMemEval
  4. Run TruthGit itself on LongMemEval:
     LONGMEMEVAL_LIMIT=3 bash scripts/run_longmemeval_truthgit.sh

This public benchmark is supplementary. The frozen TruthGit synthetic benchmark remains the main table.
EOF
