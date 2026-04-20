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
  1. Generate hypotheses from $OUTPUT_DIR/${SPLIT_LABEL}_prompts.jsonl.
  2. Save JSONL lines with: {"question_id": "...", "hypothesis": "..."}.
  3. Use the official LongMemEval evaluator:
     https://github.com/xiaowu0162/LongMemEval

This public benchmark is supplementary. The frozen TruthGit synthetic benchmark remains the main table.
EOF
