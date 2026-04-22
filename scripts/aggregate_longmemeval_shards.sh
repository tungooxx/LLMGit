#!/usr/bin/env bash
set -euo pipefail

DATA_FILE="${LONGMEMEVAL_DATA:-data/longmemeval_s_cleaned.json}"
OUTPUT_DIR="${LONGMEMEVAL_OUTPUT_DIR:-experiments/public_results/longmemeval}"
SPLIT_LABEL="${LONGMEMEVAL_SPLIT_LABEL:-longmemeval_s_cleaned}"
SYSTEM_LABEL="${LONGMEMEVAL_SYSTEM_LABEL:-truthgit_gpt-4o-mini_record_batch_ms12_beliefs_and_excerpts}"
JUDGE_MODEL="${LONGMEMEVAL_JUDGE_MODEL:-gpt-4o}"
ALLOW_INCOMPLETE="${LONGMEMEVAL_ALLOW_INCOMPLETE:-0}"

if command -v py >/dev/null 2>&1; then
  PYTHON_CMD=(py -3)
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_CMD=(python.exe)
else
  PYTHON_CMD=(python)
fi

shopt -s nullglob
eval_logs=("${OUTPUT_DIR}/${SPLIT_LABEL}_${SYSTEM_LABEL}_shard_"*_limit_*.eval-results-"${JUDGE_MODEL}".jsonl)
shopt -u nullglob
if [[ ${#eval_logs[@]} -eq 0 ]]; then
  echo "No shard eval logs matched ${SPLIT_LABEL}_${SYSTEM_LABEL}_shard_*_limit_*.eval-results-${JUDGE_MODEL}.jsonl in ${OUTPUT_DIR}" >&2
  exit 2
fi

aggregate_eval_log="${OUTPUT_DIR}/${SPLIT_LABEL}_${SYSTEM_LABEL}_full_aggregated.eval-results-${JUDGE_MODEL}.jsonl"
summary_json="${OUTPUT_DIR}/${SPLIT_LABEL}_${SYSTEM_LABEL}_full_aggregated.summary.json"
args=()
for path in "${eval_logs[@]}"; do
  args+=(--eval-log "$path")
done
if [[ "$ALLOW_INCOMPLETE" == "1" ]]; then
  args+=(--allow-incomplete)
fi

"${PYTHON_CMD[@]}" -m experiments.public_benchmarks.longmemeval aggregate \
  --data "$DATA_FILE" \
  "${args[@]}" \
  --output-log "$aggregate_eval_log" \
  --output-json "$summary_json"

cat <<EOF

Aggregated LongMemEval shards.
Shard count: ${#eval_logs[@]}
Eval log:    $aggregate_eval_log
Summary:     $summary_json
EOF
