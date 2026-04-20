#!/usr/bin/env bash
set -euo pipefail

DATA_FILE="${LONGMEMEVAL_DATA:-data/longmemeval_s_cleaned.json}"
OUTPUT_DIR="${LONGMEMEVAL_OUTPUT_DIR:-experiments/public_results/longmemeval}"
SPLIT_LABEL="${LONGMEMEVAL_SPLIT_LABEL:-longmemeval_s_cleaned}"
MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
JUDGE_MODEL="${LONGMEMEVAL_JUDGE_MODEL:-gpt-4o}"
LIMIT="${LONGMEMEVAL_LIMIT:-}"
SAMPLE_SIZE="${LONGMEMEVAL_SAMPLE_SIZE:-}"
SAMPLE_SEED="${LONGMEMEVAL_SAMPLE_SEED:-0}"
HISTORY_FORMAT="${LONGMEMEVAL_HISTORY_FORMAT:-json}"
READER_MODE="${LONGMEMEVAL_READER_MODE:-con}"
MAX_OUTPUT_TOKENS="${LONGMEMEVAL_MAX_OUTPUT_TOKENS:-256}"
SKIP_EVALUATION="${LONGMEMEVAL_SKIP_EVALUATION:-0}"

if command -v py >/dev/null 2>&1; then
  PYTHON_CMD=(py -3)
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_CMD=(python.exe)
else
  PYTHON_CMD=(python)
fi

if [[ ! -f "$DATA_FILE" ]]; then
  echo "Downloading LongMemEval-S cleaned data to $DATA_FILE"
  "${PYTHON_CMD[@]}" -m experiments.public_benchmarks.longmemeval download \
    --split s_cleaned \
    --output "$DATA_FILE"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required for generation/evaluation." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
SAFE_MODEL="$(echo "$MODEL" | tr -c 'A-Za-z0-9_.-' '_')"
if [[ -n "$LIMIT" && -n "$SAMPLE_SIZE" ]]; then
  echo "Use either LONGMEMEVAL_LIMIT or LONGMEMEVAL_SAMPLE_SIZE, not both." >&2
  exit 2
fi
if [[ -n "$SAMPLE_SIZE" ]]; then
  RUN_LABEL="random_${SAMPLE_SIZE}_seed_${SAMPLE_SEED}"
  LIMIT_ARGS=(--sample-size "$SAMPLE_SIZE" --sample-seed "$SAMPLE_SEED")
elif [[ -n "$LIMIT" ]]; then
  RUN_LABEL="sample_${LIMIT}"
  LIMIT_ARGS=(--limit "$LIMIT")
else
  RUN_LABEL="full"
  LIMIT_ARGS=()
fi

BASE_NAME="${SPLIT_LABEL}_${SAFE_MODEL}_${READER_MODE}_${RUN_LABEL}"
PROMPTS="${OUTPUT_DIR}/${BASE_NAME}.prompts.jsonl"
HYPOTHESES="${OUTPUT_DIR}/${BASE_NAME}.hypotheses.jsonl"
EVAL_LOG="${OUTPUT_DIR}/${BASE_NAME}.eval-results-${JUDGE_MODEL}.jsonl"
SUMMARY="${OUTPUT_DIR}/${BASE_NAME}.summary.json"

"${PYTHON_CMD[@]}" -m experiments.public_benchmarks.longmemeval inspect \
  --data "$DATA_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --split-label "$SPLIT_LABEL" \
  "${LIMIT_ARGS[@]}"

"${PYTHON_CMD[@]}" -m experiments.public_benchmarks.longmemeval make-prompts \
  --data "$DATA_FILE" \
  --output-jsonl "$PROMPTS" \
  --history-format "$HISTORY_FORMAT" \
  --reader-mode "$READER_MODE" \
  "${LIMIT_ARGS[@]}"

"${PYTHON_CMD[@]}" -m experiments.public_benchmarks.longmemeval generate \
  --data "$DATA_FILE" \
  --output-jsonl "$HYPOTHESES" \
  --model "$MODEL" \
  --history-format "$HISTORY_FORMAT" \
  --reader-mode "$READER_MODE" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  "${LIMIT_ARGS[@]}"

if [[ "$SKIP_EVALUATION" != "1" ]]; then
  "${PYTHON_CMD[@]}" -m experiments.public_benchmarks.longmemeval evaluate \
    --data "$DATA_FILE" \
    --hypotheses "$HYPOTHESES" \
    --output-log "$EVAL_LOG" \
    --judge-model "$JUDGE_MODEL" \
    "${LIMIT_ARGS[@]}"

  "${PYTHON_CMD[@]}" -m experiments.public_benchmarks.longmemeval summarize \
    --data "$DATA_FILE" \
    --eval-log "$EVAL_LOG" \
    --output-json "$SUMMARY" \
    "${LIMIT_ARGS[@]}"
fi

cat <<EOF

LongMemEval run complete.
Prompts:    $PROMPTS
Hypotheses: $HYPOTHESES
Eval log:   $EVAL_LOG
Summary:    $SUMMARY
EOF
