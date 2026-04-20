#!/usr/bin/env bash
set -euo pipefail

DATA_FILE="${LONGMEMEVAL_DATA:-data/longmemeval_s_cleaned.json}"
OUTPUT_DIR="${LONGMEMEVAL_OUTPUT_DIR:-experiments/public_results/longmemeval}"
SPLIT_LABEL="${LONGMEMEVAL_SPLIT_LABEL:-longmemeval_s_cleaned}"
ANSWER_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
EXTRACTION_MODEL="${LONGMEMEVAL_EXTRACTION_MODEL:-$ANSWER_MODEL}"
JUDGE_MODEL="${LONGMEMEVAL_JUDGE_MODEL:-gpt-4o}"
EXTRACTION_MODE="${LONGMEMEVAL_EXTRACTION_MODE:-per_session}"
MAX_SESSIONS="${LONGMEMEVAL_MAX_SESSIONS:-12}"
LIMIT="${LONGMEMEVAL_LIMIT:-}"
SAMPLE_SIZE="${LONGMEMEVAL_SAMPLE_SIZE:-}"
SAMPLE_SEED="${LONGMEMEVAL_SAMPLE_SEED:-0}"
SKIP_EVALUATION="${LONGMEMEVAL_SKIP_EVALUATION:-0}"
TRACE="${LONGMEMEVAL_TRACE:-0}"

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
  echo "OPENAI_API_KEY is required. Set it before running TruthGit on LongMemEval." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
SAFE_MODEL="$(echo "$ANSWER_MODEL" | tr -c 'A-Za-z0-9_.-' '_')"
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

BASE_NAME="${SPLIT_LABEL}_truthgit_${SAFE_MODEL}_${EXTRACTION_MODE}_${RUN_LABEL}"
HYPOTHESES="${OUTPUT_DIR}/${BASE_NAME}.hypotheses.jsonl"
EVAL_LOG="${OUTPUT_DIR}/${BASE_NAME}.eval-results-${JUDGE_MODEL}.jsonl"
SUMMARY="${OUTPUT_DIR}/${BASE_NAME}.summary.json"
TRACE_DIR="${OUTPUT_DIR}/${BASE_NAME}.traces"
TRACE_ARGS=()
if [[ "$TRACE" == "1" ]]; then
  TRACE_ARGS=(--trace-dir "$TRACE_DIR")
fi

"${PYTHON_CMD[@]}" -m experiments.public_benchmarks.longmemeval_truthgit generate \
  --data "$DATA_FILE" \
  --output-jsonl "$HYPOTHESES" \
  --answer-model "$ANSWER_MODEL" \
  --extraction-model "$EXTRACTION_MODEL" \
  --extraction-mode "$EXTRACTION_MODE" \
  --max-sessions "$MAX_SESSIONS" \
  "${LIMIT_ARGS[@]}" \
  "${TRACE_ARGS[@]}"

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

TruthGit LongMemEval run complete.
Hypotheses: $HYPOTHESES
Eval log:   $EVAL_LOG
Summary:    $SUMMARY
EOF
if [[ "$TRACE" == "1" ]]; then
  echo "Traces:     $TRACE_DIR"
fi
