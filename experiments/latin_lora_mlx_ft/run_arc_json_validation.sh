#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

RUN_ID="${1:-arc-json-validation-001}"
MODEL="${MODEL:-mlx-community/Qwen3.5-0.8B-MLX-4bit}"
TEST_FILE="${TEST_FILE:-latin_lora_data_macsafe_2048/test.jsonl}"
PROMPT_FILE="${PROMPT_FILE:-prompts/dependency_arc_json_fewshot.txt}"
MAX_TOKENS="${MAX_TOKENS:-768}"
LIMIT="${LIMIT:-}"
SYSTEM_NAME="${SYSTEM_NAME:-arc_json_llm}"
CHAT_TEMPLATE_CONFIG="${CHAT_TEMPLATE_CONFIG:-{\"enable_thinking\": false}}"
FALLBACK_ROOT="${FALLBACK_ROOT:-0}"
RUN_DIR="runs/${RUN_ID}"

mkdir -p "$RUN_DIR"

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi

LIMIT_ARGS=""
if [ -n "$LIMIT" ]; then
  LIMIT_ARGS="--limit $LIMIT"
fi

FALLBACK_ARGS=""
if [ "$FALLBACK_ROOT" = "1" ]; then
  FALLBACK_ARGS="--fallback-root"
fi

python3 scripts/extract_gold_conllu_from_mlx_jsonl.py \
  "$TEST_FILE" \
  "$RUN_DIR/gold.conllu" \
  $LIMIT_ARGS

.venv/bin/python scripts/generate_mlx_arc_json_predictions.py \
  --input "$TEST_FILE" \
  --output-jsonl "$RUN_DIR/${SYSTEM_NAME}.arcs.jsonl" \
  --output-conllu "$RUN_DIR/${SYSTEM_NAME}.conllu" \
  --report "$RUN_DIR/${SYSTEM_NAME}.arc_report.json" \
  --model "$MODEL" \
  --system-prompt-file "$PROMPT_FILE" \
  --chat-template-config "$CHAT_TEMPLATE_CONFIG" \
  --max-tokens "$MAX_TOKENS" \
  $LIMIT_ARGS \
  $FALLBACK_ARGS

python3 scripts/report_conllu_prediction_quality.py \
  "$RUN_DIR/${SYSTEM_NAME}.conllu" \
  > "$RUN_DIR/format_report.txt"

python3 scripts/score_single_conllu_run.py \
  --gold "$RUN_DIR/gold.conllu" \
  --pred "$RUN_DIR/${SYSTEM_NAME}.conllu" \
  --system-name "$SYSTEM_NAME" \
  --run-name "$RUN_ID" \
  --model "$MODEL"
