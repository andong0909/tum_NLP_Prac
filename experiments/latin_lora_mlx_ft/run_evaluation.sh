#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

RUN_ID="${1:-qwen-conllu-eval-001}"
MODEL="${MODEL:-mlx-community/Qwen3-0.6B-4bit}"
ADAPTER_PATH="${ADAPTER_PATH:-adapters/latin-qwen-copycols-prompt-001}"
TEST_FILE="${TEST_FILE:-latin_lora_data/test.jsonl}"
LIMIT="${LIMIT:-}"
MAX_TOKENS="${MAX_TOKENS:-768}"
RUN_DIR="runs/${RUN_ID}"

mkdir -p "$RUN_DIR"

if [ ! -x .venv/bin/mlx_lm.generate ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi

LIMIT_ARGS=""
if [ -n "$LIMIT" ]; then
  LIMIT_ARGS="--limit $LIMIT"
fi

python3 scripts/extract_gold_conllu_from_mlx_jsonl.py \
  "$TEST_FILE" \
  "$RUN_DIR/gold.conllu" \
  $LIMIT_ARGS

.venv/bin/python scripts/generate_mlx_conllu_predictions.py \
  --input "$TEST_FILE" \
  --output "$RUN_DIR/base_qwen.conllu" \
  --model "$MODEL" \
  --max-tokens "$MAX_TOKENS" \
  $LIMIT_ARGS

.venv/bin/python scripts/generate_mlx_conllu_predictions.py \
  --input "$TEST_FILE" \
  --output "$RUN_DIR/lora_qwen.conllu" \
  --model "$MODEL" \
  --adapter-path "$ADAPTER_PATH" \
  --max-tokens "$MAX_TOKENS" \
  $LIMIT_ARGS

python3 scripts/report_conllu_prediction_quality.py \
  "$RUN_DIR/base_qwen.conllu" \
  "$RUN_DIR/lora_qwen.conllu" \
  > "$RUN_DIR/format_report.txt"

python3 scripts/score_conllu_run.py \
  --gold "$RUN_DIR/gold.conllu" \
  --base-pred "$RUN_DIR/base_qwen.conllu" \
  --adapter-pred "$RUN_DIR/lora_qwen.conllu" \
  --run-name "$RUN_ID" \
  --model "$MODEL" \
  --adapter-path "$ADAPTER_PATH"
