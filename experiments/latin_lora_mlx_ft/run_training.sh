#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

MODEL="${MODEL:-mlx-community/Qwen3.5-0.8B-MLX-4bit}"
DATA_DIR="${DATA_DIR:-latin_lora_data_macsafe_2048}"
ADAPTER_PATH="${ADAPTER_PATH:-adapters/latin-qwen-copycols-macsafe-2048-001}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
ITERS="${ITERS:-200}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-4}"
NUM_LAYERS="${NUM_LAYERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
STEPS_PER_REPORT="${STEPS_PER_REPORT:-10}"
STEPS_PER_EVAL="${STEPS_PER_EVAL:-50}"
VAL_BATCHES="${VAL_BATCHES:-10}"
TEST_BATCHES="${TEST_BATCHES:-10}"
SAVE_EVERY="${SAVE_EVERY:-100}"

if [ ! -x .venv/bin/mlx_lm.lora ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install "mlx-lm[train]"
fi

.venv/bin/mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --test \
  --data "$DATA_DIR" \
  --fine-tune-type lora \
  --mask-prompt \
  --iters "$ITERS" \
  --batch-size "$BATCH_SIZE" \
  --grad-accumulation-steps "$GRAD_ACCUMULATION_STEPS" \
  --num-layers "$NUM_LAYERS" \
  --learning-rate "$LEARNING_RATE" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --steps-per-report "$STEPS_PER_REPORT" \
  --steps-per-eval "$STEPS_PER_EVAL" \
  --val-batches "$VAL_BATCHES" \
  --test-batches "$TEST_BATCHES" \
  --save-every "$SAVE_EVERY" \
  --grad-checkpoint \
  --adapter-path "$ADAPTER_PATH"
