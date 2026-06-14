#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_ID="${1:-hf-token-head-deprel-eval-001}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH-hf_outputs/qwen25-05b-token-head-deprel-a100-lora-full}"
SYSTEM_NAME="${SYSTEM_NAME:-hf_lora_token_head_deprel}"
TEST_FILE="${TEST_FILE:-latin_token_head_deprel_data/test.jsonl}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
LIMIT_SENTENCES="${LIMIT_SENTENCES:-}"
PRECISION="${PRECISION:-bf16}"
RUN_DIR="runs/${RUN_ID}"
EVAL_INPUT="$TEST_FILE"

mkdir -p "$RUN_DIR"

if [ ! -d .venv-hf ]; then
  python3 -m venv .venv-hf
fi
source .venv-hf/bin/activate

PRECISION_ARGS=""
if [ "$PRECISION" = "bf16" ]; then
  PRECISION_ARGS="--bf16"
elif [ "$PRECISION" = "fp16" ]; then
  PRECISION_ARGS="--fp16"
elif [ "$PRECISION" != "fp32" ]; then
  echo "Unsupported PRECISION=$PRECISION; use fp16, bf16, or fp32" >&2
  exit 2
fi

if [ -n "$LIMIT_SENTENCES" ]; then
  EVAL_INPUT="$RUN_DIR/input_token_subset.jsonl"
  python - "$TEST_FILE" "$EVAL_INPUT" "$LIMIT_SENTENCES" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
limit = int(sys.argv[3])
seen = []
with src.open(encoding="utf-8") as handle, dst.open("w", encoding="utf-8") as out:
    for line in handle:
        if not line.strip():
            continue
        record = json.loads(line)
        sent_id = record.get("sent_id", "")
        if sent_id not in seen:
            if len(seen) >= limit:
                break
            seen.append(sent_id)
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"Wrote {len(seen)} complete sentences to {dst}")
PY
fi

python - "$EVAL_INPUT" "$RUN_DIR/gold_token_predictions.jsonl" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
with src.open(encoding="utf-8") as handle, dst.open("w", encoding="utf-8") as out:
    for line in handle:
        if not line.strip():
            continue
        record = json.loads(line)
        out.write(
            json.dumps(
                {"sent_id": record.get("sent_id", ""), "assistant": record["messages"][2]["content"]},
                ensure_ascii=False,
            )
            + "\n"
        )
PY

python scripts/render_token_head_deprel_to_conllu.py \
  --input-jsonl "$EVAL_INPUT" \
  --pred-jsonl "$RUN_DIR/gold_token_predictions.jsonl" \
  --output-conllu "$RUN_DIR/gold.conllu" \
  --report "$RUN_DIR/gold_render_report.json"

if [ -n "$ADAPTER_PATH" ]; then
  python scripts/generate_hf_head_deprel_predictions.py \
    --input "$EVAL_INPUT" \
    --output "$RUN_DIR/head_deprel_token_predictions.jsonl" \
    --model "$MODEL" \
    --adapter-path "$ADAPTER_PATH" \
    --max-input-length "$MAX_INPUT_LENGTH" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    $PRECISION_ARGS
else
  python scripts/generate_hf_head_deprel_predictions.py \
    --input "$EVAL_INPUT" \
    --output "$RUN_DIR/head_deprel_token_predictions.jsonl" \
    --model "$MODEL" \
    --max-input-length "$MAX_INPUT_LENGTH" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    $PRECISION_ARGS
fi

if ! python scripts/render_token_head_deprel_to_conllu.py \
    --input-jsonl "$EVAL_INPUT" \
    --pred-jsonl "$RUN_DIR/head_deprel_token_predictions.jsonl" \
    --output-conllu "$RUN_DIR/pred.conllu" \
    --report "$RUN_DIR/render_report.json" \
    2> "$RUN_DIR/render_error.txt"; then
  {
    echo "# CoNLL-U Evaluation Summary"
    echo
    echo "| System | UPOS | UAS | LAS | CLAS | MLAS | BLEX |"
    echo "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    echo "| $SYSTEM_NAME | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR |"
    echo
    echo "## Render Error"
    echo
    echo '```text'
    cat "$RUN_DIR/render_error.txt"
    echo '```'
  } > "$RUN_DIR/summary.md"
  cat "$RUN_DIR/summary.md"
  exit 0
fi

python ../latin_lora_mlx_ft/scripts/report_conllu_prediction_quality.py \
  "$RUN_DIR/pred.conllu" \
  > "$RUN_DIR/format_report.txt"

python ../latin_lora_mlx_ft/scripts/score_single_conllu_run.py \
  --gold "$RUN_DIR/gold.conllu" \
  --pred "$RUN_DIR/pred.conllu" \
  --system-name "$SYSTEM_NAME" \
  --scorer ../latin_lora_mlx_ft/scripts/conll18_ud_eval.py \
  --run-name "$RUN_ID" \
  --out-root runs \
  --model "$MODEL" \
  --adapter-path "$ADAPTER_PATH"
