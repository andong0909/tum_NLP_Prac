# Latin LoRA MLX Fine-Tuning

Local MLX-LM LoRA experiment for teaching a small model to complete Latin
dependency parses in CoNLL-U format.

## Setup

```sh
cd /Users/antoniii/Desktop/tum_NLP_Prac/experiments/latin_lora_mlx_ft
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Pipeline

The project keeps three layers:

- `../../data/`: original EvaLatin 2024 files, managed by the parent repo.
- `processed_data/`: canonical JSONL pairing original input CoNLL-U with gold CoNLL-U.
- `latin_lora_data/`: MLX-LM chat JSONL where the user message is original input CoNLL-U and the assistant output is gold CoNLL-U.

Build the canonical paired JSONL:

```sh
python3 scripts/convert_paired_conllu_to_dependency_jsonl.py \
  --pair \
  ../../data/test/EvaLatin_2024_Syntactic_Parsing_test_data/EvaLatin_2024_prose-test-data.conllu \
  ../../data/gold/EvaLatin_2024_Syntactic_Parsing_test_data_gold/EvaLatin_2024_prose_gold.conllu \
  --pair \
  ../../data/test/EvaLatin_2024_Syntactic_Parsing_test_data/EvaLatin_2024_poetry_test_data.conllu \
  ../../data/gold/EvaLatin_2024_Syntactic_Parsing_test_data_gold/EvaLatin_2024_poetry_gold.conllu \
  --out-file processed_data/evalatin24_paired_conllu_examples.jsonl
```

Build MLX-LM training splits:

```sh
python3 scripts/split_dependency_jsonl_for_mlx.py \
  processed_data/evalatin24_paired_conllu_examples.jsonl \
  --out-dir latin_lora_data
```

For local MacBook training, build a length-filtered split so MLX-LM does not
truncate long CoNLL-U tables or run out of Metal memory:

```sh
python3 scripts/split_dependency_jsonl_for_mlx.py \
  processed_data/evalatin24_paired_conllu_examples.jsonl \
  --out-dir latin_lora_data_macsafe_2048 \
  --max-seq-tokens 2048
```

This keeps only examples whose rendered chat sequence fits within the selected
token cap. Dropped examples are reported in:

```text
latin_lora_data_macsafe_2048/dropped_too_long.jsonl
```

The default training format uses a concise system prompt and passes the input
CoNLL-U block directly as the user message. To try a custom prompt:

```sh
python3 scripts/split_dependency_jsonl_for_mlx.py \
  processed_data/evalatin24_paired_conllu_examples.jsonl \
  --out-dir latin_lora_data_custom \
  --system-prompt-file prompts/my_prompt.txt
```

To wrap the input block in explicit tags:

```sh
python3 scripts/split_dependency_jsonl_for_mlx.py \
  processed_data/evalatin24_paired_conllu_examples.jsonl \
  --out-dir latin_lora_data_wrapped \
  --wrap-input
```

Validate the training files:

```sh
python3 scripts/validate_mlx_conllu_jsonl.py \
  latin_lora_data_macsafe_2048/train.jsonl \
  latin_lora_data_macsafe_2048/valid.jsonl \
  latin_lora_data_macsafe_2048/test.jsonl
```

## Train

Run this from a normal macOS Terminal, not from a sandboxed/headless session:

```sh
./run_training.sh
```

The default run uses Qwen3.5 4-bit MLX, the Mac-safe 2048-token data split,
200 iterations, and saves
the adapter to:

```text
adapters/latin-qwen-copycols-macsafe-2048-001
```

To train on the full unfiltered split or try a different cap, override the
environment variables:

```sh
DATA_DIR=latin_lora_data MAX_SEQ_LENGTH=2048 ./run_training.sh
```

## CUDA Training

MLX is Apple-only. Remote GPU cluster training with Hugging Face/PEFT now lives
in the sibling experiment folder:

```text
../latin_lora_hf_ft/
```

## Evaluate Base vs LoRA

After training finishes, compare the original Qwen model against the LoRA-adapted
model on the held-out `test.jsonl` split:

```sh
./run_evaluation.sh qwen-conllu-eval-001
```

For early debugging, run a small smoke evaluation first:

```sh
LIMIT=5 MAX_TOKENS=768 ./run_evaluation.sh qwen-conllu-smoke-005
```

This creates:

```text
runs/qwen-conllu-eval-001/
  gold.conllu
  base_qwen.conllu
  lora_qwen.conllu
  base_qwen.score.txt
  lora_qwen.score.txt
  metadata.json
  summary.csv
  summary.md
```

The scorer is `scripts/conll18_ud_eval.py`, the same style of evaluation used
for UDPipe baselines. The key metrics to report are `UPOS`, `UAS`, `LAS`,
`CLAS`, `MLAS`, and `BLEX`.

The run also writes `format_report.txt`. If predictions are malformed, the
summary will show `ERROR` instead of fake zero scores.

## Evaluate LLM Arc JSON

For generic LLMs, a more stable parser-style interface is to ask the model for
only dependency arcs:

```json
[{"id":1,"head":3,"deprel":"nsubj"},{"id":2,"head":3,"deprel":"obj"},{"id":3,"head":0,"deprel":"root"}]
```

The renderer copies the fixed EvaLatin input token columns and inserts only the
model's predicted `HEAD` and `DEPREL`. This tests dependency prediction without
also asking the model to regenerate all 10 CoNLL-U columns.

Strict raw smoke test:

```sh
LIMIT=5 \
MODEL="mlx-community/Qwen3.5-0.8B-MLX-4bit" \
SYSTEM_NAME="qwen35_arc_json" \
./run_arc_json_validation.sh qwen35-arc-json-smoke-005
```

Gemma smoke test:

```sh
LIMIT=5 \
MODEL="mlx-community/gemma-3-1b-it-4bit" \
SYSTEM_NAME="gemma3_1b_arc_json" \
CHAT_TEMPLATE_CONFIG='{}' \
./run_arc_json_validation.sh gemma3-1b-arc-json-smoke-005
```

The run writes:

```text
runs/<run_id>/
  gold.conllu
  <system>.arcs.jsonl
  <system>.conllu
  <system>.arc_report.json
  <system>.score.txt
  format_report.txt
  summary.md
```

By default, missing or invalid arcs are left invalid, so the official scorer may
return `ERROR`. To reproduce a benchmark-style scoreable diagnostic with a
reported fallback rate, add:

```sh
FALLBACK_ROOT=1
```

Do not mix strict scores and fallback scores in the same table without labeling
them separately.

## Training Format

Each MLX-LM line is a chat example:

- system: strict CoNLL-U instruction
- user: original input CoNLL-U with missing HEAD/DEPREL columns
- assistant: gold completed CoNLL-U with `# sent_id`, `# text`, and 10-column token rows

Default system prompt:

```text
Complete Latin CoNLL-U. Copy every input line exactly except replace HEAD and DEPREL. Return only valid 10-column CoNLL-U.
```
