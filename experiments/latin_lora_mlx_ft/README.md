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

Validate the training files:

```sh
python3 scripts/validate_mlx_conllu_jsonl.py \
  latin_lora_data/train.jsonl \
  latin_lora_data/valid.jsonl \
  latin_lora_data/test.jsonl
```

## Train

Run this from a normal macOS Terminal, not from a sandboxed/headless session:

```sh
./run_training.sh
```

The first run uses `mlx-community/Qwen3-0.6B-4bit`, 200 iterations, and saves
the adapter to:

```text
adapters/latin-qwen-copycols-prompt-001
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

## Training Format

Each MLX-LM line is a chat example:

- system: strict CoNLL-U instruction
- user: original input CoNLL-U with missing HEAD/DEPREL columns
- assistant: gold completed CoNLL-U with `# sent_id`, `# text`, and 10-column token rows
