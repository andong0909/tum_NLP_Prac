# LT4HALA 2024 EvaLatin Local Notes

This local folder is set up for the EvaLatin 2024 dependency parsing task.

## Repository Files Used

- Scorer: `2024/conll18_ud_eval.py`
- Test data ZIP: `2024/data_and_doc/EvaLatin_2024_Syntactic_Parsing_test_data.zip`
- Gold data ZIP: `2024/data_and_doc/EvaLatin_2024_Syntactic_Parsing_test_data_gold.zip`

## Download Data

```sh
mkdir -p data/test data/gold scripts runs
curl -L -o scripts/conll18_ud_eval.py \
  https://raw.githubusercontent.com/CIRCSE/LT4HALA/master/2024/conll18_ud_eval.py
curl -L -o data/EvaLatin_2024_Syntactic_Parsing_test_data.zip \
  https://github.com/CIRCSE/LT4HALA/raw/master/2024/data_and_doc/EvaLatin_2024_Syntactic_Parsing_test_data.zip
curl -L -o data/EvaLatin_2024_Syntactic_Parsing_test_data_gold.zip \
  https://github.com/CIRCSE/LT4HALA/raw/master/2024/data_and_doc/EvaLatin_2024_Syntactic_Parsing_test_data_gold.zip
unzip -o data/EvaLatin_2024_Syntactic_Parsing_test_data.zip -d data/test
unzip -o data/EvaLatin_2024_Syntactic_Parsing_test_data_gold.zip -d data/gold
```

## Run UDPipe Through The API

The important API settings are:

- endpoint: `https://lindat.mff.cuni.cz/services/udpipe/api/process`
- model: `latin-evalatin24-240520`
- input: `conllu`
- parser: empty value, meaning "run parser"

Using `input=conllu` preserves the EvaLatin tokenization and morphology and fills the missing HEAD/DEPREL columns.

```sh
python3 scripts/run_udpipe_api.py \
  data/test/EvaLatin_2024_Syntactic_Parsing_test_data/EvaLatin_2024_prose-test-data.conllu \
  runs/udpipe-evalatin24-baseline-20260519T012905+0200-01/prose_latin-evalatin24-240520.conllu

python3 scripts/run_udpipe_api.py \
  data/test/EvaLatin_2024_Syntactic_Parsing_test_data/EvaLatin_2024_poetry_test_data.conllu \
  runs/udpipe-evalatin24-baseline-20260519T012905+0200-01/poetry_latin-evalatin24-240520.conllu
```

## Evaluate

```sh
python3 scripts/conll18_ud_eval.py -v \
  data/gold/EvaLatin_2024_Syntactic_Parsing_test_data_gold/EvaLatin_2024_prose_gold.conllu \
  runs/udpipe-evalatin24-baseline-20260519T012905+0200-01/prose_latin-evalatin24-240520.conllu

python3 scripts/conll18_ud_eval.py -v \
  data/gold/EvaLatin_2024_Syntactic_Parsing_test_data_gold/EvaLatin_2024_poetry_gold.conllu \
  runs/udpipe-evalatin24-baseline-20260519T012905+0200-01/poetry_latin-evalatin24-240520.conllu
```

Verified local scores for `latin-evalatin24-240520`:

| Split | UAS | LAS |
| --- | ---: | ---: |
| Prose | 80.49 | 75.20 |
| Poetry | 78.31 | 72.36 |

## LLM LoRA Fine-Tuning Status

Two LLM fine-tuning tracks are now included under `experiments/`:

- `experiments/latin_lora_mlx_ft/`: Apple MLX experiments and data preparation.
- `experiments/latin_lora_hf_ft/`: Hugging Face/PEFT training for CUDA Slurm clusters.

For a chronological lab notebook of the LLM attempts, motivations, outcomes,
failure modes, and next steps, see
[`experiments/latin_lora_hf_ft/EXPERIMENT_LOG.md`](experiments/latin_lora_hf_ft/EXPERIMENT_LOG.md).

The latest Hugging Face protocol follows a parser-wrapper design rather than
asking the LLM to regenerate full CoNLL-U:

- input: original EvaLatin CoNLL-U with tokenization, lemmas, POS, morphology, and blank `HEAD`/`DEPREL`
- current target: one `ID<TAB>HEAD<TAB>DEPREL` line per syntactic token, in token order
- post-processing: render predicted dependency rows back into the original CoNLL-U skeleton, then score with `conll18_ud_eval.py`
- split: 80/10/10 over 854 paired examples
  - train: 684 examples
  - validation: 85 examples
  - test: 85 examples

The first CUDA cluster run successfully fine-tuned `Qwen/Qwen2.5-0.5B-Instruct`
with normal LoRA, bf16, and an A100 40 GB allocation on the original
`HEAD<TAB>DEPREL` target. That run trained stably but remained hard to score
because generation did not reliably produce exactly one line per token.

```text
experiments/latin_lora_hf_ft/hf_outputs/qwen25-05b-head-deprel-a100-lora-full/
```

Training finished cleanly after 3 epochs:

```text
eval_loss: 0.5038
train_loss: 0.7946
```

Evaluation is intentionally strict. The renderer requires exactly one
`HEAD<TAB>DEPREL` prediction per syntactic token. Current smoke tests show:

- base Qwen2.5 output is unscoreable when it predicts too few dependency lines
- the LoRA adapter is closer but still unscoreable in the current smoke run when it predicts extra dependency lines

This means the next review task is not training stability; it is inference
format control. Candidate fixes include reducing `MAX_NEW_TOKENS`, adding an
explicit stop sequence, or making the evaluation wrapper record line-count
validity separately before running the official scorer. Do not report repaired
or truncated predictions as raw model scores unless they are clearly labeled as
fallback-assisted.

The newer sentence-level `ID<TAB>HEAD<TAB>DEPREL` protocol is the current best
candidate. It keeps full sentence context in the prompt but anchors every
generated row with a token id.

For final reporting, we now prioritize the **full 85-sentence EvaLatin test
split**. Partial scores are useful diagnostics, but they are optimistic because
they exclude invalid predicted trees. Penalized full scores keep all 85
sentences by replacing invalid predicted trees with adversarial dummy valid
trees before running the official CoNLL-18 scorer.

Current full-test results:

| Model | Score Type | Scope | UPOS | UAS | LAS | CLAS | MLAS | BLEX |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B LoRA | Penalized full | 85/85, 8 dummy-replaced | 100.00 | 51.16 | 47.07 | 42.71 | 40.44 | 42.71 |
| Qwen2.5-3B LoRA | Penalized full | 85/85, 9 dummy-replaced | 100.00 | 62.11 | 58.03 | 54.13 | 51.83 | 54.13 |
| Qwen2.5-1.5B LoRA | Partial diagnostic | 77/85 tree-valid only | 100.00 | 64.16 | 59.04 | 56.30 | 53.31 | 56.30 |
| Qwen2.5-3B LoRA | Partial diagnostic | 76/85 tree-valid only | 100.00 | 70.34 | 65.72 | 63.44 | 60.74 | 63.44 |

The 3B model is strongest under both fairer penalized full scoring and the
older partial diagnostic view. The remaining blocker is invalid-tree reduction:
the model is accurate when it produces a valid tree, but invalid predictions
must count against the full score.

Generated adapters, run logs, and prediction outputs are intentionally ignored
by Git under:

```text
experiments/latin_lora_mlx_ft/adapters/
experiments/latin_lora_mlx_ft/runs/
experiments/latin_lora_hf_ft/hf_outputs/
experiments/latin_lora_hf_ft/runs/
```

## Other Latin Models To Try

List available models:

```sh
curl -s https://lindat.mff.cuni.cz/services/udpipe/api/models \
  | python3 -c "import sys,json; print('\n'.join(k for k in json.load(sys.stdin)['models'] if 'latin' in k.lower()))"
```

Then pass a model id:

```sh
python3 scripts/run_udpipe_api.py --model latin-circse-ud-2.17-251125 input.conllu output.conllu
```

## Experiment Tracking

Each experiment should live in its own run folder and include a `metadata.json` file. Use a readable but unique `run_id`, for example:

```text
udpipe-model-sweep-20260519T021543+0200-01
```

The run metadata should record:

- `run_id` and `run_timestamp`
- model names, model versions, and model configuration
- `training_data` used by each trained or fine-tuned model
- evaluation data and scorer used for the run
- metrics for each evaluated split: `UAS`, `LAS`, `CLAS`, `MLAS`, and `BLEX`
- output files and score files needed to inspect the run later

For a pretrained external model that was not trained locally, `training_data` should still say what is known about its training source and note that the local run only performed inference. For a model trained in this project, `training_data` should name the exact corpora, splits, preprocessing version, and any extra Latin text used before or during fine-tuning.

## UDPipe Latin Model Comparison

The six current Latin UD 2.17 UDPipe models were run on the EvaLatin 2024 prose and poetry test files, then scored against the released gold data with the same evaluator above.

| Rank | Model | Prose UAS | Prose LAS | Poetry UAS | Poetry LAS | Avg UAS | Avg LAS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | CIRCSE | 73.83 | 67.57 | 73.68 | 67.59 | 73.75 | 67.58 |
| 2 | UDante | 69.70 | 62.90 | 69.02 | 61.71 | 69.36 | 62.30 |
| 3 | Perseus | 69.33 | 62.43 | 68.98 | 61.19 | 69.16 | 61.81 |
| 4 | ITTB | 69.35 | 61.14 | 66.33 | 57.64 | 67.84 | 59.39 |
| 5 | PROIEL | 68.79 | 60.50 | 65.02 | 55.91 | 66.91 | 58.20 |
| 6 | LLCT | 45.42 | 31.32 | 36.09 | 21.05 | 40.76 | 26.19 |

`latin-circse-ud-2.17-251125` is the strongest of these six general Latin models on this test set. The EvaLatin-specific `latin-evalatin24-240520` model above remains a stronger baseline, with 75.20 LAS on prose and 72.36 LAS on poetry.

The comparison artifacts are saved in `runs/udpipe-model-sweep-20260519T021543+0200-01/`:

- `metadata.json` records the run id, model training-data provenance, and scorer metrics.
- `summary.md` and `summary.csv` contain the scored table.
- `latin_model_las_chart.svg` charts prose and poetry LAS.
- The generated CoNLL-U outputs and evaluator score files are kept for each tested model.
