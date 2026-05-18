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
  runs/prose_latin-evalatin24-240520.conllu

python3 scripts/run_udpipe_api.py \
  data/test/EvaLatin_2024_Syntactic_Parsing_test_data/EvaLatin_2024_poetry_test_data.conllu \
  runs/poetry_latin-evalatin24-240520.conllu
```

## Evaluate

```sh
python3 scripts/conll18_ud_eval.py -v \
  data/gold/EvaLatin_2024_Syntactic_Parsing_test_data_gold/EvaLatin_2024_prose_gold.conllu \
  runs/prose_latin-evalatin24-240520.conllu

python3 scripts/conll18_ud_eval.py -v \
  data/gold/EvaLatin_2024_Syntactic_Parsing_test_data_gold/EvaLatin_2024_poetry_gold.conllu \
  runs/poetry_latin-evalatin24-240520.conllu
```

Verified local scores for `latin-evalatin24-240520`:

| Split | UAS | LAS |
| --- | ---: | ---: |
| Prose | 80.49 | 75.20 |
| Poetry | 78.31 | 72.36 |

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
