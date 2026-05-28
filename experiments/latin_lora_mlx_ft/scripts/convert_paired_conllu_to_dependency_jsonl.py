#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read_blocks(path):
    blocks = {}
    current = []

    def flush():
        nonlocal current
        if not current:
            return
        sent_id = None
        text = None
        for line in current:
            if line.startswith("# sent_id = "):
                sent_id = line[len("# sent_id = ") :].strip()
            elif line.startswith("# text = "):
                text = line[len("# text = ") :].strip()
        if sent_id is None:
            raise ValueError(f"{path}: sentence block without # sent_id")
        blocks[sent_id] = {
            "sent_id": sent_id,
            "text": text,
            "conllu": "\n".join(current).strip(),
        }
        current = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip():
            current.append(raw_line.rstrip())
        else:
            flush()
    flush()
    return blocks


def convert_pair(input_path, gold_path):
    input_blocks = read_blocks(input_path)
    gold_blocks = read_blocks(gold_path)

    missing_gold = sorted(set(input_blocks) - set(gold_blocks))
    missing_input = sorted(set(gold_blocks) - set(input_blocks))
    if missing_gold or missing_input:
        raise ValueError(
            f"Unpaired sent_ids for {input_path.name}/{gold_path.name}: "
            f"missing_gold={missing_gold[:5]}, missing_input={missing_input[:5]}"
        )

    examples = []
    for sent_id in input_blocks:
        input_block = input_blocks[sent_id]
        gold_block = gold_blocks[sent_id]
        examples.append(
            {
                "task": "latin_dependency_parsing_conllu",
                "source_input": input_path.name,
                "source_gold": gold_path.name,
                "sent_id": sent_id,
                "text": input_block["text"],
                "input": {
                    "format": "conllu",
                    "conllu": input_block["conllu"],
                },
                "expected_output": {
                    "format": "conllu",
                    "conllu": gold_block["conllu"],
                },
            }
        )
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Pair original EvaLatin input CoNLL-U blocks with gold CoNLL-U targets."
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("INPUT_CONLLU", "GOLD_CONLLU"),
        required=True,
        help="Input/gold CoNLL-U file pair. Can be repeated.",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("processed_data/evalatin24_paired_conllu_examples.jsonl"),
    )
    args = parser.parse_args()

    examples = []
    for input_file, gold_file in args.pair:
        examples.extend(convert_pair(Path(input_file), Path(gold_file)))

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} paired CoNLL-U examples to {args.out_file}")


if __name__ == "__main__":
    main()
