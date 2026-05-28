#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

from prompting import SYSTEM_PROMPT, wrap_user_conllu


def format_feats(feats):
    if not feats:
        return "_"
    return "|".join(f"{key}={feats[key]}" for key in sorted(feats))


def value_or_underscore(value):
    if value is None or value == "":
        return "_"
    return str(value)


def expected_output_to_conllu(expected_output):
    lines = [
        f"# sent_id = {value_or_underscore(expected_output.get('sent_id'))}",
        f"# text = {expected_output['text']}",
    ]

    for token in expected_output["tokens"]:
        columns = [
            value_or_underscore(token.get("id")),
            value_or_underscore(token.get("form")),
            value_or_underscore(token.get("lemma")),
            value_or_underscore(token.get("upos")),
            value_or_underscore(token.get("xpos")),
            format_feats(token.get("feats")),
            value_or_underscore(token.get("head")),
            value_or_underscore(token.get("deprel")),
            "_",
            "_",
        ]
        lines.append("\t".join(columns))

    return "\n".join(lines)


def canonical_to_chat(example):
    if example["input"].get("format") == "conllu":
        user_content = wrap_user_conllu(example["input"]["conllu"])
        assistant_content = example["expected_output"]["conllu"]
    else:
        user_content = example["input"]["text"]
        assistant_content = expected_output_to_conllu(example["expected_output"])

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def write_jsonl(path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Split canonical Latin dependency JSONL into MLX-LM chat JSONL."
    )
    parser.add_argument("input", type=Path, help="Canonical processed JSONL file")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("latin_lora_data"),
        help="Output folder for train.jsonl, valid.jsonl, and test.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap for a tiny first experiment.",
    )
    args = parser.parse_args()

    examples = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rng = random.Random(args.seed)
    rng.shuffle(examples)

    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    chat_examples = [canonical_to_chat(example) for example in examples]

    total = len(chat_examples)
    test_count = max(1, round(total * args.test_ratio)) if total >= 3 else 0
    valid_count = max(1, round(total * args.valid_ratio)) if total >= 3 else 0
    train_count = total - valid_count - test_count

    train = chat_examples[:train_count]
    valid = chat_examples[train_count : train_count + valid_count]
    test = chat_examples[train_count + valid_count :]

    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "valid.jsonl", valid)
    write_jsonl(args.out_dir / "test.jsonl", test)

    print(f"Read {total} examples from {args.input}")
    print(f"Wrote {len(train)} train examples to {args.out_dir / 'train.jsonl'}")
    print(f"Wrote {len(valid)} valid examples to {args.out_dir / 'valid.jsonl'}")
    print(f"Wrote {len(test)} test examples to {args.out_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()
