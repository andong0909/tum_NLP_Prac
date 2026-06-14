#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def sentence_blocks(text):
    return [block for block in text.strip().split("\n\n") if block.strip()]


def parse_sent_id(block):
    for line in block.splitlines():
        if line.startswith("# sent_id"):
            return line.split("=", 1)[1].strip()
    return None


def token_rows(block):
    return [
        line.split("\t")
        for line in block.splitlines()
        if line and not line.startswith("#") and "-" not in line.split("\t", 1)[0]
    ]


def is_int(value):
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def root_token_id(rows):
    for columns in rows:
        if len(columns) >= 4 and columns[3] in {"VERB", "AUX"}:
            return columns[0]
    return rows[0][0] if rows else "0"


def fallback_head_deprel(columns, root_id):
    token_id = columns[0]
    upos = columns[3] if len(columns) > 3 else "_"
    if token_id == root_id:
        return "0", "root"
    if upos in {"PUNCT"}:
        return root_id, "punct"
    if upos in {"CCONJ"}:
        return root_id, "cc"
    if upos in {"SCONJ"}:
        return root_id, "mark"
    return root_id, "dep"


def prediction_map(block):
    by_id = {}
    for columns in token_rows(block):
        if columns:
            by_id[columns[0]] = columns
    return by_id


def choose_prediction_block(input_sent_id, prediction_blocks, fallback_index):
    for block in prediction_blocks:
        if parse_sent_id(block) == input_sent_id:
            return block
    if fallback_index < len(prediction_blocks):
        return prediction_blocks[fallback_index]
    return ""


def repair_example(example, prediction_block):
    input_block = example["messages"][1]["content"].strip()
    input_lines = input_block.splitlines()
    rows = token_rows(input_block)
    root_id = root_token_id(rows)
    predicted = prediction_map(prediction_block)

    repaired = []
    for line in input_lines:
        if line.startswith("#") or not line.strip():
            repaired.append(line)
            continue

        columns = line.split("\t")
        if len(columns) < 10:
            columns = columns + ["_"] * (10 - len(columns))
        columns = columns[:10]

        pred_columns = predicted.get(columns[0], [])
        pred_head = pred_columns[6] if len(pred_columns) > 6 else "_"
        pred_deprel = pred_columns[7] if len(pred_columns) > 7 else "_"

        if is_int(pred_head) and pred_deprel and pred_deprel != "_":
            columns[6] = pred_head
            columns[7] = pred_deprel
        else:
            columns[6], columns[7] = fallback_head_deprel(columns, root_id)

        columns[8] = columns[8] or "_"
        columns[9] = columns[9] or "_"
        repaired.append("\t".join(columns))

    return "\n".join(repaired)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Repair messy model CoNLL-U predictions by aligning them to the original "
            "input JSONL. This produces a scoreable diagnostic baseline."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    examples = [
        json.loads(line)
        for line in args.input_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        examples = examples[: args.limit]

    prediction_blocks = sentence_blocks(
        args.prediction.read_text(encoding="utf-8", errors="replace")
    )

    repaired_blocks = []
    for index, example in enumerate(examples):
        input_sent_id = parse_sent_id(example["messages"][1]["content"])
        prediction_block = choose_prediction_block(input_sent_id, prediction_blocks, index)
        repaired_blocks.append(repair_example(example, prediction_block))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(repaired_blocks) + "\n\n", encoding="utf-8")
    print(
        f"Repaired {len(repaired_blocks)} predictions from {args.prediction} "
        f"to {args.output}"
    )
    print(
        "Note: invalid or missing HEAD/DEPREL values were replaced with a simple "
        "root-attached fallback, so this is a diagnostic scoreable baseline."
    )


if __name__ == "__main__":
    main()
