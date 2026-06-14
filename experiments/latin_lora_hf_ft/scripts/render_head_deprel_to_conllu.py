#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_head_deprel_lines(text):
    pairs = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) != 2:
            raise ValueError(f"Expected 2 tab-separated columns, got {len(columns)}: {line!r}")
        head, deprel = columns
        int(head)
        if not deprel or deprel == "_":
            raise ValueError(f"Invalid DEPREL: {line!r}")
        pairs.append((head, deprel))
    return pairs


def render_one(input_conllu, head_deprel_text):
    pairs = parse_head_deprel_lines(head_deprel_text)
    pair_index = 0
    rendered = []

    for line in input_conllu.strip().splitlines():
        if line.startswith("#") or not line:
            rendered.append(line)
            continue

        columns = line.split("\t")
        while len(columns) < 10:
            columns.append("_")
        columns = columns[:10]

        token_id = columns[0]
        if "-" in token_id or "." in token_id:
            rendered.append("\t".join(columns))
            continue

        if pair_index >= len(pairs):
            raise ValueError("Prediction has fewer HEAD/DEPREL lines than input tokens")
        columns[6], columns[7] = pairs[pair_index]
        pair_index += 1
        rendered.append("\t".join(columns))

    if pair_index != len(pairs):
        raise ValueError("Prediction has more HEAD/DEPREL lines than input tokens")

    return "\n".join(rendered)


def main():
    parser = argparse.ArgumentParser(
        description="Render HEAD/DEPREL-only predictions into scoreable CoNLL-U."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument(
        "--pred-jsonl",
        type=Path,
        required=True,
        help="JSONL records with an assistant/prediction field containing HEAD<TAB>DEPREL lines.",
    )
    parser.add_argument("--output-conllu", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    inputs = [
        json.loads(line)
        for line in args.input_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    preds = [
        json.loads(line)
        for line in args.pred_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        inputs = inputs[: args.limit]
        preds = preds[: args.limit]
    if len(inputs) != len(preds):
        raise ValueError(f"Input/prediction count mismatch: {len(inputs)} != {len(preds)}")

    blocks = []
    report = []
    for index, (input_record, pred_record) in enumerate(zip(inputs, preds), 1):
        input_conllu = input_record["messages"][1]["content"]
        pred_text = pred_record.get("assistant") or pred_record.get("prediction") or pred_record.get("text")
        if pred_text is None and "messages" in pred_record:
            pred_text = pred_record["messages"][-1]["content"]
        if pred_text is None:
            raise ValueError(f"Prediction record {index} has no assistant/prediction/text field")
        try:
            blocks.append(render_one(input_conllu, pred_text))
            report.append({"index": index, "ok": True, "error": None})
        except Exception as exc:
            report.append({"index": index, "ok": False, "error": str(exc)})
            raise

    args.output_conllu.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output_conllu.write_text("\n\n".join(blocks) + "\n\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(blocks)} rendered CoNLL-U sentences to {args.output_conllu}")


if __name__ == "__main__":
    main()
