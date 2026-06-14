#!/usr/bin/env python3
import argparse
import json
from collections import OrderedDict
from pathlib import Path


def load_jsonl(path, limit=None):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def conllu_token_rows(conllu):
    rows = []
    for index, line in enumerate(conllu.splitlines()):
        if not line or line.startswith("#"):
            continue
        columns = line.split("\t")
        if columns and "-" not in columns[0] and "." not in columns[0]:
            rows.append((index, columns))
    return rows


def parse_prediction(text):
    nonempty = [line.strip() for line in text.splitlines() if line.strip()]
    if len(nonempty) != 1:
        raise ValueError(f"Expected exactly one non-empty output line, got {len(nonempty)}")
    columns = nonempty[0].split("\t")
    if len(columns) != 3:
        columns = nonempty[0].split()
    if len(columns) != 3:
        raise ValueError(f"Expected ID HEAD DEPREL, got: {nonempty[0]!r}")
    token_id, head, deprel = columns
    if not token_id.isdigit():
        raise ValueError(f"Cannot parse token ID: {token_id!r}")
    if not head.isdigit():
        raise ValueError(f"Cannot parse HEAD: {head!r}")
    if not deprel or deprel == "_":
        raise ValueError(f"Cannot parse DEPREL: {deprel!r}")
    return token_id, head, deprel


def render_sentences(input_records, prediction_records):
    by_sentence = OrderedDict()
    for record in input_records:
        key = record.get("sent_id") or str(len(by_sentence))
        if key not in by_sentence:
            by_sentence[key] = {"conllu": record["input_conllu"], "predictions": {}}

    errors = []
    for record, prediction in zip(input_records, prediction_records):
        key = record.get("sent_id") or ""
        expected_token_id = str(record.get("token_id", ""))
        pred_text = prediction.get("assistant", prediction.get("prediction", ""))
        try:
            token_id, head, deprel = parse_prediction(pred_text)
            if expected_token_id and token_id != expected_token_id:
                raise ValueError(f"Expected token ID {expected_token_id}, got {token_id}")
            by_sentence[key]["predictions"][token_id] = (head, deprel)
        except Exception as exc:
            errors.append(
                {
                    "sent_id": key,
                    "token_id": expected_token_id,
                    "error": str(exc),
                    "prediction": pred_text,
                }
            )

    blocks = []
    for key, payload in by_sentence.items():
        lines = payload["conllu"].splitlines()
        rows = conllu_token_rows(payload["conllu"])
        predictions = payload["predictions"]
        missing = []
        for line_index, columns in rows:
            token_id = columns[0]
            if token_id not in predictions:
                missing.append(token_id)
                continue
            head, deprel = predictions[token_id]
            columns[6] = head
            columns[7] = deprel
            lines[line_index] = "\t".join(columns)
        if missing:
            errors.append({"sent_id": key, "missing_token_ids": missing})
        blocks.append("\n".join(lines))
    return blocks, errors


def main():
    parser = argparse.ArgumentParser(
        description="Render one-example-per-token-row ID/HEAD/DEPREL predictions back into CoNLL-U."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--pred-jsonl", type=Path, required=True)
    parser.add_argument("--output-conllu", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    input_records = load_jsonl(args.input_jsonl, limit=args.limit)
    prediction_records = load_jsonl(args.pred_jsonl, limit=args.limit)
    if len(input_records) != len(prediction_records):
        raise ValueError(
            f"Input/prediction count mismatch: {len(input_records)} input records, "
            f"{len(prediction_records)} predictions"
        )

    blocks, errors = render_sentences(input_records, prediction_records)
    if errors:
        raise ValueError(json.dumps({"errors": errors[:20], "error_count": len(errors)}, ensure_ascii=False, indent=2))

    args.output_conllu.parent.mkdir(parents=True, exist_ok=True)
    args.output_conllu.write_text("\n\n".join(blocks) + "\n\n", encoding="utf-8")
    if args.report:
        args.report.write_text(
            json.dumps(
                {
                    "row_predictions": len(input_records),
                    "sentences": len(blocks),
                    "errors": 0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"Wrote {len(blocks)} rendered CoNLL-U sentences to {args.output_conllu}")


if __name__ == "__main__":
    main()
