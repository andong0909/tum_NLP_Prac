#!/usr/bin/env python3
import argparse
import json
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


def parse_token_count(conllu):
    for line in conllu.splitlines():
        if line.startswith("# token_count"):
            _, value = line.split("=", 1)
            return int(value.strip())
    return None


def parse_predictions(text):
    predictions = []
    saw_end = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "END":
            saw_end = True
            break
        columns = line.split("\t")
        if len(columns) != 3:
            columns = line.split()
        if len(columns) != 3:
            raise ValueError(f"Expected ID HEAD DEPREL or END, got: {raw_line!r}")
        token_id, head, deprel = columns
        if not token_id.isdigit():
            raise ValueError(f"Cannot parse token ID: {token_id!r}")
        if not head.isdigit():
            raise ValueError(f"Cannot parse HEAD: {head!r}")
        if not deprel or deprel == "_":
            raise ValueError(f"Cannot parse DEPREL: {deprel!r}")
        predictions.append((token_id, head, deprel))
    if not saw_end:
        raise ValueError("Missing END marker")
    return predictions


def render_one(input_conllu, pred_text):
    predictions = parse_predictions(pred_text)
    lines = input_conllu.splitlines()
    rows = conllu_token_rows(input_conllu)
    token_count = parse_token_count(input_conllu)
    if token_count is not None and token_count != len(rows):
        raise ValueError(f"# token_count={token_count} does not match {len(rows)} syntactic token rows")

    expected_ids = [columns[0] for _, columns in rows]
    predicted_ids = [token_id for token_id, _, _ in predictions]
    if predicted_ids != expected_ids:
        raise ValueError(f"Predicted token IDs do not match input order: expected {expected_ids}, got {predicted_ids}")

    for (line_index, columns), (_, head, deprel) in zip(rows, predictions):
        columns[6] = head
        columns[7] = deprel
        lines[line_index] = "\t".join(columns)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Render sentence-level ID/HEAD/DEPREL+END predictions back into CoNLL-U."
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

    blocks = []
    errors = []
    for index, (input_record, prediction_record) in enumerate(zip(input_records, prediction_records), 1):
        input_conllu = input_record["messages"][1]["content"]
        pred_text = prediction_record.get("assistant", prediction_record.get("prediction", ""))
        try:
            blocks.append(render_one(input_conllu, pred_text))
        except Exception as exc:
            sent_id = next(
                (
                    line.split("=", 1)[1].strip()
                    for line in input_conllu.splitlines()
                    if line.startswith("# sent_id")
                ),
                str(index),
            )
            errors.append({"index": index, "sent_id": sent_id, "error": str(exc), "prediction": pred_text})

    if errors:
        raise ValueError(json.dumps({"errors": errors[:20], "error_count": len(errors)}, ensure_ascii=False, indent=2))

    args.output_conllu.parent.mkdir(parents=True, exist_ok=True)
    args.output_conllu.write_text("\n\n".join(blocks) + "\n\n", encoding="utf-8")
    if args.report:
        args.report.write_text(
            json.dumps(
                {
                    "predictions": len(input_records),
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
