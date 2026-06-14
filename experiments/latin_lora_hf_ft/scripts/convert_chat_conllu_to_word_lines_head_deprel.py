#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a Latin dependency parser. Given token lines in the format "
    "ID<TAB>FORM, output exactly one dependency line for each input token, "
    "in the same order. Return only ID<TAB>HEAD<TAB>DEPREL lines. Do not output "
    "CoNLL-U columns, markdown, comments, or explanations."
)


def conllu_comments(conllu):
    comments = {}
    for line in conllu.splitlines():
        if line.startswith("# sent_id = "):
            comments["sent_id"] = line.removeprefix("# sent_id = ").strip()
        elif line.startswith("# text = "):
            comments["text"] = line.removeprefix("# text = ").strip()
    return comments


def conllu_token_rows(conllu):
    rows = []
    for line in conllu.splitlines():
        if not line or line.startswith("#"):
            continue
        columns = line.split("\t")
        if not columns or "-" in columns[0] or "." in columns[0]:
            continue
        if len(columns) != 10:
            raise ValueError(f"Expected 10 CoNLL-U columns, got {len(columns)}: {line}")
        rows.append(columns)
    return rows


def word_lines(rows):
    return "\n".join(f"{columns[0]}\t{columns[1]}" for columns in rows)


def id_head_deprel_target(rows):
    return "\n".join(f"{columns[0]}\t{columns[6]}\t{columns[7]}" for columns in rows)


def user_prompt(user_conllu):
    comments = conllu_comments(user_conllu)
    rows = conllu_token_rows(user_conllu)
    parts = []
    if comments.get("sent_id"):
        parts.append(f"# sent_id = {comments['sent_id']}")
    if comments.get("text"):
        parts.append(f"# text = {comments['text']}")
    parts.append(word_lines(rows))
    return "\n".join(parts)


def convert_record(record, system_prompt):
    messages = record.get("messages", [])
    if len(messages) != 3:
        raise ValueError("Expected chat record with exactly three messages")

    user_conllu = messages[1]["content"].strip()
    gold_conllu = messages[2]["content"].strip()
    user_rows = conllu_token_rows(user_conllu)
    gold_rows = conllu_token_rows(gold_conllu)
    if len(user_rows) != len(gold_rows):
        raise ValueError(
            f"Input/gold token count mismatch: {len(user_rows)} input rows, {len(gold_rows)} gold rows"
        )
    for user_columns, gold_columns in zip(user_rows, gold_rows):
        if user_columns[0] != gold_columns[0] or user_columns[1] != gold_columns[1]:
            raise ValueError(
                f"Input/gold token mismatch: {user_columns[0]} {user_columns[1]} vs "
                f"{gold_columns[0]} {gold_columns[1]}"
            )

    comments = conllu_comments(gold_conllu)
    return {
        "sent_id": comments.get("sent_id", ""),
        "text": comments.get("text", ""),
        "input_conllu": user_conllu,
        "gold_conllu": gold_conllu,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt(user_conllu)},
            {"role": "assistant", "content": id_head_deprel_target(gold_rows)},
        ],
    }


def convert_file(input_path, output_path, system_prompt):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                converted = convert_record(json.loads(line), system_prompt)
            except Exception as exc:
                raise ValueError(f"{input_path}:{line_number}: {exc}") from exc
            dst.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Convert full CoNLL-U chat JSONL into compact word-line ID/HEAD/DEPREL chat JSONL."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    args = parser.parse_args()

    total = 0
    for split in ["train", "valid", "test"]:
        count = convert_file(
            args.input_dir / f"{split}.jsonl",
            args.out_dir / f"{split}.jsonl",
            args.system_prompt,
        )
        total += count
        print(f"Wrote {count} {split} examples to {args.out_dir / f'{split}.jsonl'}")
    print(f"Total: {total} examples")


if __name__ == "__main__":
    main()
