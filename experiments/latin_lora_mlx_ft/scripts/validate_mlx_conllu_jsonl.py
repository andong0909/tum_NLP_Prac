#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def validate_conllu(content, path, line_number):
    rows = content.splitlines()
    if len(rows) < 3:
        raise ValueError(f"{path}:{line_number}: expected comments and token rows")
    if not any(row.startswith("# sent_id = ") for row in rows):
        raise ValueError(f"{path}:{line_number}: missing # sent_id comment")
    if not any(row.startswith("# text = ") for row in rows):
        raise ValueError(f"{path}:{line_number}: missing # text comment")

    root_count = 0
    for row in rows:
        if not row or row.startswith("#"):
            continue
        columns = row.split("\t")
        if len(columns) != 10:
            raise ValueError(
                f"{path}:{line_number}: token row has {len(columns)} columns"
            )
        if "-" in columns[0] or "." in columns[0]:
            continue
        head = int(columns[6])
        if head == 0:
            root_count += 1
    if root_count != 1:
        raise ValueError(f"{path}:{line_number}: expected one root, found {root_count}")


def validate_file(path):
    count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        example = json.loads(line)
        messages = example.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"{path}:{line_number}: expected exactly 3 messages")

        roles = [message.get("role") for message in messages]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"{path}:{line_number}: unexpected message roles {roles}")

        validate_conllu(messages[2].get("content", ""), path, line_number)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Validate MLX-LM chat JSONL files with CoNLL-U assistant outputs."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    total = 0
    for path in args.paths:
        count = validate_file(path)
        total += count
        print(f"{path}: {count} valid CoNLL-U chat examples")
    print(f"Total: {total} valid CoNLL-U chat examples")


if __name__ == "__main__":
    main()
