#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Extract assistant CoNLL-U gold outputs from MLX chat JSONL."
    )
    parser.add_argument("input", type=Path, help="MLX chat JSONL file")
    parser.add_argument("output", type=Path, help="Output .conllu file")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    blocks = []
    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        example = json.loads(line)
        messages = example.get("messages", [])
        if len(messages) != 3 or messages[2].get("role") != "assistant":
            raise ValueError(f"{args.input}:{line_number}: malformed chat example")
        blocks.append(messages[2]["content"].strip())
        if args.limit is not None and len(blocks) >= args.limit:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(blocks) + "\n\n", encoding="utf-8")
    print(f"Wrote {len(blocks)} gold CoNLL-U sentences to {args.output}")


if __name__ == "__main__":
    main()
