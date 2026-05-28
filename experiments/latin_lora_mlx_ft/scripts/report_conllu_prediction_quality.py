#!/usr/bin/env python3
import argparse
from pathlib import Path


def sentence_blocks(text):
    return [block for block in text.strip().split("\n\n") if block.strip()]


def inspect_block(block):
    lines = block.splitlines()
    has_sent_id = bool(lines) and lines[0].startswith("# sent_id = ")
    has_text = len(lines) > 1 and lines[1].startswith("# text = ")
    token_rows = [line for line in lines if line and not line.startswith("#")]
    rows_with_10_columns = sum(1 for row in token_rows if len(row.split("\t")) == 10)
    return {
        "has_sent_id": has_sent_id,
        "has_text": has_text,
        "token_rows": len(token_rows),
        "rows_with_10_columns": rows_with_10_columns,
        "fully_valid_shape": has_sent_id
        and has_text
        and bool(token_rows)
        and rows_with_10_columns == len(token_rows),
    }


def main():
    parser = argparse.ArgumentParser(description="Report coarse CoNLL-U prediction quality.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        blocks = sentence_blocks(text)
        stats = [inspect_block(block) for block in blocks]
        print(f"{path}")
        print(f"  sentence_blocks: {len(blocks)}")
        print(f"  with_sent_id: {sum(item['has_sent_id'] for item in stats)}")
        print(f"  with_text: {sum(item['has_text'] for item in stats)}")
        print(f"  fully_valid_shape: {sum(item['fully_valid_shape'] for item in stats)}")
        print(f"  token_rows: {sum(item['token_rows'] for item in stats)}")
        print(f"  rows_with_10_columns: {sum(item['rows_with_10_columns'] for item in stats)}")


if __name__ == "__main__":
    main()
