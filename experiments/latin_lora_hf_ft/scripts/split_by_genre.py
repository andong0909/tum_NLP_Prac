#!/usr/bin/env python3
"""Split a combined CoNLL-U file into poetry and prose files by sent_id.

Poetry sentences have sent_id of the form  xxx-P-xx-xx  (e.g. SenHerFu-P-15-81)
Prose  sentences have sent_id of the form  xxx-Q-xx-xx  (e.g. TacGerma-Q-01-164)

Run it on BOTH your lumped gold file and your lumped prediction file:

    python scripts/split_by_genre.py combined_gold.conllu --out-dir gold_split
    python scripts/split_by_genre.py combined_pred.conllu --out-dir pred_split

Output: <out-dir>/<stem>_poetry.conllu and <out-dir>/<stem>_prose.conllu,
preserving original sentence order within each genre (which is what the
sequential CoNLL-18 scorer alignment requires — as long as gold and pred
were in the same order before splitting, they stay aligned after).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SENT_ID_RE = re.compile(r"^#\s*sent_id\s*=\s*(\S+)")
GENRE_RE = re.compile(r"^[^-]+-([PQ])-")

GENRE_NAMES = {"P": "poetry", "Q": "prose"}


def split_sentences(text: str) -> list[str]:
    """Split raw CoNLL-U text into sentence blocks (comments + token rows)."""
    # Sentences are separated by blank lines; keep each block intact.
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    return blocks


def genre_of(block: str) -> str:
    """Return 'poetry' or 'prose' based on the sent_id line; raise if unknown."""
    for line in block.splitlines():
        m = SENT_ID_RE.match(line)
        if m:
            sent_id = m.group(1)
            g = GENRE_RE.match(sent_id)
            if not g:
                raise SystemExit(
                    f"sent_id {sent_id!r} doesn't match the expected "
                    f"xxx-P-... / xxx-Q-... pattern"
                )
            return GENRE_NAMES[g.group(1)]
    raise SystemExit(
        "sentence block without a '# sent_id = ...' line:\n"
        + block.splitlines()[0]
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", type=Path, help="combined .conllu file")
    ap.add_argument("--out-dir", type=Path, default=Path("."),
                    help="directory for the two output files (default: cwd)")
    args = ap.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"input not found: {args.input}")

    blocks = split_sentences(args.input.read_text(encoding="utf-8"))
    buckets: dict[str, list[str]] = {"poetry": [], "prose": []}
    for block in blocks:
        buckets[genre_of(block)].append(block)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    for genre, sents in buckets.items():
        out_path = args.out_dir / f"{stem}_{genre}.conllu"
        # CoNLL-U requires a blank line after every sentence, incl. the last.
        out_path.write_text("\n\n".join(sents) + "\n\n", encoding="utf-8")
        print(f"wrote {out_path}  ({len(sents)} sentences)")

    if not buckets["poetry"] or not buckets["prose"]:
        print("WARNING: one genre is empty — check your input file",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())