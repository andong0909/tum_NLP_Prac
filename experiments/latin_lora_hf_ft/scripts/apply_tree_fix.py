#!/usr/bin/env python3
"""Apply the latinbench minimal-repair protocol to a CoNLL-U prediction file.

Mirrors lmstudio_llm.py's two-level repair so results are protocol-comparable:

  1. token-level: any word row with an unusable HEAD (non-integer, out of
     range, self-loop) gets a right-branching default — head = next token,
     last token = root, deprel "dep" (model deprel kept if present).
  2. tree-level (_repair_tree semantics): smallest mutation to reach a valid
     tree — first head=0 kept as root; extra roots re-pointed to it; if no
     root, last token becomes root; cycles broken by re-pointing the
     highest-id member of each cycle to the root. Deprels preserved.

Every changed head increments the fallback count, reported per file and per
sentence so you can fill meta.yaml's repair_pct and caveats.

    python apply_minimal_repair.py pred_poetry.conllu

Writes, next to the input (or into --out-dir):
    pred_poetry_tree_safe.conllu      the legalized predictions
    meta_tree_fix_pred_poetry.json    summary of what was changed
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SENT_ID_RE = re.compile(r"^#\s*sent_id\s*=\s*(\S+)")


def split_sentences(text: str) -> list[str]:
    return [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]


def sent_id_of(block: str) -> str:
    for line in block.splitlines():
        m = SENT_ID_RE.match(line)
        if m:
            return m.group(1)
    return "<no sent_id>"


def repair_block(block: str) -> tuple[str, int]:
    """Return (repaired block, number of head pointers changed)."""
    lines = block.splitlines()
    # collect word rows (integer ID); remember their line indices
    word_idx: list[int] = []
    for i, line in enumerate(lines):
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) == 10 and re.fullmatch(r"\d+", cols[0]):
            word_idx.append(i)
    n = len(word_idx)
    if n == 0:
        return block, 0

    heads: dict[int, int] = {}
    deprels: dict[int, str] = {}
    changed: set[int] = set()

    # ---- level 1: token-level validation with right-branching default ----
    for pos, i in enumerate(word_idx):
        cols = lines[i].split("\t")
        tid = int(cols[0])
        try:
            head = int(cols[6])
            # NB: head == tid (self-loop) is token-valid in Nico's protocol;
            # it's a length-1 cycle handled at tree level (re-point to root).
            ok = 0 <= head <= n
        except ValueError:
            ok = False
        if not ok:
            head = tid + 1 if pos < n - 1 else 0  # right-branching; last = root
            changed.add(tid)
        heads[tid] = head
        deprels[tid] = cols[7] if cols[7] not in ("", "_") else "dep"

    ids = sorted(heads)

    # ---- level 2: tree-level minimal repair ----
    roots = [t for t in ids if heads[t] == 0]
    if roots:
        root = roots[0]
        for extra in roots[1:]:
            heads[extra] = root
            changed.add(extra)
    else:
        root = ids[-1]           # no root: last token becomes root
        heads[root] = 0
        changed.add(root)

    # break cycles: walk each token to root; re-point highest-id cycle member
    def find_cycle(start: int) -> list[int] | None:
        seen: dict[int, int] = {}
        cur, step = start, 0
        while cur != 0:
            if cur in seen:
                path = list(seen)
                return path[path.index(cur):]  # the cycle portion
            seen[cur] = step
            cur = heads[cur]
            step += 1
        return None

    for t in ids:
        cycle = find_cycle(t)
        while cycle:
            fix = max(cycle)
            heads[fix] = root if fix != root else 0
            changed.add(fix)
            cycle = find_cycle(t)

    # ---- write back: only HEAD column mutated (deprel of repaired root) ----
    for i in word_idx:
        cols = lines[i].split("\t")
        tid = int(cols[0])
        if tid in changed:
            cols[6] = str(heads[tid])
            cols[7] = "root" if heads[tid] == 0 else deprels[tid]
            lines[i] = "\t".join(cols)
    return "\n".join(lines), len(changed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="directory for outputs (default: alongside the input)")
    args = ap.parse_args()

    # Derive output paths from the input name:
    #   <stem>.conllu            -> <stem>_tree_safe.conllu
    #   meta_tree_fix_<stem>.json
    out_dir = args.out_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    out_path = out_dir / f"{stem}_tree_safe{args.input.suffix}"
    meta_path = out_dir / f"meta_tree_fix_{stem}.json"

    blocks = split_sentences(args.input.read_text(encoding="utf-8"))
    out_blocks: list[str] = []
    total_tokens = 0
    fallback_tokens = 0
    fallback_sents = 0
    repaired_detail: list[dict] = []
    for block in blocks:
        repaired, n_changed = repair_block(block)
        out_blocks.append(repaired)
        n_words = sum(1 for l in block.splitlines()
                      if not l.startswith("#")
                      and len(l.split("\t")) == 10
                      and re.fullmatch(r"\d+", l.split("\t")[0]))
        total_tokens += n_words
        if n_changed:
            fallback_tokens += n_changed
            fallback_sents += 1
            sid = sent_id_of(block)
            print(f"repaired {sid}: {n_changed} head(s) changed")
            repaired_detail.append({
                "sent_id": sid,
                "tokens": n_words,
                "heads_changed": n_changed,
            })

    out_path.write_text("\n\n".join(out_blocks) + "\n\n", encoding="utf-8")

    pct_tok = 100.0 * fallback_tokens / max(total_tokens, 1)
    pct_sent = 100.0 * fallback_sents / max(len(blocks), 1)

    summary = {
        "source_file": args.input.name,
        "output_file": out_path.name,
        "protocol": "latinbench minimal-repair (right-branching token fallback "
                    "+ minimal tree legalization; heads legalized, deprels kept)",
        "sentences_total": len(blocks),
        "sentences_touched": fallback_sents,
        "sentences_touched_pct": round(pct_sent, 2),
        "tokens_total": total_tokens,
        "fallback_tokens": fallback_tokens,
        "fallback_tokens_pct": round(pct_tok, 2),
        "repaired_sentences": repaired_detail,
    }
    meta_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")

    print(f"\nwrote {out_path}")
    print(f"wrote {meta_path}")
    print(f"fallback tokens: {fallback_tokens}/{total_tokens} ({pct_tok:.2f}%)")
    print(f"sentences touched: {fallback_sents}/{len(blocks)} ({pct_sent:.2f}%)"
          f"  <- use this for repair_pct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())