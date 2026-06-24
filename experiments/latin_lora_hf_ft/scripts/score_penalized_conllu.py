#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def split_conllu_blocks(path):
    text = path.read_text(encoding="utf-8")
    return [block for block in text.strip().split("\n\n") if block.strip()]


def sent_id_from_block(block, fallback):
    for line in block.splitlines():
        if line.startswith("# sent_id"):
            return line.split("=", 1)[1].strip()
    return str(fallback)


def token_rows_from_block(block):
    rows = []
    for line_index, line in enumerate(block.splitlines()):
        if not line or line.startswith("#"):
            continue
        columns = line.split("\t")
        if columns and "-" not in columns[0] and "." not in columns[0]:
            rows.append((line_index, columns))
    return rows


def tree_error(block):
    rows = token_rows_from_block(block)
    ids = []
    heads = {}
    for _, columns in rows:
        if len(columns) != 10:
            return f"Expected 10 columns, got {len(columns)} in row: {columns}"
        try:
            token_id = int(columns[0])
        except ValueError:
            return f"Cannot parse token ID: {columns[0]!r}"
        try:
            head = int(columns[6])
        except ValueError:
            return f"Cannot parse HEAD: {columns[6]!r}"
        ids.append(token_id)
        heads[token_id] = head

    id_set = set(ids)
    roots = [token_id for token_id, head in heads.items() if head == 0]
    if len(roots) != 1:
        return f"Expected exactly one root, got {len(roots)} roots: {roots}"

    for token_id, head in heads.items():
        if head != 0 and head not in id_set:
            return f"Token {token_id} points to missing HEAD {head}"

    for token_id in ids:
        seen = set()
        current = token_id
        while current != 0:
            if current in seen:
                return f"Cycle detected while following token {token_id}: revisited {current}"
            seen.add(current)
            current = heads[current]
    return None


def parse_gold_heads(gold_block):
    rows = token_rows_from_block(gold_block)
    heads = {}
    for _, columns in rows:
        heads[int(columns[0])] = int(columns[6])
    return heads


def choose_adversarial_root(gold_block):
    """Choose a non-root gold leaf when possible.

    For sentences with two or more syntactic tokens, making a gold leaf the
    predicted root and attaching every other token to it guarantees zero UAS:
    the dummy root is wrong because it was not gold root, and all other heads
    are wrong because no gold token depends on a leaf.
    """
    heads = parse_gold_heads(gold_block)
    dependents = {head for head in heads.values() if head != 0}
    non_root_leaves = [
        token_id
        for token_id, head in heads.items()
        if head != 0 and token_id not in dependents
    ]
    if non_root_leaves:
        return non_root_leaves[0], "gold_non_root_leaf"

    # Single-token sentences cannot be forced to zero UAS while remaining valid:
    # the only legal valid tree has HEAD=0. Keep the file scoreable and report it.
    return next(iter(heads)), "fallback_first_token"


def make_dummy_block(gold_block):
    dummy_root, strategy = choose_adversarial_root(gold_block)
    lines = gold_block.splitlines()
    rows = token_rows_from_block(gold_block)
    for line_index, columns in rows:
        columns = columns.copy()
        token_id = int(columns[0])
        columns[6] = "0" if token_id == dummy_root else str(dummy_root)
        columns[7] = "dep:dummy"
        lines[line_index] = "\t".join(columns)
    return "\n".join(lines), dummy_root, strategy


def run_command(command):
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(str(part) for part in command)
            + "\n\nSTDOUT:\n"
            + result.stdout
            + "\nSTDERR:\n"
            + result.stderr
        )
    return result


def parse_scores(summary_text):
    for line in summary_text.splitlines():
        if line.startswith("|") and "ERROR" not in line and "---" not in line and "System" not in line:
            columns = [column.strip() for column in line.strip("|").split("|")]
            if len(columns) == 7:
                return {
                    "system": columns[0],
                    "UPOS": columns[1],
                    "UAS": columns[2],
                    "LAS": columns[3],
                    "CLAS": columns[4],
                    "MLAS": columns[5],
                    "BLEX": columns[6],
                }
    return {}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Score a full CoNLL-U run by replacing dependency-tree-invalid "
            "prediction blocks with adversarial dummy valid trees, then running "
            "the official scorer over all sentences."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--pred", type=Path)
    parser.add_argument("--out-name", default="penalized_full_score")
    parser.add_argument("--system-name", default="lora_qwen_penalized")
    parser.add_argument("--model", default="")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument(
        "--scorer-wrapper",
        type=Path,
        default=Path("../latin_lora_mlx_ft/scripts/score_single_conllu_run.py"),
    )
    parser.add_argument(
        "--conll18-scorer",
        type=Path,
        default=Path("../latin_lora_mlx_ft/scripts/conll18_ud_eval.py"),
    )
    args = parser.parse_args()

    gold_path = args.gold or args.run_dir / "gold.conllu"
    pred_path = args.pred or args.run_dir / "pred.conllu"
    out_dir = args.run_dir / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_blocks = split_conllu_blocks(gold_path)
    pred_blocks = split_conllu_blocks(pred_path)
    if len(gold_blocks) != len(pred_blocks):
        raise ValueError(
            f"Gold/prediction block count mismatch: {len(gold_blocks)} gold blocks, "
            f"{len(pred_blocks)} prediction blocks. This scorer expects rendered "
            "prediction files; handle render failures before using it."
        )

    penalized_blocks = []
    replacements = []
    for index, (gold_block, pred_block) in enumerate(zip(gold_blocks, pred_blocks), 1):
        error = tree_error(pred_block)
        if error is None:
            penalized_blocks.append(pred_block)
            continue

        dummy_block, dummy_root, strategy = make_dummy_block(gold_block)
        dummy_error = tree_error(dummy_block)
        if dummy_error is not None:
            raise ValueError(f"Internal error: dummy block is invalid for sentence {index}: {dummy_error}")
        penalized_blocks.append(dummy_block)
        replacements.append(
            {
                "index": index,
                "sent_id": sent_id_from_block(gold_block, index),
                "raw_error": error,
                "dummy_root": dummy_root,
                "dummy_strategy": strategy,
                "expected_dependency_score": (
                    "guaranteed_zero_uas_las_for_multi_token_sentence"
                    if strategy == "gold_non_root_leaf"
                    else "single_token_sentence_cannot_force_zero_uas_with_valid_tree"
                ),
            }
        )

    penalized_pred = out_dir / "pred_penalized.conllu"
    penalized_pred.write_text("\n\n".join(penalized_blocks) + "\n\n", encoding="utf-8")
    gold_copy = out_dir / "gold.conllu"
    gold_copy.write_text("\n\n".join(gold_blocks) + "\n\n", encoding="utf-8")

    report = {
        "label": "Penalized full score: invalid prediction trees replaced with adversarial dummy valid trees.",
        "run_dir": str(args.run_dir),
        "gold": str(gold_path),
        "raw_pred": str(pred_path),
        "penalized_pred": str(penalized_pred),
        "total_sentences": len(gold_blocks),
        "valid_raw_sentences": len(gold_blocks) - len(replacements),
        "dummy_replaced_sentences": len(replacements),
        "dummy_strategy": "Use a gold non-root leaf as predicted root, attach all other tokens to it, use DEPREL=dep:dummy.",
        "guarantee": (
            "For multi-token sentences this guarantees zero UAS/LAS for the replaced sentence. "
            "For one-token sentences UAS cannot be forced to zero while keeping a valid tree."
        ),
        "replacements": replacements,
    }
    (out_dir / "dummy_penalty_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    run_command(
        [
            "python3",
            str(args.scorer_wrapper),
            "--gold",
            str(gold_copy),
            "--pred",
            str(penalized_pred),
            "--system-name",
            args.system_name,
            "--scorer",
            str(args.conll18_scorer),
            "--run-name",
            args.out_name,
            "--out-root",
            str(args.run_dir),
            "--model",
            args.model,
            "--adapter-path",
            args.adapter_path,
        ]
    )

    summary_path = out_dir / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8")
    scores = parse_scores(summary_text)
    report["scores"] = scores
    (out_dir / "dummy_penalty_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    note = (
        "# Penalized Full Score\n\n"
        "This score keeps every sentence in the split. Any rendered prediction "
        "with an invalid dependency tree is replaced by an adversarial dummy "
        "valid tree before running the official CoNLL-18 scorer.\n\n"
        f"- Total sentences: {len(gold_blocks)}\n"
        f"- Raw tree-valid sentences: {len(gold_blocks) - len(replacements)}\n"
        f"- Dummy-replaced sentences: {len(replacements)}\n"
        f"- Dummy-replaced sent_ids: {', '.join(item['sent_id'] for item in replacements)}\n"
        "- Dummy strategy: choose a gold non-root leaf as dummy root; attach all other tokens to it; use `dep:dummy`.\n"
        "- Guarantee: multi-token dummy replacements receive 0 UAS/LAS for that sentence.\n\n"
        + summary_text
    )
    (out_dir / "PENALIZED_SCORE.md").write_text(note, encoding="utf-8")
    print((out_dir / "PENALIZED_SCORE.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
