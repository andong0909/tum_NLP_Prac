#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def extract_error_json(render_error_text):
    marker = "ValueError:"
    if marker not in render_error_text:
        raise ValueError("Could not find a ValueError JSON payload in render error text")
    payload = render_error_text.split(marker, 1)[1].strip()
    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Could not locate JSON object in render error text")
    return json.loads(payload[start : end + 1])


def sent_id_from_record(record, fallback):
    user_content = record["messages"][1]["content"]
    for line in user_content.splitlines():
        if line.startswith("# sent_id"):
            return line.split("=", 1)[1].strip()
    return str(fallback)


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
    for line in block.splitlines():
        if not line or line.startswith("#"):
            continue
        columns = line.split("\t")
        if columns and "-" not in columns[0] and "." not in columns[0]:
            rows.append(columns)
    return rows


def tree_error(block):
    rows = token_rows_from_block(block)
    ids = []
    heads = {}
    for columns in rows:
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


def validate_tree_subset(conllu_path, kept_inputs, kept_preds):
    valid_inputs = []
    valid_preds = []
    invalid = []
    blocks = split_conllu_blocks(conllu_path)
    if len(blocks) != len(kept_inputs):
        raise ValueError(
            f"Rendered block count mismatch: {len(blocks)} CoNLL-U blocks for {len(kept_inputs)} input records"
        )
    for kept_index, (block, input_record, pred_record) in enumerate(zip(blocks, kept_inputs, kept_preds), 1):
        error = tree_error(block)
        if error:
            invalid.append(
                {
                    "kept_index": kept_index,
                    "sent_id": sent_id_from_block(block, kept_index),
                    "error": error,
                }
            )
            continue
        valid_inputs.append(input_record)
        valid_preds.append(pred_record)
    return valid_inputs, valid_preds, invalid


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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Score the renderable subset of a sentence-ID HEAD/DEPREL run by "
            "excluding sentence indices reported in render_error.txt."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--pred-jsonl", type=Path)
    parser.add_argument("--render-error", type=Path)
    parser.add_argument("--out-name", default="partial_renderable_score")
    parser.add_argument("--system-name", default="lora_qwen25_sentence_id_head_deprel_partial")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument(
        "--renderer",
        type=Path,
        default=Path("scripts/render_sentence_id_head_deprel_to_conllu.py"),
    )
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

    pred_jsonl = args.pred_jsonl or args.run_dir / "id_head_deprel_predictions.jsonl"
    render_error = args.render_error or args.run_dir / "render_error.txt"
    out_dir = args.run_dir / args.out_name

    error_payload = extract_error_json(render_error.read_text(encoding="utf-8", errors="replace"))
    errors = error_payload.get("errors", [])
    bad_indices = {int(error["index"]) for error in errors if "index" in error}
    if not bad_indices:
        raise ValueError("No error indices found in render error payload")

    input_records = load_jsonl(args.input_jsonl)
    pred_records = load_jsonl(pred_jsonl)
    if len(input_records) != len(pred_records):
        raise ValueError(
            f"Input/prediction count mismatch: {len(input_records)} input records, "
            f"{len(pred_records)} predictions"
        )

    kept_inputs = []
    kept_preds = []
    excluded = []
    for one_based_index, (input_record, pred_record) in enumerate(zip(input_records, pred_records), 1):
        if one_based_index in bad_indices:
            excluded.append(
                {
                    "index": one_based_index,
                    "sent_id": sent_id_from_record(input_record, one_based_index),
                    "error": next(
                        (error for error in errors if int(error.get("index", -1)) == one_based_index),
                        {},
                    ),
                }
            )
            continue
        kept_inputs.append(input_record)
        kept_preds.append(pred_record)

    gold_prediction_records = [{"assistant": record["messages"][2]["content"]} for record in kept_inputs]

    write_jsonl(out_dir / "input_renderable_pre_tree.jsonl", kept_inputs)
    write_jsonl(out_dir / "pred_renderable_pre_tree.jsonl", kept_preds)
    write_jsonl(out_dir / "gold_id_head_deprel_renderable_pre_tree.jsonl", gold_prediction_records)
    (out_dir / "excluded_errors.json").write_text(
        json.dumps(
            {
                "excluded_count": len(excluded),
                "kept_count": len(kept_inputs),
                "total_count": len(input_records),
                "excluded": excluded,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    run_command(
        [
            "python3",
            str(args.renderer),
            "--input-jsonl",
            str(out_dir / "input_renderable_pre_tree.jsonl"),
            "--pred-jsonl",
            str(out_dir / "gold_id_head_deprel_renderable_pre_tree.jsonl"),
            "--output-conllu",
            str(out_dir / "gold_renderable_pre_tree.conllu"),
            "--report",
            str(out_dir / "gold_renderable_pre_tree_report.json"),
        ]
    )
    run_command(
        [
            "python3",
            str(args.renderer),
            "--input-jsonl",
            str(out_dir / "input_renderable_pre_tree.jsonl"),
            "--pred-jsonl",
            str(out_dir / "pred_renderable_pre_tree.jsonl"),
            "--output-conllu",
            str(out_dir / "pred_renderable_pre_tree.conllu"),
            "--report",
            str(out_dir / "pred_renderable_pre_tree_report.json"),
        ]
    )

    tree_valid_inputs, tree_valid_preds, tree_invalid = validate_tree_subset(
        out_dir / "pred_renderable_pre_tree.conllu",
        kept_inputs,
        kept_preds,
    )
    tree_valid_gold_predictions = [
        {"assistant": record["messages"][2]["content"]} for record in tree_valid_inputs
    ]
    write_jsonl(out_dir / "input_renderable_tree_valid.jsonl", tree_valid_inputs)
    write_jsonl(out_dir / "pred_renderable_tree_valid.jsonl", tree_valid_preds)
    write_jsonl(out_dir / "gold_id_head_deprel_tree_valid.jsonl", tree_valid_gold_predictions)
    (out_dir / "excluded_tree_errors.json").write_text(
        json.dumps(
            {
                "excluded_count": len(tree_invalid),
                "kept_count": len(tree_valid_inputs),
                "pre_tree_count": len(kept_inputs),
                "excluded": tree_invalid,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    run_command(
        [
            "python3",
            str(args.renderer),
            "--input-jsonl",
            str(out_dir / "input_renderable_tree_valid.jsonl"),
            "--pred-jsonl",
            str(out_dir / "gold_id_head_deprel_tree_valid.jsonl"),
            "--output-conllu",
            str(out_dir / "gold.conllu"),
            "--report",
            str(out_dir / "gold_render_report.json"),
        ]
    )
    run_command(
        [
            "python3",
            str(args.renderer),
            "--input-jsonl",
            str(out_dir / "input_renderable_tree_valid.jsonl"),
            "--pred-jsonl",
            str(out_dir / "pred_renderable_tree_valid.jsonl"),
            "--output-conllu",
            str(out_dir / "pred.conllu"),
            "--report",
            str(out_dir / "pred_render_report.json"),
        ]
    )

    score_run_name = args.out_name
    run_command(
        [
            "python3",
            str(args.scorer_wrapper),
            "--gold",
            str(out_dir / "gold.conllu"),
            "--pred",
            str(out_dir / "pred.conllu"),
            "--system-name",
            args.system_name,
            "--scorer",
            str(args.conll18_scorer),
            "--run-name",
            score_run_name,
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
    diagnostic = {
        "label": "Partial diagnostic score over renderable and tree-valid sentence subset only. Not an official full-split score.",
        "run_dir": str(args.run_dir),
        "input_jsonl": str(args.input_jsonl),
        "pred_jsonl": str(pred_jsonl),
        "total_sentences": len(input_records),
        "renderable_sentences": len(kept_inputs),
        "tree_valid_sentences": len(tree_valid_inputs),
        "render_excluded_sentences": len(excluded),
        "tree_excluded_sentences": len(tree_invalid),
        "render_excluded_indices": sorted(bad_indices),
        "tree_excluded": tree_invalid,
        "scores": scores,
    }
    (out_dir / "partial_score_metadata.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    note = (
        "# Partial Renderable-Subset Score\n\n"
        "This is a diagnostic score over only the sentences whose predictions rendered successfully "
        "and passed a basic dependency-tree validity check. "
        "It is not an official full-split score.\n\n"
        f"- Total sentences: {len(input_records)}\n"
        f"- Renderable sentences: {len(kept_inputs)}\n"
        f"- Tree-valid sentences scored: {len(tree_valid_inputs)}\n"
        f"- Render-excluded sentences: {len(excluded)}\n"
        f"- Tree-excluded sentences: {len(tree_invalid)}\n"
        f"- Render-excluded indices: {', '.join(str(index) for index in sorted(bad_indices))}\n"
        f"- Tree-excluded sent_ids: {', '.join(item['sent_id'] for item in tree_invalid)}\n\n"
        + summary_text
    )
    (out_dir / "PARTIAL_SCORE.md").write_text(note, encoding="utf-8")
    print((out_dir / "PARTIAL_SCORE.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
