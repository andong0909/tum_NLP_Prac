#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


METRICS = ["Tokens", "Sentences", "Words", "UPOS", "UAS", "LAS", "CLAS", "MLAS", "BLEX"]


def parse_score_file(path):
    scores = {}
    pattern = re.compile(r"^\s*([A-Za-z]+)\s+\|\s+([0-9.]+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            scores[match.group(1)] = float(match.group(2))
    return scores


def run_scorer(scorer, gold, system_file, score_file):
    with score_file.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            ["python3", str(scorer), "-v", str(gold), str(system_file)],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    scores = parse_score_file(score_file)
    error = None
    if result.returncode != 0:
        error = score_file.read_text(encoding="utf-8", errors="replace").strip()
    elif not scores:
        error = "Scorer completed but no metric rows were parsed."
    return {"ok": result.returncode == 0 and bool(scores), "scores": scores, "error": error}


def write_summary_markdown(path, rows):
    lines = [
        "# CoNLL-U Evaluation Summary",
        "",
        "| System | UPOS | UAS | LAS | CLAS | MLAS | BLEX |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["ok"]:
            lines.append(
                "| {name} | {UPOS:.2f} | {UAS:.2f} | {LAS:.2f} | {CLAS:.2f} | {MLAS:.2f} | {BLEX:.2f} |".format(
                name=row["name"],
                UPOS=row["scores"].get("UPOS", 0.0),
                UAS=row["scores"].get("UAS", 0.0),
                LAS=row["scores"].get("LAS", 0.0),
                CLAS=row["scores"].get("CLAS", 0.0),
                MLAS=row["scores"].get("MLAS", 0.0),
                BLEX=row["scores"].get("BLEX", 0.0),
                )
            )
        else:
            lines.append(f"| {row['name']} | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR |")
    errors = [row for row in rows if not row["ok"]]
    if errors:
        lines.extend(["", "## Scoring Errors", ""])
        for row in errors:
            lines.append(f"### {row['name']}")
            lines.append("")
            lines.append("```text")
            lines.append(row["error"] or "Unknown scoring error")
            lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["system", "ok", "error", *METRICS])
        for row in rows:
            writer.writerow(
                [
                    row["name"],
                    row["ok"],
                    row["error"] or "",
                    *[row["scores"].get(metric, "") for metric in METRICS],
                ]
            )


def main():
    parser = argparse.ArgumentParser(description="Score base and adapted CoNLL-U predictions.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--base-pred", type=Path, required=True)
    parser.add_argument("--adapter-pred", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, default=Path("scripts/conll18_ud_eval.py"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--out-root", type=Path, default=Path("runs"))
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--adapter-path", default="adapters/latin-qwen-conllu-test-001")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%dT%H%M%S%z")
    run_name = args.run_name or f"qwen-conllu-eval-{timestamp}"
    run_dir = args.out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    systems = [
        ("base_qwen", args.base_pred),
        ("lora_qwen", args.adapter_pred),
    ]

    rows = []
    for name, pred_path in systems:
        score_file = run_dir / f"{name}.score.txt"
        result = run_scorer(args.scorer, args.gold, pred_path, score_file)
        rows.append(
            {
                "name": name,
                "prediction_file": str(pred_path),
                "score_file": str(score_file),
                "ok": result["ok"],
                "scores": result["scores"],
                "error": result["error"],
            }
        )

    metadata = {
        "run_id": run_name,
        "run_timestamp": timestamp,
        "task": "Latin dependency parsing to CoNLL-U",
        "gold_file": str(args.gold),
        "scorer": str(args.scorer),
        "model": args.model,
        "adapter_path": args.adapter_path,
        "systems": rows,
    }

    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_markdown(run_dir / "summary.md", rows)
    write_summary_csv(run_dir / "summary.csv", rows)

    print(f"Wrote evaluation run to {run_dir}")
    print((run_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
