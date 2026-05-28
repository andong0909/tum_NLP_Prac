#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


METRICS = ["UPOS", "UAS", "LAS", "CLAS", "MLAS", "BLEX"]


def parse_score_file(path):
    scores = {}
    pattern = re.compile(r"^\s*([A-Za-z]+)\s+\|\s+([0-9.]+)")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            scores[match.group(1)] = float(match.group(2))
    return scores


def main():
    parser = argparse.ArgumentParser(description="Score one CoNLL-U prediction file.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--system-name", default="lora_qwen")
    parser.add_argument("--scorer", type=Path, default=Path("scripts/conll18_ud_eval.py"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-root", type=Path, default=Path("runs"))
    parser.add_argument("--model", default="")
    parser.add_argument("--adapter-path", default="")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%dT%H%M%S%z")
    run_dir = args.out_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    score_file = run_dir / f"{args.system_name}.score.txt"
    with score_file.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            ["python3", str(args.scorer), "-v", str(args.gold), str(args.pred)],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    scores = parse_score_file(score_file)
    ok = result.returncode == 0 and bool(scores)
    error = None
    if not ok:
        error = score_file.read_text(encoding="utf-8", errors="replace").strip()

    metadata = {
        "run_id": args.run_name,
        "run_timestamp": timestamp,
        "task": "Latin dependency parsing to CoNLL-U",
        "system": args.system_name,
        "model": args.model,
        "adapter_path": args.adapter_path,
        "gold_file": str(args.gold),
        "prediction_file": str(args.pred),
        "score_file": str(score_file),
        "ok": ok,
        "scores": scores,
        "error": error,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["system", "ok", "error", *METRICS])
        writer.writerow([args.system_name, ok, error or "", *[scores.get(metric, "") for metric in METRICS]])

    lines = [
        "# CoNLL-U Evaluation Summary",
        "",
        "| System | UPOS | UAS | LAS | CLAS | MLAS | BLEX |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if ok:
        lines.append(
            "| {system} | {UPOS:.2f} | {UAS:.2f} | {LAS:.2f} | {CLAS:.2f} | {MLAS:.2f} | {BLEX:.2f} |".format(
                system=args.system_name,
                UPOS=scores.get("UPOS", 0.0),
                UAS=scores.get("UAS", 0.0),
                LAS=scores.get("LAS", 0.0),
                CLAS=scores.get("CLAS", 0.0),
                MLAS=scores.get("MLAS", 0.0),
                BLEX=scores.get("BLEX", 0.0),
            )
        )
    else:
        lines.append(f"| {args.system_name} | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR |")
        lines.extend(["", "## Scoring Error", "", "```text", error or "Unknown scoring error", "```"])
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print((run_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
