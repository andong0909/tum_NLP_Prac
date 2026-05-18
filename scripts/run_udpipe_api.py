#!/usr/bin/env python3
"""Run UDPipe's REST API on a CoNLL-U file and write CoNLL-U output."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request


API_PROCESS_URL = "https://lindat.mff.cuni.cz/services/udpipe/api/process"


def compact_blank_lines(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    previous_blank = False
    for line in lines:
        blank = not line.strip()
        if blank and previous_blank:
            continue
        out.append(line)
        previous_blank = blank
    return "\n".join(out).rstrip() + "\n\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input CoNLL-U file")
    parser.add_argument("output", help="Output CoNLL-U file")
    parser.add_argument(
        "--model",
        default="latin-evalatin24-240520",
        help="UDPipe model id from /api/models",
    )
    parser.add_argument(
        "--endpoint",
        default=API_PROCESS_URL,
        help="UDPipe /process API endpoint",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        input_data = handle.read()

    payload = urllib.parse.urlencode(
        {
            "model": args.model,
            "input": "conllu",
            "parser": "",
            "data": input_data,
        }
    ).encode("utf-8")

    request = urllib.request.Request(args.endpoint, data=payload)
    with urllib.request.urlopen(request) as response:
        body = response.read().decode("utf-8")

    parsed = json.loads(body)
    if "result" not in parsed:
        print(body, file=sys.stderr)
        return 1

    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(compact_blank_lines(parsed["result"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
