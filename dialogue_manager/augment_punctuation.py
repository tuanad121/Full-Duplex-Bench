#!/usr/bin/env python3
"""Create base/stripped/period-added views of label-only DM decisions."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from export_paper_format import serialize_input


TRAILING = re.compile(r"[.?!,:;。！？，；]$")


def variant(text: str, kind: str) -> str:
    text = text.strip()
    if kind == "base":
        return text
    if kind == "strip":
        return TRAILING.sub("", text)
    return text if TRAILING.search(text) else text + "."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = []
    for row in rows:
        for kind in ("base", "strip", "add"):
            copy = dict(row)
            copy["input"] = dict(row["input"])
            copy["input"]["streaming_asr"] = variant(row["input"]["streaming_asr"], kind)
            copy["example_id"] = f"{row['example_id']}_p{kind}"
            copy["messages"] = [dict(message) for message in row["messages"]]
            copy["messages"][-2]["content"] = serialize_input(copy["input"])
            copy["punctuation_view"] = kind
            output.append(copy)
    random.Random(args.seed).shuffle(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({"source_decisions": len(rows), "augmented_decisions": len(output)}, indent=2))


if __name__ == "__main__":
    main()
