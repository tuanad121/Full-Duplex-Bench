#!/usr/bin/env python3
"""Validate invariants of the authors' deterministic full-duplex transforms."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


CL = "<|continue-listen|>"
SL = "<|switch-to-listen|>"
CS = "<|continue_speak|>"


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--tokenized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clean, tokenized = read_jsonl(args.clean), read_jsonl(args.tokenized)
    errors = []
    counts = Counter()
    if len(clean) != len(tokenized):
        errors.append(f"dialogue count differs: {len(clean)} != {len(tokenized)}")
    for dialogue_index, (source, transformed) in enumerate(zip(clean, tokenized)):
        for round_key, result in transformed.items():
            original = source[round_key]
            counts[result["type"]] += 1
            fragments = [result["user"]]
            continuation = result.get("assistant_continuation")
            continuation3 = result.get("assistant_continuation3")
            if continuation:
                fragments.append(continuation["user"])
                action = continuation["assistant"]
                if action not in {CL, SL}:
                    errors.append(f"{dialogue_index}/{round_key}: invalid continuation")
            if continuation3:
                fragments.extend([continuation3["user"], continuation3["user2"]])
                if continuation3["assistant"] != CL or continuation3["assistant2"] != CL:
                    errors.append(f"{dialogue_index}/{round_key}: invalid double CL")
            if continuation or continuation3:
                reconstructed = "".join(fragments)
                if reconstructed != original["user"]:
                    errors.append(f"{dialogue_index}/{round_key}: user reconstruction")
                if not original["user"].startswith(result["user"]):
                    errors.append(f"{dialogue_index}/{round_key}: not strict prefix")
                if result["user"] == original["user"]:
                    errors.append(f"{dialogue_index}/{round_key}: empty truncation")
            if result["type"] == "Real INTR":
                if not continuation or continuation["assistant"] != SL:
                    errors.append(f"{dialogue_index}/{round_key}: real without S-L")
            if result["type"] == "Fake INTR":
                fake = result.get("assistant_continuation2")
                if not fake or not fake["assistant"].lstrip().startswith(CS):
                    errors.append(f"{dialogue_index}/{round_key}: fake without C-S")
                if fake and any("\u4e00" <= char <= "\u9fff" for char in fake["user"]):
                    errors.append(f"{dialogue_index}/{round_key}: non-Vietnamese fake")

    report = {
        "dialogues": len(tokenized),
        "round_types": dict(counts),
        "errors": errors,
        "passed": not errors,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
