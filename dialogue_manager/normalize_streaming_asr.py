#!/usr/bin/env python3
"""Normalize synthetic streaming hypotheses to ChunkFormer-like plain text."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path


def normalize(text: str) -> str:
    """Keep lexical content while removing display punctuation and casing."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(
        " " if unicodedata.category(char).startswith("P") else char for char in text
    )
    return " ".join(text.split())


def process(path: Path) -> tuple[int, int]:
    record = json.loads(path.read_text(encoding="utf-8"))
    turns = record.get("response", {}).get("turns", [])
    changed = 0
    for turn in turns:
        before = str(turn.get("streaming_asr", ""))
        after = normalize(before)
        if after != before:
            turn["streaming_asr"] = after
            changed += 1
    if changed:
        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return changed, len(turns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--include-raw", action="store_true")
    args = parser.parse_args()
    directories = [args.root / "accepted"]
    if args.include_raw:
        directories.append(args.root / "raw")
    files = changed_files = changed_turns = total_turns = 0
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            changed, turns = process(path)
            files += 1
            total_turns += turns
            changed_turns += changed
            changed_files += int(changed > 0)
    print(json.dumps({
        "files": files,
        "changed_files": changed_files,
        "turns": total_turns,
        "changed_turns": changed_turns,
    }, indent=2))


if __name__ == "__main__":
    main()
