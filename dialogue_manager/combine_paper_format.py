#!/usr/bin/env python3
"""Combine validated paper-format DM exports without conversation leakage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    seen_examples: set[str] = set()
    conversation_sources: dict[str, str] = {}
    action_counts: Counter[str] = Counter()
    source_stats = []
    for source in args.inputs:
        source_rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        source_conversations = {row["conversation_id"] for row in source_rows}
        for cid in source_conversations:
            previous = conversation_sources.get(cid)
            if previous:
                raise ValueError(
                    f"Conversation {cid} occurs in both {previous} and {source}"
                )
            conversation_sources[cid] = str(source)
        for row in source_rows:
            eid = row["example_id"]
            if eid in seen_examples:
                raise ValueError(f"Duplicate example_id: {eid}")
            seen_examples.add(eid)
            action_counts[row["target"]] += 1
            rows.append(row)
        source_stats.append(
            {
                "path": str(source),
                "conversations": len(source_conversations),
                "decisions": len(source_rows),
            }
        )

    rows.sort(key=lambda row: (row["conversation_id"], row["round"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "vi_dm_combined_paper_format_v1",
        "sources": source_stats,
        "conversations": len(conversation_sources),
        "decisions": len(rows),
        "action_counts": dict(sorted(action_counts.items())),
        "duplicate_conversations": 0,
        "duplicate_examples": 0,
    }
    manifest_path = args.output.with_name("manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
