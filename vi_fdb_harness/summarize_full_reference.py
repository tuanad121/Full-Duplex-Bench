#!/usr/bin/env python3
"""Summarize pilot/expansion semantic parity from a reference package."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.metadata.read_text(encoding="utf-8").splitlines()]
    events = [row for row in rows if row["condition"] == "event"]
    groups = defaultdict(Counter)
    for row in events:
        split = row.get("benchmark_split", "pilot_160")
        for key in ((split, row["source_suite"], row["task"]), (split, "all", "all"), ("full_400", row["source_suite"], row["task"]), ("full_400", "all", "all")):
            groups[key]["total"] += 1
            groups[key]["pass"] += int(row.get("final_pass") is True)
    records = []
    for (split, suite, task), counts in sorted(groups.items()):
        records.append({"split": split, "source_suite": suite, "task": task, "pass": counts["pass"], "total": counts["total"], "pass_rate": counts["pass"] / counts["total"]})
    invalid_labels = [
        {"split": row.get("benchmark_split", "pilot_160"), "source_suite": row["source_suite"], "task": row["task"], "sample_id": row["benchmark_sample_id"], "observed_behavior": row.get("observed_behavior")}
        for row in events
        if row["task"] == "pause_handling" and row.get("observed_behavior") == "MISSED_INTERRUPTION"
    ]
    report = {
        "model": "gpt-realtime",
        "semantic_judge": "role-aware Vietnamese GPT-4.1-mini; pilot includes 29 native-speaker adjudications; expansion is automated",
        "records": records,
        "known_label_audit": {"invalid_pause_label_count": len(invalid_labels), "pass_counts_affected": False, "cases": invalid_labels},
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.summary_output:
        full = next(record for record in records if record["split"] == "full_400" and record["task"] == "all")
        summary = {
            "model": "gpt-realtime",
            "event_samples": len(events),
            "clean_control_samples": len(rows) - len(events),
            "semantic_evaluated_samples": full["total"],
            "semantic_pass": full["pass"],
            "semantic_pass_rate": full["pass_rate"],
            "pilot_native_speaker_adjudications": 29,
            "semantic_evaluation_note": "pilot_160 includes native-speaker corrections; expansion_240 is automated GPT-4.1-mini judging",
        }
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
