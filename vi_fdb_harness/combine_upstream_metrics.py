#!/usr/bin/env python3
"""Combine disjoint original-FDB metric reports without changing formulas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    metrics = {}
    for task in ("pause_handling", "smooth_turn_taking", "user_interruption"):
        records = [record for report in reports for record in report["metrics"][task]["records"]]
        turns = [record for record in records if record["takes_turn"]]
        latencies = [record["latency"] for record in turns]
        result = {
            "samples": len(records),
            "take_over_count": len(turns),
            "take_over_rate": len(turns) / len(records),
            "mean_latency_seconds_given_take_over": sum(latencies) / len(latencies) if latencies else None,
            "records": records,
        }
        if task == "pause_handling":
            result["correct_wait_rate"] = 1.0 - result["take_over_rate"]
        metrics[task] = result
    combined = {
        "metric_source": "Original Full-Duplex-Bench v1.0 formulas and thresholds",
        "inputs": [str(path) for path in args.inputs],
        "thresholds": reports[0]["thresholds"],
        "metrics": metrics,
        "not_transferred": reports[0]["not_transferred"],
    }
    args.output.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({task: {k: v for k, v in result.items() if k != "records"} for task, result in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
