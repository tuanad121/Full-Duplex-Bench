#!/usr/bin/env python3
"""Compute the language-independent metrics defined by English FDB.

The thresholds and formulas intentionally match the upstream v1/v1.5 scripts:
one second of detected speech (or more than three ASR chunks) counts as a turn;
negative latency is clipped to zero for aggregation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TURN_DURATION_THRESHOLD = 1.0
TURN_NUM_WORDS_THRESHOLD = 3


def speech_bounds(output: dict) -> tuple[float, float] | None:
    intervals = output.get("evaluation_speech_intervals") or output.get("timing_provenance", {}).get("speech_intervals", [])
    if intervals:
        return float(intervals[0]["start"]), float(intervals[-1]["end"])
    return None


def detected_turn(chunks: list[dict], bounds: tuple[float, float] | None) -> bool:
    if not chunks or not bounds:
        return False
    start, end = bounds
    duration = end - start
    return duration >= TURN_DURATION_THRESHOLD or len(chunks) > TURN_NUM_WORDS_THRESHOLD


def task_records(task_dir: Path, annotation: str, event_time_index: int) -> list[dict]:
    records = []
    for sample in sorted(path for path in task_dir.iterdir() if path.is_dir()):
        output = json.loads((sample / "output.json").read_text(encoding="utf-8"))
        metadata = json.loads((sample / annotation).read_text(encoding="utf-8"))
        chunks = output["chunks"]
        bounds = speech_bounds(output)
        takes_turn = detected_turn(chunks, bounds)
        latency = None
        if takes_turn:
            event_time = metadata[0]["timestamp"][event_time_index]
            latency = max(0.0, bounds[0] - event_time)
        records.append({"sample_id": sample.name, "takes_turn": takes_turn, "latency": latency})
    return records


def summarize(records: list[dict]) -> dict:
    turns = [record for record in records if record["takes_turn"]]
    latencies = [record["latency"] for record in turns]
    return {
        "samples": len(records),
        "take_over_count": len(turns),
        "take_over_rate": len(turns) / len(records),
        "mean_latency_seconds_given_take_over": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    v1 = args.compat_root / "v1_0"
    pause = summarize(task_records(v1 / "pause_handling", "pause.json", 0))
    smooth = summarize(task_records(v1 / "smooth_turn_taking", "turn_taking.json", 0))
    interruption = summarize(task_records(v1 / "user_interruption", "interrupt.json", 1))

    report = {
        "metric_source": "English FDB v1/v1.5 evaluation scripts",
        "thresholds": {
            "minimum_turn_duration_seconds": TURN_DURATION_THRESHOLD,
            "short_turn_max_asr_chunks": TURN_NUM_WORDS_THRESHOLD,
            "negative_latency": "clipped to zero",
        },
        "metrics": {
            "pause_handling": {
                **pause,
                "correct_wait_rate": 1.0 - pause["take_over_rate"],
            },
            "smooth_turn_taking": smooth,
            "user_interruption": interruption,
        },
        "not_transferred": {
            "backchannel_jsd": "English ICC ground-truth distribution is language/corpus specific.",
            "user_interruption_relevance": "Requires a Vietnamese semantic judge; timing metrics above are unchanged.",
            "v1_5_behavior": "Requires Vietnamese ASR and a localized role-aware behavior prompt.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {
        task: {key: value for key, value in values.items() if key != "records"}
        for task, values in report["metrics"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
