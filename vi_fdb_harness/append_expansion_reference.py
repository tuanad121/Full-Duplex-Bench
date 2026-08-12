#!/usr/bin/env python3
"""Append a sanitized expansion run to an existing public reference package."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from package_reference_run import sanitized_events


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def release_id(version: str, task: str, sample_id: str) -> str:
    if task != "user_interruption":
        return sample_id
    prefix = "standard" if version == "v1_0" else "counterfactual"
    return sample_id if sample_id.startswith(prefix + "_") else f"{prefix}_{sample_id}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--benchmark-revision", required=True)
    args = parser.parse_args()

    metadata_path = args.package_root / "data" / "metadata.jsonl"
    pilot_rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    pilot_rows = [row for row in pilot_rows if row.get("benchmark_split") != "expansion_240"]
    rows = list(pilot_rows)

    for version in ("v1_0", "v1_5"):
        for task_root in sorted(path for path in (args.run_root / version).iterdir() if path.is_dir()):
            for sample in sorted(path for path in task_root.iterdir() if path.is_dir()):
                rid = release_id(version, task_root.name, sample.name)
                for condition in ("event", "clean"):
                    prefix = "" if condition == "event" else "clean_"
                    audio = sample / f"{prefix}output.wav"
                    if not audio.exists():
                        continue
                    asr_name = "output.phowhisper_vad.json" if condition == "event" else "clean_output.phowhisper.json"
                    asr = read(sample / asr_name)
                    timing = read(sample / f"{prefix}output_timing.json")
                    judge_path = sample / "judge.json"
                    judge = read(judge_path) if condition == "event" and judge_path.exists() else None
                    relative = f"audio/{version}/{task_root.name}/{sample.name}/{condition}.wav"
                    destination = args.package_root / "data" / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(audio, destination)
                    base = (
                        "https://huggingface.co/datasets/tuanamz/vi-fdb-v1/resolve/"
                        f"{args.benchmark_revision}/data/expansion_240/{task_root.name}/{rid}"
                    )
                    chunks = asr.get("chunks", [])
                    rows.append({
                        "file_name": relative,
                        "benchmark_repo": "tuanamz/vi-fdb-v1",
                        "benchmark_revision": args.benchmark_revision,
                        "benchmark_split": "expansion_240",
                        "benchmark_sample_id": rid,
                        "source_suite": version.replace("_", "."),
                        "task": task_root.name,
                        "condition": condition,
                        "model": "gpt-realtime",
                        "benchmark_input_url": f"{base}/{'input.wav' if condition == 'event' else 'clean_input.wav'}",
                        "benchmark_metadata_url": f"{base}/metadata.json",
                        "transcript": asr.get("text", ""),
                        "transcript_source": "vinai/PhoWhisper-large",
                        "asr_backend": asr.get("backend"),
                        "asr_chunks_json": json.dumps(chunks, ensure_ascii=False),
                        "asr_chunks_scope": "full_output_word_level",
                        "asr_timing_provenance_json": json.dumps(asr.get("timing_provenance"), ensure_ascii=False),
                        "clock": timing.get("clock"),
                        "input_duration_seconds": timing.get("input_duration"),
                        "output_duration_seconds": timing.get("output_timeline_duration"),
                        "timing_events_json": json.dumps(sanitized_events(timing), ensure_ascii=False),
                        "automated_pass": judge.get("pass") if judge else None,
                        "final_pass": judge.get("pass") if judge else None,
                        "adjudication": "automated" if judge else None,
                        "observed_behavior": judge.get("observed_behavior") if judge else None,
                        "desired_behavior": judge.get("desired_behavior") if judge else None,
                        "judge_model": judge.get("judge_model") if judge else None,
                        "judge_confidence": judge.get("confidence") if judge else None,
                        "judge_evidence_vi": judge.get("evidence_vi") if judge else None,
                    })

    metadata_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "event": sum(r["condition"] == "event" for r in rows), "clean": sum(r["condition"] == "clean" for r in rows)}))


if __name__ == "__main__":
    main()
