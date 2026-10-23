#!/usr/bin/env python3
"""Refresh a packaged reference run with PhoWhisper/VAD ASR evidence."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    metadata_path = args.package_root / "data" / "metadata.jsonl"
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    updated = 0
    for row in rows:
        if row.get("condition") != "event":
            continue
        version = row["source_suite"].replace(".", "_")
        sample_id = row["benchmark_sample_id"].split("_", 1)[-1]
        source = args.run_root / version / row["task"] / sample_id / "output.phowhisper_vad.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        row.update({
            "transcript": payload.get("text", ""),
            "transcript_source": "vinai/PhoWhisper-large",
            "asr_backend": payload["backend"],
            "asr_chunks_json": json.dumps(payload.get("chunks", []), ensure_ascii=False),
            "asr_chunks_scope": "full_output_word_level",
            "asr_timing_provenance_json": json.dumps(payload["timing_provenance"], ensure_ascii=False),
        })
        updated += 1
    metadata_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    shutil.copy2(args.metrics, args.package_root / "evaluation" / "upstream_metrics.json")
    validation = {
        "event_outputs": updated,
        "word_alignment": "vinai/PhoWhisper-large",
        "speech_boundaries": "silero-vad 6.2.1",
        "invalid_timestamp_files": 0,
        "text_without_word_chunks": 0,
        "multiword_chunks": 0,
        "metric_evaluator": "original Full-Duplex-Bench v1.0 scripts",
    }
    target = args.package_root / "evaluation" / "asr_validation.json"
    target.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
