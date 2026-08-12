#!/usr/bin/env python3
"""Export a Vi-FDB system run in the directory layout expected by English FDB.

The exporter does not alter annotations or metric logic. It links the Vietnamese
input files and maps the selected timestamped ASR artifact to the upstream
``output.json`` filename, slicing only where the original FDB evaluator expects
an event-relative transcript.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


V1_TASK_FILES = {
    "pause_handling": ("pause.json",),
    "smooth_turn_taking": ("turn_taking.json",),
    "user_interruption": ("interrupt.json",),
}


def run_sample_id(task: str, source_sample_id: str) -> str:
    """Map normalized source folders to the run layout used by the harness."""
    if task == "user_interruption" and not source_sample_id.startswith("standard_"):
        return f"standard_{source_sample_id}"
    return source_sample_id


def link(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(source.resolve())


def write_event_output(source: Path, destination: Path, task: str, annotation: Path) -> None:
    """Write the event-relative output segment consumed by upstream FDB.

    Our review artifacts retain the complete aligned assistant stream. Upstream
    pause files contain only premature speech, while interruption files contain
    the response after the interrupting turn. Smooth-turn output needs no slice.
    """
    payload = json.loads(source.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    intervals = payload.get("timing_provenance", {}).get("speech_intervals", [])
    boundary = json.loads(annotation.read_text(encoding="utf-8"))[0]["timestamp"]
    if task == "pause_handling":
        # Premature assistant speech that begins before the user resumes.
        chunks = [chunk for chunk in chunks if chunk["timestamp"][0] < boundary[1]]
        intervals = [interval for interval in intervals if interval["start"] < boundary[1]]
    elif task == "user_interruption":
        # The assistant's response after the interrupting turn has completed.
        chunks = [chunk for chunk in chunks if chunk["timestamp"][0] >= boundary[1]]
        intervals = [interval for interval in intervals if interval["end"] > boundary[1]]
    payload["chunks"] = chunks
    payload["text"] = " ".join(chunk["text"].strip() for chunk in chunks).strip()
    payload["event_slice"] = {
        "task": task,
        "boundary": boundary,
        "source": str(source.resolve()),
    }
    payload["evaluation_speech_intervals"] = intervals
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export(source_data: Path, system_run: Path, output: Path, asr_backend: str) -> dict:
    manifest: dict = {
        "format": "english-fdb-evaluation-compatible",
        "source_data": str(source_data.resolve()),
        "system_run": str(system_run.resolve()),
        "mapping": {
            f"output.{asr_backend}.json": "event-sliced output.json",
            "annotation_files": "unchanged",
        },
        "tasks": {},
    }

    for task, annotations in V1_TASK_FILES.items():
        source_task = source_data / "v1_0" / task
        run_task = system_run / "v1_0" / task
        target_task = output / "v1_0" / task
        exported = 0
        for sample_dir in sorted(path for path in source_task.iterdir() if path.is_dir()):
            run_sample = run_task / run_sample_id(task, sample_dir.name)
            target_sample = target_task / sample_dir.name
            annotation_path = sample_dir / annotations[0]
            write_event_output(
                run_sample / f"output.{asr_backend}.json",
                target_sample / "output.json",
                task,
                annotation_path,
            )
            link(run_sample / "output.wav", target_sample / "output.wav")
            if (sample_dir / "input.wav").exists():
                link(sample_dir / "input.wav", target_sample / "input.wav")
            link(sample_dir / "metadata.json", target_sample / "metadata.json")
            for annotation in annotations:
                link(sample_dir / annotation, target_sample / annotation)
            exported += 1
        manifest["tasks"][f"v1_0/{task}"] = exported

    output.mkdir(parents=True, exist_ok=True)
    (output / "compat_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--system-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asr-backend", default="phowhisper_vad")
    args = parser.parse_args()
    manifest = export(args.source_data, args.system_run, args.output, args.asr_backend)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
