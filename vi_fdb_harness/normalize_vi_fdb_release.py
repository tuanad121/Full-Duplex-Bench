#!/usr/bin/env python3
"""Flatten inherited English suite versions into the Vi-FDB v1.0 taxonomy."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


TASK_MAP = {
    ("v1_0", "backchannel"): "backchannel",
    ("v1_0", "pause_handling"): "pause_handling",
    ("v1_0", "smooth_turn_taking"): "smooth_turn_taking",
    ("v1_0", "user_interruption"): "user_interruption",
    ("v1_5", "background_speech"): "background_speech",
    ("v1_5", "talking_to_other"): "talking_to_other",
    ("v1_5", "user_backchannel"): "user_backchannel",
    ("v1_5", "user_interruption"): "user_interruption_counterfactual",
}


def rewrite_metadata(task_dir: Path, task: str) -> None:
    for path in task_dir.glob("*/metadata.json"):
        obj = json.loads(path.read_text(encoding="utf-8"))
        obj["dataset_version"] = "1.0"
        obj["task"] = task
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_split(split: Path) -> None:
    for (suite, old_task), new_task in TASK_MAP.items():
        source = split / suite / old_task
        destination = split / new_task
        if source.exists():
            if destination.exists():
                raise FileExistsError(destination)
            shutil.move(str(source), str(destination))
        if not destination.exists():
            raise FileNotFoundError(destination)
        rewrite_metadata(destination, new_task)

    for suite in ("v1_0", "v1_5"):
        directory = split / suite
        if directory.exists():
            directory.rmdir()

    manifest_path = split / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest:
        suite = row.pop("version").replace(".", "_")
        old_task = row["task"]
        new_task = TASK_MAP[(suite, old_task)]
        row["dataset_version"] = "1.0"
        row["task"] = new_task
        for field in ("input", "clean_input", "metadata"):
            if row.get(field):
                parts = Path(row[field]).parts
                row[field] = str(Path(new_task, *parts[2:]))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for name in ("pilot_160", "expansion_240"):
        normalize_split(args.root / "data" / name)
    print("Normalized Vi-FDB v1.0 release taxonomy")


if __name__ == "__main__":
    main()
