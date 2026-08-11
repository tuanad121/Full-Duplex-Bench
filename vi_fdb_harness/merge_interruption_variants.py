#!/usr/bin/env python3
"""Merge interruption evaluation variants under one Vi-FDB task."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def update_metadata(path: Path, variant: str) -> None:
    obj = json.loads(path.read_text(encoding="utf-8"))
    obj["dataset_version"] = "1.0"
    obj["task"] = "user_interruption"
    obj["evaluation_variant"] = variant
    obj["has_clean_control"] = variant == "counterfactual"
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def merge(split: Path) -> None:
    standard = split / "user_interruption"
    counterfactual = split / "user_interruption_counterfactual"

    for sample in sorted(path for path in standard.iterdir() if path.is_dir()):
        destination = standard / f"standard_{sample.name}"
        if not destination.exists():
            sample.rename(destination)
        update_metadata(destination / "metadata.json", "standard")

    for sample in sorted(path for path in counterfactual.iterdir() if path.is_dir()):
        destination = standard / f"counterfactual_{sample.name}"
        shutil.move(str(sample), str(destination))
        update_metadata(destination / "metadata.json", "counterfactual")
    counterfactual.rmdir()

    manifest_path = split / "manifest.json"
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in rows:
        if row["task"] not in {"user_interruption", "user_interruption_counterfactual"}:
            continue
        variant = "counterfactual" if row["task"].endswith("counterfactual") else "standard"
        old_id = row["id"]
        new_id = f"{variant}_{old_id}"
        row["task"] = "user_interruption"
        row["evaluation_variant"] = variant
        row["has_clean_control"] = variant == "counterfactual"
        row["id"] = new_id
        for field in ("input", "clean_input", "metadata"):
            if row.get(field):
                parts = Path(row[field]).parts
                row[field] = str(Path("user_interruption", new_id, *parts[2:]))
    manifest_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for name in ("pilot_160", "expansion_240"):
        merge(args.root / "data" / name)
    print("Merged user-interruption variants")


if __name__ == "__main__":
    main()
