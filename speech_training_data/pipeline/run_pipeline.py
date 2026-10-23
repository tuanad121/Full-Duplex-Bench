#!/usr/bin/env python3
"""Run or print the full Vietnamese full-duplex data-preparation pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


STAGES = ["generate", "text-qc", "synthesize", "render", "audio-qc", "windows", "vibe"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--vieneu-root", type=Path, required=True)
    parser.add_argument("--from-stage", choices=STAGES, default="generate")
    parser.add_argument("--to-stage", choices=STAGES, default="vibe")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    py = sys.executable
    root = args.root.resolve()
    commands = {
        "generate": [py, str(here / "generate_dialogues.py"), "--config", str(args.config), "--output", str(root / "01_generation")],
        "text-qc": [py, str(here / "review_text.py"), "--config", str(args.config), "--input", str(root / "01_generation/generated.jsonl"), "--output", str(root / "02_text_qc")],
        "synthesize": [py, str(here / "synthesize_utterances.py"), "--config", str(args.config), "--input", str(root / "02_text_qc/accepted.jsonl"), "--output", str(root / "03_utterances"), "--vieneu-root", str(args.vieneu_root)],
        "render": [py, str(here / "render_timelines.py"), "--config", str(args.config), "--input", str(root / "03_utterances/utterance_manifest.jsonl"), "--output", str(root / "04_rendered")],
        "audio-qc": [py, str(here / "audio_qc.py"), "--config", str(args.config), "--input", str(root / "04_rendered/rendered_manifest.jsonl"), "--output", str(root / "05_audio_qc")],
        "windows": [py, str(here / "extract_windows.py"), "--config", str(args.config), "--input", str(root / "05_audio_qc/accepted.jsonl"), "--output", str(root / "06_windows")],
        "vibe": [py, str(here / "render_vibe.py"), "--input", str(root / "05_audio_qc/accepted.jsonl"), "--output", str(root / "vibe_check.html")],
    }
    first, last = STAGES.index(args.from_stage), STAGES.index(args.to_stage)
    if first > last:
        raise SystemExit("--from-stage must precede --to-stage")
    for stage in STAGES[first:last + 1]:
        command = commands[stage]
        print(f"[{stage}]", subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            env = os.environ.copy()
            subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
