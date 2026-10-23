#!/usr/bin/env python3
"""Extract fixed-duration causal training windows from validated conversations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from schema import read_jsonl, write_jsonl


def slice_pad(audio: np.ndarray, start: int, frames: int) -> np.ndarray:
    chunk = audio[max(start, 0):max(start, 0) + frames]
    if start < 0:
        chunk = np.pad(chunk, (-start, 0))
    return np.pad(chunk[:frames], (0, max(0, frames - len(chunk))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    rows = read_jsonl(args.input)
    seconds = float(cfg["window_seconds"])
    event_position = float(cfg["window_event_position_seconds"])
    manifest, skipped = [], []
    for row in rows:
        cid = row["conversation"]["conversation_id"]
        tracks = {name: sf.read(path, dtype="float32", always_2d=False) for name, path in row["files"].items()}
        sample_rates = {sr for _, sr in tracks.values()}
        if len(sample_rates) != 1:
            raise ValueError(f"{cid}: sample-rate mismatch")
        sr = sample_rates.pop()
        frames = round(seconds * sr)
        onsets = [event["start_sec"] for event in row["events"]]
        for event_index, event in enumerate(row["events"], 1):
            start_sec = max(0.0, event["start_sec"] - event_position)
            max_start = max(0.0, row["duration_sec"] - seconds)
            start_sec = min(start_sec, max_start)
            end_sec = start_sec + seconds
            other = [onset for onset in onsets if onset != event["start_sec"] and start_sec <= onset < end_sec]
            if other:
                skipped.append({"conversation_id": cid, "round": event["round"], "reason": "multiple_events_in_window", "other_onsets": other})
                continue
            start_frame = round(start_sec * sr)
            window_id = f"{cid}_e{event_index:02d}_{event['type']}"
            out = args.output / window_id
            out.mkdir(parents=True, exist_ok=True)
            for name, (audio, _) in tracks.items():
                sf.write(out / f"{name}.wav", slice_pad(audio, start_frame, frames), sr)
            meta = {
                "window_id": window_id, "conversation_id": cid, "source_start_sec": round(start_sec, 4),
                "duration_sec": seconds, "sample_rate": sr,
                "event": {**event, "start_sec": round(event["start_sec"] - start_sec, 4), "end_sec": round(event["end_sec"] - start_sec, 4),
                          "agent_stop_target_sec": None if event["agent_stop_target_sec"] is None else round(event["agent_stop_target_sec"] - start_sec, 4)},
                "history_rounds": row["conversation"]["rounds"][:event["round"]],
                "files": {name: str(out / f"{name}.wav") for name in tracks},
            }
            (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest.append(meta)
    write_jsonl(args.output / "windows.jsonl", manifest)
    write_jsonl(args.output / "skipped.jsonl", skipped)
    summary = {"conversations": len(rows), "windows": len(manifest), "skipped": len(skipped), "window_seconds": seconds}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
