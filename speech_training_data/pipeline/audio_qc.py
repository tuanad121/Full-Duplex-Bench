#!/usr/bin/env python3
"""Automated structural and timing QC for rendered conversations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from schema import read_jsonl, write_jsonl


def check(row: dict, expected_sr: int, reaction: float) -> list[str]:
    errors = []
    tracks = {}
    infos = {}
    for name, path in row["files"].items():
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
            tracks[name], infos[name] = audio, (sr, len(audio))
            if audio.ndim != 1:
                errors.append(f"{name}:not_mono")
            if sr != expected_sr:
                errors.append(f"{name}:sample_rate")
            if not np.isfinite(audio).all():
                errors.append(f"{name}:non_finite")
            if float(np.max(np.abs(audio))) > 1.0:
                errors.append(f"{name}:clipping")
            if name != "mixture" and float(np.max(np.abs(audio))) < 1e-5:
                errors.append(f"{name}:empty")
        except Exception as exc:
            errors.append(f"{name}:read_error:{type(exc).__name__}")
    if len({value for value in infos.values()}) > 1:
        errors.append("tracks_not_synchronized")
    for event in row["events"]:
        if event["end_sec"] <= event["start_sec"]:
            errors.append(f"round_{event['round']}:invalid_event_interval")
            continue
        sr = infos.get("user", (expected_sr, 0))[0]
        start = round(event["start_sec"] * sr)
        end = round(event["end_sec"] * sr)
        user_event = tracks.get("user", np.zeros(0))[start:end]
        if not len(user_event) or float(np.sqrt(np.mean(np.square(user_event)))) < 1e-5:
            errors.append(f"round_{event['round']}:missing_user_event_audio")
        if event["type"] == "backchannel":
            matching = next(turn for turn in row["timeline"] if turn["round"] == event["round"])
            if matching["agent"][1] <= event["start_sec"]:
                errors.append(f"round_{event['round']}:agent_not_active_during_backchannel")
            agent_overlap = tracks.get("agent", np.zeros(0))[start:end]
            if not len(agent_overlap) or float(np.sqrt(np.mean(np.square(agent_overlap)))) < 1e-5:
                errors.append(f"round_{event['round']}:missing_agent_backchannel_overlap")
        else:
            stop = event["agent_stop_target_sec"]
            if stop is None or abs((stop - event["start_sec"]) - reaction) > 0.03:
                errors.append(f"round_{event['round']}:reaction_latency")
            elif end > round(stop * sr):
                post_stop = tracks.get("agent", np.zeros(0))[round(stop * sr):end]
                if len(post_stop) and float(np.max(np.abs(post_stop))) > 1e-4:
                    errors.append(f"round_{event['round']}:agent_active_after_stop")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    rows = read_jsonl(args.input)
    accepted, rejected, audits = [], [], []
    for row in rows:
        errors = check(row, int(cfg["sample_rate"]), float(cfg["reaction_seconds"]))
        audit = {"conversation_id": row["conversation"]["conversation_id"], "errors": errors, "decision": "reject" if errors else "accept"}
        audits.append(audit)
        (rejected if errors else accepted).append(row)
    write_jsonl(args.output / "accepted.jsonl", accepted)
    write_jsonl(args.output / "rejected.jsonl", rejected)
    write_jsonl(args.output / "audio_qc.jsonl", audits)
    summary = {"input": len(rows), "accepted": len(accepted), "rejected": len(rejected)}
    (args.output / "audio_qc_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
