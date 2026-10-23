#!/usr/bin/env python3
"""Render atomic utterances into synchronized user, agent, and mixture tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from schema import EVENT_ACTIONS, read_jsonl, write_jsonl


def read(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def fade_cut(audio: np.ndarray, stop: int, fade: int) -> np.ndarray:
    out = audio[:min(max(stop, 1), len(audio))].copy()
    count = min(fade, len(out))
    if count:
        out[-count:] *= np.linspace(1.0, 0.0, count, dtype=np.float32)
    return out


def ensure(track: np.ndarray, frames: int) -> np.ndarray:
    return np.pad(track, (0, max(0, frames - len(track))))


def put(track: np.ndarray, audio: np.ndarray, start: int) -> np.ndarray:
    track = ensure(track, start + len(audio))
    track[start:start + len(audio)] += audio
    return track


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    rows = read_jsonl(args.input)
    rendered = []
    for ci, row in enumerate(rows, 1):
        conversation, clips = row["conversation"], row["clips"]
        cid = conversation["conversation_id"]
        user_track = np.zeros(0, np.float32)
        agent_track = np.zeros(0, np.float32)
        cursor = 0
        sr = None
        timeline = []
        events = []
        for clip in clips:
            user, user_sr = read(clip["user"]["file"])
            agent, agent_sr = read(clip["agent"]["file"])
            sr = sr or user_sr
            if user_sr != sr or agent_sr != sr:
                raise ValueError(f"{cid}: sample-rate mismatch")
            user_start = cursor
            user_track = put(user_track, user, user_start)
            agent_start = user_start + len(user) + round(float(cfg["turn_gap_seconds"]) * sr)
            event = clip["event"]
            event_start = event_end = agent_stop = after_start = None
            rendered_agent = agent
            if event:
                event_audio, event_sr = read(event["audio"]["file"])
                if event_sr != sr:
                    raise ValueError(f"{cid}: event sample-rate mismatch")
                event_start = agent_start + round(len(agent) * float(event["overlap_fraction"]))
                event_end = event_start + len(event_audio)
                user_track = put(user_track, event_audio, event_start)
                if event["type"] != "backchannel":
                    agent_stop = event_start + round(float(cfg["reaction_seconds"]) * sr)
                    rendered_agent = fade_cut(agent, agent_stop - agent_start, round(0.05 * sr))
            agent_track = put(agent_track, rendered_agent, agent_start)
            end = max(user_start + len(user), agent_start + len(rendered_agent), event_end or 0)
            if event and event["agent_after_audio"]:
                after, after_sr = read(event["agent_after_audio"]["file"])
                if after_sr != sr:
                    raise ValueError(f"{cid}: post-event sample-rate mismatch")
                after_start = event_end + round(float(cfg["post_event_gap_seconds"]) * sr)
                agent_track = put(agent_track, after, after_start)
                end = max(end, after_start + len(after))
            timeline.append({
                "round": clip["round"],
                "user": [round(user_start / sr, 4), round((user_start + len(user)) / sr, 4)],
                "agent": [round(agent_start / sr, 4), round((agent_start + len(rendered_agent)) / sr, 4)],
                "event": None if not event else [round(event_start / sr, 4), round(event_end / sr, 4)],
                "agent_stop_target_sec": None if agent_stop is None else round(agent_stop / sr, 4),
                "agent_after": None if after_start is None else [round(after_start / sr, 4), round(end / sr, 4)],
            })
            if event:
                events.append({
                    "round": clip["round"], "type": event["type"], "text": event["text"],
                    "expected_action": EVENT_ACTIONS[event["type"]],
                    "start_sec": round(event_start / sr, 4), "end_sec": round(event_end / sr, 4),
                    "agent_stop_target_sec": None if agent_stop is None else round(agent_stop / sr, 4),
                })
            cursor = end + round(float(cfg["turn_gap_seconds"]) * sr)
        total = cursor + round(float(cfg["tail_seconds"]) * sr)
        user_track, agent_track = ensure(user_track, total), ensure(agent_track, total)
        peak = max(float(np.max(np.abs(user_track))), float(np.max(np.abs(agent_track))), 1e-6)
        scale = min(0.94 / peak, 1.0)
        user_track *= scale
        agent_track *= scale
        mixture = np.clip(user_track + agent_track, -1.0, 1.0)
        out = args.output / cid
        out.mkdir(parents=True, exist_ok=True)
        for name, audio in (("user.wav", user_track), ("agent.wav", agent_track), ("mixture.wav", mixture)):
            sf.write(out / name, audio, sr)
        metadata = {
            "conversation": conversation, "sample_rate": sr, "duration_sec": round(total / sr, 4),
            "timeline": timeline, "events": events,
            "files": {"user": str(out / "user.wav"), "agent": str(out / "agent.wav"), "mixture": str(out / "mixture.wav")},
        }
        (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        rendered.append(metadata)
        print(f"[{ci}/{len(rows)}] {cid}: {total / sr:.1f}s", flush=True)
    write_jsonl(args.output / "rendered_manifest.jsonl", rendered)


if __name__ == "__main__":
    main()
