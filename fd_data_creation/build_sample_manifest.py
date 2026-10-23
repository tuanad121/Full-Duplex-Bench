#!/usr/bin/env python3
"""Build an utterance_manifest.jsonl from downloaded audio folders + accepted text.

Use this when you want a vibe-check HTML from a partial synthesis download
instead of waiting for the full utterance_manifest.jsonl from the cluster.

Usage:
    external/VieNeu-TTS/.venv/bin/python3 fd_data_creation/build_sample_manifest.py \
        --audio  outputs/full_duplex_training/vibe_audio_sample/03_utterances \
        --text   outputs/full_duplex_training/accepted_for_synthesis.jsonl \
        --output outputs/full_duplex_training/vibe_audio_sample/utterance_manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf


def wav_info(path: Path, text: str = "") -> dict | None:
    if not path.exists():
        return None
    info = sf.info(path)
    return {
        "file": str(path),
        "frames": info.frames,
        "sample_rate": info.samplerate,
        "duration_sec": round(info.duration, 4),
        "text": text,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", type=Path, required=True, help="03_utterances directory with downloaded conversation folders")
    p.add_argument("--text", type=Path, required=True, help="accepted_for_synthesis.jsonl (or any accepted.jsonl)")
    p.add_argument("--output", type=Path, required=True, help="output utterance_manifest.jsonl")
    args = p.parse_args()

    convs: dict[str, dict] = {}
    for line in args.text.read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            convs[c["conversation_id"]] = c

    manifest = []
    skipped = 0
    for cid_dir in sorted(args.audio.iterdir()):
        if not cid_dir.is_dir():
            continue
        cid = cid_dir.name
        if cid not in convs:
            print(f"SKIP {cid}: not in text source")
            skipped += 1
            continue
        conv = convs[cid]
        source_dir = cid_dir / "source"
        clips = []
        for ri, turn in enumerate(conv["rounds"], 1):
            u = wav_info(source_dir / f"r{ri:02d}_user.wav", turn["user"])
            a = wav_info(source_dir / f"r{ri:02d}_agent.wav", turn["agent"])
            if not u or not a:
                print(f"  missing wav {cid} r{ri}")
                continue
            item: dict = {"round": ri, "user": u, "agent": a, "event": None}
            ev = turn.get("event")
            if ev:
                ev_wav = wav_info(source_dir / f"r{ri:02d}_event.wav", ev["text"])
                after_text = ev.get("agent_after", "")
                after_wav = (
                    wav_info(source_dir / f"r{ri:02d}_agent_after.wav", after_text)
                    if after_text else None
                )
                if ev_wav:
                    item["event"] = {**ev, "audio": ev_wav, "agent_after_audio": after_wav}
            clips.append(item)
        manifest.append({"conversation": conv, "clips": clips})
        print(f"  {cid}: {len(clips)} rounds")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {len(manifest)} conversations → {args.output}  (skipped {skipped})")


if __name__ == "__main__":
    main()
