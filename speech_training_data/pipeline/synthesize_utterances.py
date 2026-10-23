#!/usr/bin/env python3
"""Synthesize atomic user and agent utterances with VieNeu-TTS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from vieneu import Vieneu

from schema import read_jsonl, write_jsonl


def generate(tts: Vieneu, text: str, ref_codes: np.ndarray, output: Path, temperature: float) -> dict:
    """Synthesize one utterance. ref_codes must be pre-computed via tts.encode_reference()."""
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        audio = np.asarray(
            tts.infer(text, ref_codes=ref_codes, temperature=temperature, show_progress=False),
            dtype=np.float32,
        )
        sf.write(output, audio, int(tts.sample_rate))
    info = sf.info(output)
    return {"file": str(output), "frames": info.frames, "sample_rate": info.samplerate, "duration_sec": round(info.duration, 4), "text": text}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vieneu-root", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=None,
                        help="TTS temperature (default: synthesis_temperature from config, or 0.5)")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.temperature is None:
        args.temperature = float(config.get("synthesis_temperature", 0.5))
    rows = read_jsonl(args.input)
    refs = {name: args.vieneu_root / path for name, path in config["voices"].items()}
    missing = [str(path) for path in refs.values() if not path.exists()]
    if missing:
        raise SystemExit(f"missing voice references: {missing}")
    tts = Vieneu(mode="v3turbo", backend="pytorch")
    # Encode each reference voice ONCE so the exact same speaker embedding is
    # reused for every utterance. Re-encoding per call causes subtle speaker
    # drift even though the WAV is the same.
    ref_codes = {name: tts.encode_reference(path) for name, path in refs.items()}
    print(f"Encoded {len(ref_codes)} reference voices.", flush=True)
    manifest = []
    for ci, conversation in enumerate(rows, 1):
        cid = conversation["conversation_id"]
        clips = []
        for ri, turn in enumerate(conversation["rounds"], 1):
            root = args.output / cid / "source"
            item = {
                "round": ri,
                "user": generate(tts, turn["user"], ref_codes["user"], root / f"r{ri:02d}_user.wav", args.temperature),
                "agent": generate(tts, turn["agent"], ref_codes["agent"], root / f"r{ri:02d}_agent.wav", args.temperature),
                "event": None,
            }
            event = turn["event"]
            if event:
                item["event"] = {
                    **event,
                    "audio": generate(tts, event["text"], ref_codes["user"], root / f"r{ri:02d}_event.wav", args.temperature),
                    "agent_after_audio": None,
                }
                if event["agent_after"]:
                    item["event"]["agent_after_audio"] = generate(
                        tts, event["agent_after"], ref_codes["agent"], root / f"r{ri:02d}_agent_after.wav", args.temperature
                    )
            clips.append(item)
        manifest.append({"conversation": conversation, "clips": clips})
        print(f"[{ci}/{len(rows)}] {cid}: {len(clips)} rounds", flush=True)
    write_jsonl(args.output / "utterance_manifest.jsonl", manifest)


if __name__ == "__main__":
    main()
