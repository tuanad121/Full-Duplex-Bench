#!/usr/bin/env python3
"""Attach audio-derived speech boundaries to PhoWhisper word transcripts.

PhoWhisper supplies lexical units and word timestamps, while Silero VAD supplies
the observable speech onset/offset used by the original FDB timing metrics. This
avoids treating Whisper's occasional leading-silence timestamp as speech.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from silero_vad import get_speech_timestamps, load_silero_vad, read_audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    model = load_silero_vad()
    sources = sorted(args.root.rglob("output.phowhisper.json"))
    for index, source in enumerate(sources, 1):
        destination = source.with_name("output.phowhisper_vad.json")
        if destination.exists() and not args.overwrite:
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        waveform = read_audio(str(source.with_name("output.wav")), sampling_rate=16000)
        intervals = get_speech_timestamps(waveform, model, sampling_rate=16000, return_seconds=True)
        chunks = payload.get("chunks", [])
        if chunks and intervals:
            chunks[0]["timestamp"][0] = round(float(intervals[0]["start"]), 3)
            chunks[-1]["timestamp"][1] = round(float(intervals[-1]["end"]), 3)
        payload["backend"] = "phowhisper-large+silero-vad"
        payload["timing_provenance"] = {
            "word_alignment": "vinai/PhoWhisper-large",
            "speech_boundaries": "silero-vad",
            "speech_intervals": intervals,
        }
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[{index}/{len(sources)}] wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
