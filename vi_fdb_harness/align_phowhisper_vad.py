#!/usr/bin/env python3
"""Attach audio-derived speech boundaries without changing word timestamps.

PhoWhisper supplies lexical units and word timestamps. Silero VAD supplies
observable speech intervals as separate evidence. Mixing the two coordinate
sources inside a word interval can create impossible timestamps, so this module
never mutates ``chunks``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

def validate_word_chunks(chunks: list[dict]) -> bool:
    previous_start = -1.0
    monotonic = True
    for index, chunk in enumerate(chunks):
        timestamp = chunk.get("timestamp")
        if not isinstance(timestamp, list) or len(timestamp) != 2:
            raise ValueError(f"chunk {index} has no [start, end] timestamp")
        start, end = timestamp
        if start is None or end is None or start > end:
            raise ValueError(f"chunk {index} has invalid timestamp: {timestamp}")
        if start < previous_start:
            monotonic = False
        previous_start = start
    return monotonic


def validate_speech_intervals(intervals: list[dict]) -> None:
    previous_end = -1.0
    for index, interval in enumerate(intervals):
        start, end = float(interval["start"]), float(interval["end"])
        if start > end or start < previous_end:
            raise ValueError(f"VAD interval {index} is invalid: {interval}")
        previous_end = end


def main() -> None:
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

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
        word_timestamps_monotonic = validate_word_chunks(chunks)
        validate_speech_intervals(intervals)
        payload["backend"] = "phowhisper-large+silero-vad"
        payload["timing_provenance"] = {
            "word_alignment": "vinai/PhoWhisper-large",
            "speech_boundaries": "silero-vad",
            "speech_intervals": intervals,
            "word_timestamps_mutated": False,
            "word_timestamps_monotonic": word_timestamps_monotonic,
        }
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[{index}/{len(sources)}] wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
