#!/usr/bin/env python3
"""Vietnamese ASR adapters producing one normalized timestamp schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def write_result(audio: Path, backend: str, result: dict) -> Path:
    destination = audio.with_suffix(f".{backend}.json")
    payload = {
        "schema_version": 1,
        "backend": backend,
        "audio": audio.name,
        "text": result.get("text", "").strip(),
        "chunks": result.get("chunks", []),
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def timestamp_seconds(value):
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().strip("[]")
    parts = text.split(":")
    if len(parts) == 4:  # ChunkFormer runtime: HH:MM:SS:milliseconds
        hours, minutes, seconds, milliseconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(text)


def normalized_chunk(text, start, end):
    return {"text": str(text).strip(), "timestamp": [round(timestamp_seconds(start), 3), round(timestamp_seconds(end), 3)]}


def transcribe_whisper(model, audio: Path) -> dict:
    result = model.transcribe(str(audio), language="vi", task="transcribe", word_timestamps=True, verbose=False)
    chunks = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            chunks.append(normalized_chunk(word["word"], word["start"], word["end"]))
    return {"text": result.get("text", ""), "chunks": chunks}


def transcribe_phowhisper(pipe, audio: Path) -> dict:
    result = pipe(str(audio), return_timestamps="word", generate_kwargs={"language": "vi", "task": "transcribe"})
    chunks = []
    for chunk in result.get("chunks", []):
        timestamps = chunk.get("timestamp") or [None, None]
        if timestamps[0] is not None and timestamps[1] is not None:
            chunks.append(normalized_chunk(chunk.get("text", ""), timestamps[0], timestamps[1]))
    return {"text": result.get("text", ""), "chunks": chunks}


TIME_PATTERN = re.compile(r"\[?(\d+):(\d+):(\d+(?:\.\d+)?)\]?\s*-\s*\[?(\d+):(\d+):(\d+(?:\.\d+)?)\]?:\s*(.*)")


def hms(hours, minutes, seconds):
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def normalize_chunkformer(raw) -> dict:
    """Accept current ChunkFormer dict/list output and its documented text form."""
    if isinstance(raw, dict):
        text = raw.get("text") or raw.get("transcription") or ""
        source_chunks = raw.get("chunks") or raw.get("segments") or raw.get("timestamps") or []
        chunks = []
        for item in source_chunks:
            stamp = item.get("timestamp") or [item.get("start"), item.get("end")]
            if stamp[0] is not None and stamp[1] is not None:
                chunks.append(normalized_chunk(item.get("text") or item.get("word") or item.get("decode") or "", stamp[0], stamp[1]))
        return {"text": text or " ".join(x["text"] for x in chunks), "chunks": chunks}
    if isinstance(raw, list):
        return normalize_chunkformer({"segments": raw})
    lines = str(raw).splitlines()
    chunks = []
    for line in lines:
        match = TIME_PATTERN.search(line.strip())
        if match:
            chunks.append(normalized_chunk(match.group(7), hms(*match.group(1, 2, 3)), hms(*match.group(4, 5, 6))))
    return {"text": " ".join(x["text"] for x in chunks) if chunks else str(raw).strip(), "chunks": chunks}


def discover(root: Path, names: list[str]) -> list[Path]:
    wanted = set(names)
    return sorted(path for path in root.rglob("*.wav") if path.name in wanted)


def main() -> None:
    parser = argparse.ArgumentParser(prog="vi-fdb-asr")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backend", choices=["chunkformer", "whisper", "phowhisper"], required=True)
    parser.add_argument("--audio-names", nargs="+", default=["output.wav", "clean_output.wav"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    audio_files = discover(args.root, args.audio_names)
    if args.limit:
        audio_files = audio_files[: args.limit]
    if args.backend == "chunkformer":
        from chunkformer import ChunkFormerModel
        model = ChunkFormerModel.from_pretrained("khanhld/chunkformer-ctc-large-vie")
        infer = lambda path: normalize_chunkformer(model.endless_decode(
            audio_path=str(path), chunk_size=64, left_context_size=128,
            right_context_size=128, total_batch_duration=14400,
            return_timestamps=True,
        ))
    elif args.backend == "whisper":
        import whisper
        model = whisper.load_model("large-v3")
        infer = lambda path: transcribe_whisper(model, path)
    else:
        import torch
        from transformers import pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model="vinai/PhoWhisper-large",
            device=0 if torch.cuda.is_available() else -1,
        )
        infer = lambda path: transcribe_phowhisper(pipe, path)
    for index, audio in enumerate(audio_files, 1):
        destination = audio.with_suffix(f".{args.backend}.json")
        if destination.exists() and not args.overwrite:
            print(f"[{index}/{len(audio_files)}] skip {destination}", flush=True)
            continue
        try:
            result = infer(audio)
            destination = write_result(audio, args.backend, result)
            print(f"[{index}/{len(audio_files)}] wrote {destination}", flush=True)
        except Exception as exc:
            print(f"[{index}/{len(audio_files)}] FAILED {audio}: {exc}", flush=True)


if __name__ == "__main__":
    main()
