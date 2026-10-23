#!/usr/bin/env python3
"""
Upload Vietnamese full-duplex training data to HuggingFace.

Text upload (run first, no cluster needed):
  /tmp/hf_upload_env/bin/python3 fd_data_creation/upload_to_hf.py --text

Audio upload (after downloading from cluster):
  /tmp/hf_upload_env/bin/python3 fd_data_creation/upload_to_hf.py --audio \
    --audio-root outputs/full_duplex_training/production_audio_v1/03_utterances

Usage requires `huggingface-cli login` first.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

REPO_ID = "tuanamz/vi-full-duplex-conversations"
REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_CARD = """\
---
language:
- vi
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
- text-to-speech
tags:
- full-duplex
- vietnamese
- conversation
- backchannel
- interruption
pretty_name: Vietnamese Full-Duplex Conversations
size_categories:
- 1K<n<10K
---

# Vietnamese Full-Duplex Conversations

Training data for Vietnamese full-duplex speech models. Each conversation contains
multi-turn dialogue with annotated speech events (backchannels, real interruptions,
ambiguous interruptions) for training models that can listen and speak simultaneously.

## Data

- **5,421 text conversations** (v1–v7 batches, LLM-reviewed)
- **~1,489 synthesized audio conversations** (VieNeu-TTS v3turbo, 48 kHz)
- Two speaker roles: user (nữ, Northern) and agent (nam, Southern)
- 6–10 turns per conversation, 3 event types

## Text format

Each JSONL line:
```json
{
  "conversation_id": "vi_fd_001001_sa",
  "topic": "đặt bàn và chọn món",
  "style": "thân mật tự nhiên",
  "rounds": [
    {
      "user": "...",
      "agent": "...",
      "event": {
        "type": "backchannel",
        "text": "Vâng.",
        "agent_after": "",
        "overlap_fraction": 0.4
      }
    }
  ]
}
```

## Audio structure (per conversation)

```
vi_fd_001001_sa/
  utterance_manifest.json   # timing + file paths
  user_00.wav               # user turn 0
  agent_00.wav              # agent turn 0
  event_00.wav              # overlap event (if present)
  agent_after_00.wav        # agent continuation after event
  ...
```
"""

def upload_text(api) -> None:
    from huggingface_hub import CommitOperationAdd
    text_dir = REPO_ROOT / "outputs" / "full_duplex_training"
    ops = []

    # Per-batch accepted files
    for v in range(1, 9):
        src = text_dir / f"production_text_v{v}" / "qc" / "accepted.jsonl"
        if src.exists():
            ops.append(CommitOperationAdd(
                path_in_repo=f"data/text/v{v}_accepted.jsonl",
                path_or_fileobj=str(src),
            ))
            print(f"  queued v{v} ({sum(1 for _ in src.open())} conversations)")

    # Merged file
    merged = text_dir / "accepted_for_synthesis.jsonl"
    if merged.exists():
        ops.append(CommitOperationAdd(
            path_in_repo="data/text/all_accepted.jsonl",
            path_or_fileobj=str(merged),
        ))
        print(f"  queued all_accepted ({sum(1 for _ in merged.open())} conversations)")

    print(f"Uploading {len(ops)} text files...")
    api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        operations=ops,
        commit_message="Add text conversations v1–v7 (5421 accepted)",
    )
    print("Text upload complete.")


def upload_audio(api, audio_root: Path) -> None:
    """Upload audio folder directly to HF — no local tar needed."""
    conv_dirs = sorted([d for d in audio_root.iterdir() if d.is_dir()])
    print(f"Found {len(conv_dirs)} conversation directories.", flush=True)
    total_wav = sum(1 for d in conv_dirs for _ in d.glob("*.wav"))
    print(f"Total WAV files: {total_wav}", flush=True)

    # upload_large_folder streams files to HF without creating a local archive.
    # Resumable: re-running skips already-uploaded files.
    # Audio dirs land at repo root (v0.36.0 doesn't support path_in_repo here).
    api.upload_large_folder(
        folder_path=str(audio_root),
        repo_id=REPO_ID,
        repo_type="dataset",
        num_workers=4,
        print_report_every=30,
    )
    print("Audio upload complete.", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", action="store_true")
    p.add_argument("--audio", action="store_true")
    p.add_argument("--audio-root", type=Path,
                   default=REPO_ROOT / "outputs/full_duplex_training/production_audio_v1/03_utterances")
    args = p.parse_args()

    if not args.text and not args.audio:
        p.error("Specify --text and/or --audio")

    from huggingface_hub import HfApi
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=True, exist_ok=True)
        print(f"Repo: https://huggingface.co/datasets/{REPO_ID}")
    except Exception as e:
        print(f"Repo create: {e}")

    # Upload dataset card
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Add dataset card",
    )

    if args.text:
        upload_text(api)

    if args.audio:
        if not args.audio_root.exists():
            raise SystemExit(f"Audio root not found: {args.audio_root}")
        upload_audio(api, args.audio_root)


if __name__ == "__main__":
    main()
