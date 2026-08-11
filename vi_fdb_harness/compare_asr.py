#!/usr/bin/env python3
"""Compare two normalized Vi-FDB ASR outputs and flag arbitration cases."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    return " ".join(re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE).split())


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def onset(data):
    chunks = data.get("chunks") or []
    return chunks[0]["timestamp"][0] if chunks else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--primary", default="chunkformer")
    parser.add_argument("--verifier", default="whisper")
    parser.add_argument("--similarity-threshold", type=float, default=0.75)
    parser.add_argument("--onset-tolerance", type=float, default=0.30)
    args = parser.parse_args()
    records = []
    for primary_path in sorted(args.root.rglob(f"*.{args.primary}.json")):
        stem = primary_path.name[: -len(f".{args.primary}.json")]
        verifier_path = primary_path.with_name(f"{stem}.{args.verifier}.json")
        if not verifier_path.exists():
            continue
        primary, verifier = read(primary_path), read(verifier_path)
        similarity = SequenceMatcher(None, normalize(primary.get("text", "")), normalize(verifier.get("text", ""))).ratio()
        p_onset, v_onset = onset(primary), onset(verifier)
        onset_delta = None if p_onset is None or v_onset is None else abs(p_onset - v_onset)
        flagged = similarity < args.similarity_threshold or onset_delta is None or onset_delta > args.onset_tolerance
        record = {
            "audio": str(primary_path.with_name(stem)),
            "primary": args.primary,
            "verifier": args.verifier,
            "text_similarity": round(similarity, 4),
            "onset_delta": None if onset_delta is None else round(onset_delta, 4),
            "flagged": flagged,
        }
        primary_path.with_name(f"{stem}.asr_comparison.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)
    summary = {"compared": len(records), "flagged": sum(row["flagged"] for row in records), "records": records}
    destination = args.root / "asr_comparison_summary.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"compared": summary["compared"], "flagged": summary["flagged"]}, indent=2))


if __name__ == "__main__":
    main()
