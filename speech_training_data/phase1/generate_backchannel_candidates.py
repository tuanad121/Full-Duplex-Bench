#!/usr/bin/env python3
"""Generate longer backchannel candidates to screen short-utterance voice drift."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

from vieneu import Vieneu


CANDIDATES = [
    "Ừ, mình hiểu rồi.",
    "Vâng, mình đang nghe đây.",
    "Dạ, đúng rồi ạ.",
    "Ừ, bạn nói tiếp đi.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vieneu-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    ref = args.vieneu_root / "src/vieneu/assets/samples/Ly (nữ miền Bắc).wav"
    if not ref.exists():
        raise SystemExit(f"missing reference: {ref}")
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ref, args.output / "source_speaker_Ly.wav")
    tts = Vieneu(mode="v3turbo", backend="pytorch")
    rows = []
    for index, text in enumerate(CANDIDATES, 1):
        filename = f"candidate_{index:02d}.wav"
        audio = tts.infer(text, ref_audio=str(ref), temperature=0.65, show_progress=False)
        tts.save(audio, str(args.output / filename))
        rows.append({"id": index, "text": text, "file": filename})
    (args.output / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cards = "".join(
        f'<article><h2>{row["id"]}. {html.escape(row["text"])}</h2>'
        f'<audio controls preload="metadata" src="{row["file"]}"></audio>'
        f'<button data-id="{row["id"]}">Use this voice</button></article>'
        for row in rows
    )
    page = f'''<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Backchannel candidates</title><style>body{{max-width:820px;margin:40px auto;padding:0 18px;background:#f3f6fb;color:#172033;font:16px/1.5 system-ui}}article,.source{{background:white;border:1px solid #dbe3ef;border-radius:14px;padding:18px;margin:14px 0}}.source{{position:sticky;top:10px;z-index:2;border-color:#8aa4cd;box-shadow:0 5px 20px #1e293b18}}.source h2{{margin-top:0}}article.selected{{border:2px solid #22c55e}}audio{{width:100%;display:block;margin:10px 0}}button{{font:inherit;padding:8px 12px;border:1px solid #b9c6d8;border-radius:8px;background:white;cursor:pointer}}</style></head><body><h1>Backchannel voice candidates</h1><p>Compare each candidate with the exact source speaker used for zero-shot cloning. Choose the cue that best preserves the speaker identity; the selection is saved locally.</p><section class="source"><h2>Source speaker — Ly (nữ miền Bắc)</h2><audio controls preload="metadata" src="source_speaker_Ly.wav"></audio></section>{cards}<script>const key='phase1_backchannel_choice';const saved=localStorage.getItem(key);if(saved)document.querySelector(`button[data-id="${{saved}}"]`)?.closest('article').classList.add('selected');document.addEventListener('click',e=>{{if(!e.target.dataset.id)return;document.querySelectorAll('article').forEach(x=>x.classList.remove('selected'));e.target.closest('article').classList.add('selected');localStorage.setItem(key,e.target.dataset.id)}});</script></body></html>'''
    (args.output / "index.html").write_text(page, encoding="utf-8")
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
