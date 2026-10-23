#!/usr/bin/env python3
"""Render before/after review of the authors' deterministic transformations."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def show(value: str) -> str:
    value = html.escape(value)
    for token, label in [
        ("&lt;|switch-to-speak|&gt;", "S-S"),
        ("&lt;|switch-to-listen|&gt;", "S-L"),
        ("&lt;|continue_speak|&gt;", "C-S"),
        ("&lt;|continue-listen|&gt;", "C-L"),
        ("&lt;|interrupted|&gt;", "CUT"),
    ]:
        value = value.replace(token, f'<b class="token">{label}</b>')
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--tokenized", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clean, transformed, provenance = (
        read(args.clean),
        read(args.tokenized),
        read(args.provenance),
    )
    cards = []
    for index, (before, after, prov) in enumerate(
        zip(clean, transformed, provenance), 1
    ):
        rows = []
        for round_key in before:
            source, result = before[round_key], after[round_key]
            timeline = [f"<b>User:</b> {show(result['user'])}"]
            if "assistant_continuation" in result:
                c = result["assistant_continuation"]
                timeline += [
                    f"<b>DM:</b> {show(c['assistant'])}",
                    f"<b>User:</b> {show(c['user'])}",
                ]
            if "assistant_continuation3" in result:
                c = result["assistant_continuation3"]
                timeline += [
                    f"<b>DM:</b> {show(c['assistant'])}",
                    f"<b>User:</b> {show(c['user'])}",
                    f"<b>DM:</b> {show(c['assistant2'])}",
                    f"<b>User:</b> {show(c['user2'])}",
                ]
            timeline.append(f"<b>Assistant:</b> {show(result['assistant'])}")
            if "assistant_continuation2" in result:
                c = result["assistant_continuation2"]
                timeline += [
                    f"<b>User event:</b> {show(c['user'])}",
                    f"<b>Assistant:</b> {show(c['assistant'])}",
                ]
            rows.append(
                f"""<section class="round">
<h3>{html.escape(round_key)} · {html.escape(result['type'])}</h3>
<details><summary>Original complete exchange</summary>
<p><b>User:</b> {show(source['user'])}</p>
<p><b>Assistant:</b> {show(source['assistant'])}</p></details>
<div class="timeline">{''.join(f'<p>{line}</p>' for line in timeline)}</div>
</section>"""
            )
        cards.append(
            f"""<article><header><b>Pilot {index:02d}</b>
<span>{html.escape(prov['requested_model'])} · {html.escape(prov['topic'])}
· {html.escape(prov['style'])}</span></header>{''.join(rows)}</article>"""
        )
    args.output.write_text(
        f"""<!doctype html><html lang="vi"><meta charset="utf-8">
<title>Vietnamese authors-repo pipeline pilot</title><style>
body{{font:16px system-ui;max-width:1250px;margin:32px auto;color:#172033}}
.legend,article{{border:1px solid #ccd7e8;border-radius:14px;margin:18px 0;padding:18px}}
article>header{{display:flex;justify-content:space-between;background:#edf4ff;
margin:-18px -18px 18px;padding:16px;border-radius:14px 14px 0 0}}
.round{{border-top:1px solid #dce3ee;padding:8px 0}}.token{{background:#dbe8ff;
border-radius:7px;padding:2px 6px;color:#173f87}}p{{margin:8px 0}}
details{{color:#4f5e75}}.timeline{{padding:8px 14px;border-left:4px solid #82a9e8}}
</style><h1>Vietnamese DM pilot using the authors’ repository logic</h1>
<div class="legend"><b>S-S</b> start speaking · <b>S-L</b> stop speaking/listen ·
<b>C-S</b> continue speaking · <b>C-L</b> continue listening ·
<b>CUT</b> deterministic assistant truncation.
<p>The LLM generated only the complete exchanges. Expand “Original complete
exchange” to compare it with the deterministic transformation.</p></div>
{''.join(cards)}</html>""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
