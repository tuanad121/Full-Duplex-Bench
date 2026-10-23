#!/usr/bin/env python3
"""Create a blinded A/B HTML review page for matched scripts."""

from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

from common import read_jsonl


def conversation_html(row: dict, label: str) -> str:
    turns = []
    for turn in row["rounds"]:
        turns.append(f'<div class="utter user"><b>User</b><span>{html.escape(turn["user"])}</span></div>')
        turns.append(f'<div class="utter agent"><b>Agent</b><span>{html.escape(turn["agent"])}</span></div>')
        event = turn.get("event")
        if event:
            turns.append(f'<div class="utter event"><b>User overlap · {html.escape(event["type"])}</b><span>{html.escape(event["text"])}</span></div>')
            if event["agent_after"]:
                turns.append(f'<div class="utter after"><b>Agent after</b><span>{html.escape(event["agent_after"])}</span></div>')
    return f'''<section class="version" data-label="{label}"><h3>Conversation {label}</h3>{''.join(turns)}
    <div class="scores">{''.join(f'<label>{name}<select data-score="{key}"><option value="">–</option>{"".join(f"<option>{n}</option>" for n in range(1,6))}</select></label>' for key,name in (("naturalness","Naturalness"),("roles","Distinct roles"),("coherence","Coherence"),("events","Event quality")))}</div></section>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    pairs = [pair for path in args.input for pair in read_jsonl(path)]
    rng = random.Random(args.seed)
    cards, answer_key = [], {}
    for pair in pairs:
        versions = [("single_author", pair["single_author"]), ("role_play", pair["role_play"])]
        rng.shuffle(versions)
        answer_key[pair["pair_key"]] = {"A": versions[0][0], "B": versions[1][0]}
        cards.append(f'''<article data-pair="{pair['pair_key']}"><h2>{html.escape(pair["blueprint"]["topic"])} <small>{pair['pair_key']}</small></h2>
        <div class="compare">{conversation_html(versions[0][1], "A")}{conversation_html(versions[1][1], "B")}</div>
        <div class="preference"><b>Overall preference</b><button data-pref="A">A</button><button data-pref="tie">Tie</button><button data-pref="B">B</button><input placeholder="Optional note"></div></article>''')
    key_json = json.dumps(answer_key, ensure_ascii=False)
    page = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blind dialogue vibe check</title><style>
    body{{max-width:1500px;margin:30px auto;padding:0 20px;background:#f3f6fa;color:#172033;font:15px/1.45 system-ui}}article{{background:white;border:1px solid #d9e1ed;border-radius:16px;padding:18px;margin:18px 0}}h2{{margin-top:0}}small{{color:#78869b}}.compare{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.version{{border:1px solid #dce4ef;border-radius:12px;padding:13px}}.utter{{display:grid;grid-template-columns:135px 1fr;gap:8px;padding:7px;border-top:1px solid #eef2f7}}.user b{{color:#1659a5}}.agent b,.after b{{color:#08775b}}.event{{background:#fff7df}}.scores,.preference{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:14px}}label{{display:flex;gap:5px;align-items:center}}select,button,input{{font:inherit;border:1px solid #b8c4d4;background:white;border-radius:8px;padding:7px 10px}}button.chosen{{background:#172033;color:white}}input{{flex:1}}@media(max-width:900px){{.compare{{grid-template-columns:1fr}}}}
    </style></head><body><h1>Blind single-author vs role-play vibe check</h1><p>Rate each independently before choosing A, B, or Tie. Generation strategy is hidden until export.</p>{''.join(cards)}<button id="export">Export ratings</button><script>
    const storage='fd_matched_vibe_v1', state=JSON.parse(localStorage.getItem(storage)||'{{}}'), answerKey={key_json};
    function save(){{localStorage.setItem(storage,JSON.stringify(state))}};
    document.querySelectorAll('article').forEach(card=>{{const id=card.dataset.pair;state[id]??={{versions:{{A:{{}},B:{{}}}},preference:'',note:''}};card.querySelectorAll('select').forEach(s=>{{const label=s.closest('.version').dataset.label;s.value=state[id].versions[label][s.dataset.score]||'';s.onchange=()=>{{state[id].versions[label][s.dataset.score]=Number(s.value);save()}}}});card.querySelectorAll('[data-pref]').forEach(b=>{{if(state[id].preference===b.dataset.pref)b.classList.add('chosen');b.onclick=()=>{{state[id].preference=b.dataset.pref;card.querySelectorAll('[data-pref]').forEach(x=>x.classList.toggle('chosen',x===b));save()}}}});const input=card.querySelector('input');input.value=state[id].note;input.oninput=()=>{{state[id].note=input.value;save()}}}});
    export.onclick=()=>{{const payload={{ratings:state,answer_key:answerKey}};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}}));a.download='matched_dialogue_vibe_ratings.json';a.click()}};
    </script></body></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
