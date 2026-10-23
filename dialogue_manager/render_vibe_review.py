#!/usr/bin/env python3
"""Render a self-contained HTML vibe review for Vietnamese DM training data."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from export_paper_format import (
    decision_input,
    quality_issues,
    serialize_input,
)

PATTERN_LABELS = {
    "normal": "Normal exchange",
    "incomplete": "Incomplete query",
    "real_interruption": "Real interruption",
    "fake_backchannel": "Backchannel while assistant speaks",
    "fake_talking_to_other": "User talks to another person",
    "fake_background_speech": "Background speech",
}


def review_safe(row: dict) -> bool:
    return not quality_issues(row)


def select(rows: list[dict], per_pattern: int) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if review_safe(row):
            buckets[row["job"]["pattern"]].append(row)
    chosen = []
    for pattern in PATTERN_LABELS:
        candidates = buckets[pattern]
        if len(candidates) < per_pattern:
            raise ValueError(f"Only {len(candidates)} review-safe rows for {pattern}")
        # Evenly cover the full generation run instead of taking adjacent IDs.
        indexes = [round(i * (len(candidates) - 1) / max(per_pattern - 1, 1)) for i in range(per_pattern)]
        chosen.extend(candidates[index] for index in indexes)
    return chosen


def select_by_model(rows: list[dict], per_model: int) -> list[dict]:
    """Sample each paid model separately while covering different patterns."""
    buckets: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if review_safe(row):
            model = row.get("provider_model") or row["job"].get("model", "unknown")
            buckets[model][row["job"]["pattern"]].append(row)
    chosen: list[dict] = []
    for model in sorted(buckets):
        model_rows: list[dict] = []
        patterns = [pattern for pattern in PATTERN_LABELS if buckets[model][pattern]]
        cursor = Counter()
        while len(model_rows) < per_model and patterns:
            next_patterns = []
            for pattern in patterns:
                candidates = buckets[model][pattern]
                index = cursor[pattern]
                if index < len(candidates) and len(model_rows) < per_model:
                    model_rows.append(candidates[index])
                    cursor[pattern] += 1
                if cursor[pattern] < len(candidates):
                    next_patterns.append(pattern)
            patterns = next_patterns
        chosen.extend(model_rows)
    return chosen


def render_card(row: dict) -> str:
    job, response = row["job"], row["response"]
    turns = []
    for index, turn in enumerate(response["turns"]):
        action = html.escape(turn["action"])
        actual_input = serialize_input(decision_input(response["turns"], index))
        turns.append(f"""
        <div class="turn action-{action[2]}" data-role="{html.escape(turn['event_role'])}">
          <div class="turn-head"><span>Round {turn.get('round', index + 1)}</span><span class="tag">{action}</span></div>
          <div class="field model-input"><b>Actual model input</b><pre>{html.escape(actual_input)}</pre></div>
          <div class="field"><b>User text</b><span>{html.escape(turn['user_text'])}</span></div>
          <div class="field"><b>Streaming ASR</b><span>{html.escape(turn['streaming_asr'])}</span></div>
          <div class="field"><b>Assistant state</b><span>{html.escape(turn['speaker_state'])}</span></div>
          <details><summary>Audit-only fields (not given to model)</summary>
            <div class="field"><b>Event role</b><span>{html.escape(turn['event_role'])}</span></div>
            <div class="field"><b>Full user text</b><span>{html.escape(turn['user_text'])}</span></div>
            <div class="field"><b>Assistant reply</b><span>{html.escape(turn.get('assistant_reply') or '∅')}</span></div>
          </details>
        </div>""")
    cid = response["conversation_id"]
    model = row.get("provider_model") or job.get("model", "unknown")
    return f"""
    <article class="card" data-pattern="{job['pattern']}" data-model="{html.escape(model)}" data-id="{cid}">
      <header><div><h2>{cid}</h2><p>{html.escape(PATTERN_LABELS[job['pattern']])}</p></div>
      <span class="pill">{len(response['turns'])} decisions</span></header>
      <div class="meta"><span><b>Topic:</b> {html.escape(job['topic'])}</span><span><b>Style:</b> {html.escape(job['style'])}</span><span><b>Context:</b> {html.escape(job['context_family'])}</span><span><b>Model:</b> {html.escape(row.get('provider_model',''))}</span></div>
      <div class="turns">{''.join(turns)}</div>
      <div class="review"><button data-v="pass">✓ Good</button><button data-v="fail">✗ Problem</button><button data-v="clear">Clear</button><input placeholder="Optional note: wrong label, unnatural text…"></div>
    </article>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-pattern", type=int, default=6)
    parser.add_argument("--per-model", type=int, default=0)
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((args.input / "accepted").glob("*.json"))]
    chosen = (
        select_by_model(rows, args.per_model)
        if args.per_model
        else select(rows, args.per_pattern)
    )
    excluded = sum(not review_safe(row) for row in rows)
    if args.per_model:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in chosen:
            grouped[row.get("provider_model") or row["job"].get("model", "unknown")].append(row)
        cards = "".join(
            f'<section class="model-group" data-model-group="{html.escape(model)}">'
            f'<h2 class="model-title">{html.escape(model)}</h2>'
            f'<p class="model-count">{len(model_rows)} sampled conversations</p>'
            f'{"".join(render_card(row) for row in model_rows)}</section>'
            for model, model_rows in sorted(grouped.items())
        )
        sample_description = (
            f"{len(chosen)} conversations, up to {args.per_model} per provider model"
        )
    else:
        sample_description = (
            f"{len(chosen)} conversations, {args.per_pattern} per generation pattern"
        )
    model_values = sorted(
        {
            row.get("provider_model") or row["job"].get("model", "unknown")
            for row in chosen
        }
    )
    payload = html.escape(json.dumps({"sampled": len(chosen), "quality_gate_excluded": excluded, "source": str(args.input)}, ensure_ascii=False))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Vi DM training data — vibe review</title>
<style>
:root{{--bg:#f3f6fb;--card:#fff;--ink:#172033;--muted:#64748b;--line:#dbe3ef;--blue:#2563eb;--green:#15803d;--red:#b91c1c}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}h1{{margin:0}}.lead{{color:var(--muted);max-width:900px}}.legend{{display:grid;grid-template-columns:repeat(4,minmax(210px,1fr));gap:10px;margin:20px 0}}.legend-item{{background:white;border:1px solid var(--line);border-radius:12px;padding:14px}}.legend-item code{{display:inline-block;background:#e8eefc;border-radius:7px;padding:3px 7px;font-size:16px;font-weight:700}}.legend-item b{{display:block;margin:8px 0 3px}}.legend-item p{{color:var(--muted);font-size:14px;margin:0}}.legend-note{{color:#475569;margin:6px 0 0}}.toolbar{{position:sticky;top:0;z-index:5;background:#f3f6fbef;padding:14px 0;display:flex;gap:10px;align-items:center;flex-wrap:wrap}}select,button,input{{font:inherit;border:1px solid #b9c6d8;border-radius:8px;padding:8px 12px;background:white}}button{{cursor:pointer}}#count{{color:var(--muted);margin-left:auto}}.model-group{{margin:30px 0 54px}}.model-title{{font-size:27px;border-bottom:3px solid #cbd5e1;padding-bottom:8px}}.model-count{{color:var(--muted)}}.card{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px;margin:18px 0;box-shadow:0 3px 12px #1e293b0b}}.card>header{{display:flex;justify-content:space-between;gap:12px}}h2{{margin:0;font-size:21px}}header p{{margin:2px 0;color:var(--muted)}}.pill,.tag{{background:#e8eefc;border-radius:999px;padding:5px 10px;white-space:nowrap}}.meta{{display:flex;gap:18px;flex-wrap:wrap;background:#f8fafc;padding:10px 12px;margin:14px 0;border-radius:10px;color:#475569}}.turns{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:12px}}.turn{{border:1px solid var(--line);border-radius:12px;overflow:hidden}}.turn-head{{display:flex;justify-content:space-between;align-items:center;background:#eef3fa;padding:9px 12px;font-weight:700}}.field{{display:grid;grid-template-columns:140px 1fr;gap:10px;padding:7px 12px;border-top:1px solid #edf1f6}}.field b{{color:#475569}}.model-input{{display:block;background:#f8fafc}}.model-input pre{{white-space:pre-wrap;margin:7px 0 0;font:14px/1.5 ui-monospace,SFMono-Regular,monospace}}details{{border-top:1px solid #edf1f6}}summary{{cursor:pointer;padding:8px 12px;color:#64748b}}.review{{display:flex;gap:8px;margin-top:15px}}.review input{{flex:1}}.card[data-review=pass]{{border:2px solid #22c55e}}.card[data-review=fail]{{border:2px solid #ef4444}}.hidden{{display:none}}@media(max-width:900px){{.legend{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:650px){{main{{padding:15px}}.legend{{grid-template-columns:1fr}}.turns{{grid-template-columns:1fr}}.field{{grid-template-columns:1fr}}#count{{margin-left:0}}}}
</style></head><body><main><h1>Vietnamese dialogue-manager training data</h1><p class="lead">Vibe-check sample: {sample_description}. A conservative consistency gate excluded {excluded} of {len(rows)} currently accepted conversations from sampling. Examples are grouped by the actual provider model. Focus on naturalness, continuity, whether the event role matches who is speaking, and whether the action token is correct.</p>
<section aria-label="Output action labels"><div class="legend">
  <div class="legend-item"><code>&lt;|C-L|&gt;</code><b>Continue listening</b><p>The assistant is listening and the user's query is incomplete. Do not answer yet.</p></div>
  <div class="legend-item"><code>&lt;|S-S|&gt;</code><b>Start speaking</b><p>The assistant is listening and the user's query is complete. Take the turn and reply.</p></div>
  <div class="legend-item"><code>&lt;|S-L|&gt;</code><b>Stop speaking; listen</b><p>The assistant is speaking, but the user gives a real correction, objection, or new request. Yield the turn.</p></div>
  <div class="legend-item"><code>&lt;|C-S|&gt;</code><b>Continue speaking</b><p>The assistant is speaking and hears a backchannel, background speech, or speech addressed to someone else. Do not yield.</p></div>
</div><p class="legend-note"><b>Mnemonic:</b> the action describes what the assistant should do next: continue/start speaking, or continue/start listening.</p></section>
<div class="toolbar"><select id="model"><option value="">All models</option>{''.join(f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in model_values)}</select><select id="pattern"><option value="">All patterns</option>{''.join(f'<option value="{p}">{html.escape(label)}</option>' for p,label in PATTERN_LABELS.items())}</select><select id="status"><option value="">All review states</option><option value="pass">Good</option><option value="fail">Problem</option><option value="unreviewed">Unreviewed</option></select><button id="export">Export review JSON</button><span id="count"></span></div><div id="cards">{cards}</div><script id="info" type="application/json">{payload}</script>
<script>
const key='vi_dm_vibe_review_v1', saved=JSON.parse(localStorage.getItem(key)||'{{}}');
const cards=[...document.querySelectorAll('.card')];
for(const c of cards){{const r=saved[c.dataset.id];if(r){{c.dataset.review=r.verdict||'';c.querySelector('input').value=r.note||''}}}}
function persist(c,v){{if(v==='clear'){{delete saved[c.dataset.id];delete c.dataset.review;c.querySelector('input').value=''}}else{{c.dataset.review=v;saved[c.dataset.id]={{verdict:v,note:c.querySelector('input').value}}}}localStorage.setItem(key,JSON.stringify(saved));filter()}}
document.addEventListener('click',e=>{{if(e.target.dataset.v)persist(e.target.closest('.card'),e.target.dataset.v)}});
document.addEventListener('change',e=>{{if(e.target.matches('.review input')){{const c=e.target.closest('.card');if(c.dataset.review)persist(c,c.dataset.review)}}else filter()}});
document.addEventListener('input',e=>{{if(e.target.matches('.review input')){{const c=e.target.closest('.card');if(c.dataset.review){{saved[c.dataset.id].note=e.target.value;localStorage.setItem(key,JSON.stringify(saved))}}}}}});
function filter(){{const m=model.value,p=pattern.value,s=status.value;let n=0;for(const c of cards){{const r=c.dataset.review||'unreviewed',show=(!m||c.dataset.model===m)&&(!p||c.dataset.pattern===p)&&(!s||r===s);c.classList.toggle('hidden',!show);if(show)n++}}for(const g of document.querySelectorAll('.model-group')){{g.classList.toggle('hidden',![...g.querySelectorAll('.card')].some(c=>!c.classList.contains('hidden')))}}count.textContent=`${{n}} shown · ${{Object.keys(saved).length}} reviewed`}}filter();
export.onclick=()=>{{const out={{schema_version:1,created_at:new Date().toISOString(),reviews:saved}};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}}));a.download='vi_dm_vibe_review.json';a.click();URL.revokeObjectURL(a.href)}};
</script></main></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(f"Wrote {args.output} with {len(chosen)} conversations; excluded {excluded} inconsistent candidates")


if __name__ == "__main__":
    main()
