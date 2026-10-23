# Matched dialogue-generation pilot

This directory compares two ways to create Vietnamese full-duplex training
dialogues: one model writes both roles, or separate user and assistant models
role-play from the same private blueprint. See [PLAN.md](PLAN.md).

## Run the text pilot

```bash
cp fd_data_creation/config.example.json outputs/full_duplex_training/fd_pilot_config.json
export OPENROUTER_API_KEY="$(tr -d '\r\n' < 'docs/open router key.txt')"

python fd_data_creation/generate.py \
  --config outputs/full_duplex_training/fd_pilot_config.json \
  --output outputs/full_duplex_training/matched_generation_pilot \
  --pairs 4

python fd_data_creation/review.py \
  --config outputs/full_duplex_training/fd_pilot_config.json \
  --input outputs/full_duplex_training/matched_generation_pilot/matched_pairs.jsonl \
  --provenance outputs/full_duplex_training/matched_generation_pilot/provenance.jsonl \
  --output outputs/full_duplex_training/matched_generation_pilot/qc

python fd_data_creation/render_blind_vibe.py \
  --input outputs/full_duplex_training/matched_generation_pilot/matched_pairs.jsonl \
  --output outputs/full_duplex_training/matched_generation_pilot/vibe_check.html
```

The HTML randomizes A/B independently for every pair and keeps the answer key
inside the exported ratings rather than displaying it. Text review selects a
reviewer from another provider family when possible.

After choosing a generation mixture, pass `qc/accepted.jsonl` into the existing
VieNeu-TTS and timeline stages in `speech_training_data/pipeline/`.

