# Vietnamese Full-Duplex Bench workflow

This repository contains the Vietnamese FDB v1.0 dataset and evaluation harness. Future work must preserve the streaming semantics and use resumable stages; do not rebuild completed outputs unnecessarily.

## Canonical paths

- Dataset: `external/VieNeu-TTS/outputs/fdb_vi_pilot_160_20260710`
- Harness: `vi_fdb_harness/`
- OpenAI run: `outputs/vi_fdb_v1_0/gpt_realtime/`
- ASR bakeoff: `outputs/vi_fdb_v1_0/asr_bakeoff_24/`
- Benchmark notes: `docs/vietnamese-fdb-benchmark-creation-notes.md`

The dataset has 160 samples: 20 samples in each of eight task categories. Treat the dataset release as Vietnamese FDB v1.0. The source category names `v1_0` and `v1_5` describe English-FDB task provenance, not separate Vietnamese dataset releases.

## Audio and timing invariants

- An event-condition input is not a mix of user input and model output. It is the complete user-side stream containing the initial request plus the timed event (backchannel, interruption, background speech, and so on).
- Feed input to the system as paced streaming audio, not as a whole-file upload.
- Assistant output remains a separate mono track.
- Input and assistant output share a monotonic clock whose `t=0` is immediately before the first input PCM frame is sent.
- Schedule assistant audio using observed arrival/playout time. Never concatenate it directly after input audio.
- Truncate buffered assistant audio when the response is cancelled.
- Preserve an output timeline at least as long as the input. Silence can be a valid action.
- Keep trailing silence when present. Do not synthesize or remove it merely to normalize duration; judge behavior using timestamps and task semantics.
- A response cancellation is not automatically a failure. Judge the observable behavior, e.g. a brief acknowledgement followed by correct continuation can be valid for a backchannel.

## Local environment and checks

Use `uv sync` to construct Python environments and `uv run` to execute Python tools. Do not install ad hoc packages into a shared interpreter.

```bash
cd vi_fdb_harness
uv sync
uv run python test_harness.py
```

Validate the dataset before inference:

```bash
uv run python harness.py validate-dataset \
  --dataset-root ../external/VieNeu-TTS/outputs/fdb_vi_pilot_160_20260710
```

Validate a completed run:

```bash
uv run python harness.py validate-run \
  --dataset-root ../external/VieNeu-TTS/outputs/fdb_vi_pilot_160_20260710 \
  --run-root ../outputs/vi_fdb_v1_0/gpt_realtime
```

Expected complete-run counts are 160 event outputs and 80 clean controls, with zero errors. Commands are resumable by default and skip completed outputs. Use overwrite only when deliberately invalidating a prior result.

## OpenAI Realtime inference

Use `v1_v1.5/model_inference/gpt-realtime/cli.js` through the harness. Never commit or print API keys. Load the key from a user-owned file into the environment.

```bash
uv run python harness.py run-openai \
  --dataset-root ../external/VieNeu-TTS/outputs/fdb_vi_pilot_160_20260710 \
  --run-root ../outputs/vi_fdb_v1_0/gpt_realtime \
  --node-cli ../v1_v1.5/model_inference/gpt-realtime/cli.js \
  --condition both --jobs 4
```

The harness enforces a subprocess timeout through `REALTIME_PROCESS_TIMEOUT_SECONDS` (default 180 seconds), so a dead WebSocket cannot block the run indefinitely. Inspect the per-case log and rerun; completed cases will be skipped.

## Persistent A100 cluster

SSH alias: `myserver`. It lands on `login-1`. Load Slurm before querying or attaching:

```bash
source /etc/profile.d/modules.sh
module load slurm/slurm/23.02.5
squeue -u "$USER"
```

Known persistent allocations at the time of writing are jobs `9417` and `9563` on `dgx-a100-5`. Always verify with `squeue`; do not assume an old job ID remains valid forever. Attach noninteractively with overlap so the hold job remains alive:

```bash
srun --jobid=9417 --overlap bash -lc 'nvidia-smi'
```

Remote project root is `/home/tuanda67/projects/Full-Duplex-Bench`. Remote `uv` is `/home/tuanda67/.local/bin/uv`. The ASR environment pins `torch==2.5.1` and `torchaudio==2.5.1` from the CUDA 12.1 PyTorch index because the cluster driver is compatible with CUDA 12.2; do not let generic resolution replace it with a CUDA 13 build.

## Data transfer policy

Prefer running model inference and ASR on the same cluster filesystem. Return only JSON, logs, summaries, and HTML to the Mac.

When audio must cross the SSH gateway:

1. Create one archive rather than transferring hundreds of small files.
2. Use resumable `rsync --partial --append`, not plain `scp`.
3. Compare local `shasum -a 256` with remote `sha256sum` before extraction.
4. Extract only after hashes match.

macOS tar extended-attribute warnings such as `LIBARCHIVE.xattr.com.apple.provenance` are harmless. Hash mismatch or a truncated archive is not harmless.

## ASR and judging policy

ChunkFormer is the primary Vietnamese ASR because the 24-case bakeoff showed close agreement with Whisper large-v3 on 20/24 cases and produced useful timestamps. Use Whisper large-v3 as a verifier, not as an unconditional second transcription of every sample.

Normalized ASR output must have this shape:

```json
{"text": "...", "chunks": [{"text": "...", "timestamp": [4.48, 5.52]}]}
```

Run the primary ASR on all event and clean outputs. Send cases to the verifier/human review queue when text similarity is below `0.75`, onset difference exceeds `0.3` seconds, either transcript is empty unexpectedly, or the timing conflicts with the Realtime event sidecar.

Do not expose `expected_action` or gold pass/fail to the behavioral judge. Do provide the neutral annotated event role (`non_addressed_background`, `non_addressed_side_conversation`, `addressed_request_change`, `addressed_backchannel`, or `within_utterance_pause`): the evaluated model must infer addressability from audio, while the judge is allowed to know the reference role it is scoring. The judge should receive observable input/output transcripts, raw timing events, paired clean-condition transcripts when available, and synchronized audio when needed. ASR is evidence, not ground truth; ambiguous cases require listening.

Use hybrid evaluation. Deterministic gates handle no speech, missed post-interruption responses, and conservative post-pause content loss. The LLM judge handles semantic continuation, adoption of changed constraints, and whether non-addressed speech influenced the response. Lexical overlap is diagnostic only for distractors because primary and background speech may share a topic. For paired v1.5-derived cases, compare event and clean responses counterfactually.

Human decisions are stored separately in `outputs/vi_fdb_v1_0/gpt_realtime/human_review_overrides.json`. Reports and final scores must prefer these adjudications over `judge.json`, while preserving both for audit. Never erase a human decision by rerunning the LLM judge.

## Current checkpoint (2026-07-18)

- Dataset validation: 160/160 valid.
- GPT-Realtime: 160/160 event outputs and 80/80 clean controls complete.
- Timing sidecars: complete; run validation has zero errors and warnings.
- Balanced ASR bakeoff: 24/24 ChunkFormer and 24/24 Whisper complete; 4/24 flagged.
- Full primary ASR on the cluster: 240/240 ChunkFormer transcripts complete under `gpt_realtime_240`.
- Final role-aware judge: 160/160 complete. The user inspected all 160 cases and supplied representative success cases and failure modes; 29 explicit example decisions are recorded. Do not treat unmentioned cases as implicitly human-approved. The effective automated-plus-explicit-example result is 101 pass and 59 review/fail; see `review_final_adjudicated.html`.

Next steps are to copy the small full-run ChunkFormer JSON bundle locally, run behavioral judging, generate the synchronized review HTML, and human-review the ASR/judge disagreement queue before publishing aggregate scores.
