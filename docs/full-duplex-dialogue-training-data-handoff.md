# Vietnamese full-duplex dialogue training-data handoff

Updated: 2026-09-03  
Repository: `/Users/tuandinh/repo/Full-Duplex-Bench`

## Executive summary

The repository now has an end-to-end prototype for creating Vietnamese
full-duplex speech training examples with VieNeu-TTS. It can:

1. generate long Vietnamese user/assistant scripts containing backchannels and
   real or ambiguous interruptions;
2. independently review the scripts;
3. synthesize atomic user, assistant, overlap-event, and post-interruption clips;
4. render synchronized user, assistant, and mono-mixture tracks;
5. apply structural and timing QC; and
6. extract fixed 24-second, event-centred training windows.

The audio half of the pipeline is implemented and has passed a synthetic
one-conversation fixture. The initial four-example VieNeu-TTS pilot also exists
and is useful for listening. The newer matched text-generation experiment has
produced several valid single-author/role-play pairs, but its only completed LLM
review run accepted zero conversations. Therefore there is not yet a production
speech dataset or a settled model-facing training format.

The next owner should first fix and rerun text review on a 100-pair matched
pilot, choose the authoring mixture using blind human review plus automated
metrics, and only then spend A100 time on VieNeu-TTS synthesis.

## Scope and non-goals

This track creates synchronized speech examples for training a full-duplex
speech-to-speech model. It is separate from:

- the 400-case Vi-FDB v1.0 evaluation benchmark under `vi_fdb_harness/`; and
- the four-control-token semantic dialogue-manager dataset under
  `dialogue_manager/`.

Do not copy Vi-FDB benchmark cases into this training set. The current training
pipeline covers clean core behaviours only. Background speakers, speech aimed
at other people, acoustic noise, room impulse responses, codecs, microphone
distance, and echo are deferred.

## Behaviour represented in the data

Each conversation has 6–10 ordinary user/assistant rounds and normally two
behavioural events. Every event occurs while the assistant is speaking.

| Event | User behaviour | Target assistant behaviour |
|---|---|---|
| `backchannel` | A short acknowledgement such as “ừm” | Continue the current utterance |
| `real_interruption` | A substantive correction or changed request | Stop after a short reaction latency, listen, then answer the new request |
| `ambiguous_interruption` | A backchannel-like onset that becomes substantive | Treat it as a real interruption once its intent becomes clear |

The event is stored inside its host round. The ordering is `user`, assistant
starts `agent`, the user speaks `event.text` at `overlap_fraction` of the
assistant clip, and—except for a backchannel—the assistant resumes with
`event.agent_after`.

## Repository map

| Path | Purpose | Current state |
|---|---|---|
| `fd_data_creation/` | New matched text-authoring experiment and production-oriented batch generators | Implemented; pilot-scale only |
| `speech_training_data/pipeline/` | VieNeu-TTS synthesis, rendering, QC, window extraction, and an older text generator/reviewer | Implemented; audio fixture passes |
| `speech_training_data/phase1/` | Earlier four-example proof of concept | Rendered and listenable |
| `outputs/full_duplex_training/phase1_pilot/` | Four rendered behaviours and a backchannel voice check | Complete prototype, about 26 MB |
| `outputs/full_duplex_training/pipeline_fixture/` | One six-round assembled conversation with two extracted windows | Complete fixture, about 38 MB |
| `outputs/full_duplex_training/matched_generation_*` | Small text-generation/review experiments | Exploratory, not a canonical dataset |
| `external/VieNeu-TTS/` | Local VieNeu-TTS checkout and voice references | Git HEAD `4c57439`; working tree contains unrelated untracked work |

All of these main trees are currently untracked from the parent repository.
Preserve them and never use `git add .` or clean the working tree wholesale.
The files `docs/open router key.txt` and
`external/VieNeu-TTS/open router key.txt` are credential-related and must not be
committed, copied into artifacts, or printed.

## Text authoring: current design

The matched pilot deliberately compares two ways to create the same private
scenario blueprint:

- `single_author`: one frontier model writes both roles and event metadata;
- `role_play`: a director writes the blueprint, separate user and assistant
  models alternate turns from public history, and the same roles generate the
  overlap and post-interruption speech.

`fd_data_creation/generate.py` emits both strategies for each matched job.
`generate_batch.py` is the resumable production variant: it writes one result
per job under `items/`, skips any existing item file, and then rebuilds the
aggregate JSONL files. Its configured 75/25 mixture is assigned with exact
counts rather than probabilistic sampling.

Generation retains requested/provider model IDs, prompt version, seed, token
usage, and reported OpenRouter cost. The reviewer is selected from another
provider family when possible. Deterministic schema validation runs before paid
LLM review.

The canonical conversation passed to synthesis has this shape:

```json
{
  "conversation_id": "vi_fd_001001_sa",
  "topic": "sắp lịch họp",
  "style": "thân mật tự nhiên",
  "rounds": [
    {"user": "...", "agent": "...", "event": null},
    {
      "user": "...",
      "agent": "...",
      "event": {
        "type": "real_interruption",
        "text": "Khoan, đổi sang chiều mai nhé.",
        "agent_after": "Được, mình chuyển sang chiều mai.",
        "overlap_fraction": 0.5
      }
    }
  ]
}
```

`fd_data_creation/review_batch.py` produces `accepted.jsonl` in exactly this
conversation schema, so it can be given directly to
`speech_training_data/pipeline/synthesize_utterances.py`. Do not pass
`matched_pairs.jsonl` to synthesis; that file wraps two conversations plus their
shared blueprint.

## VieNeu-TTS and audio pipeline

The audio stages use two fixed reference voices from the VieNeu-TTS checkout:

- user: `src/vieneu/assets/samples/Ly (nữ miền Bắc).wav`;
- assistant: `src/vieneu/assets/samples/Vĩnh (nam miền Nam).wav`.

`synthesize_utterances.py` constructs `Vieneu(mode="v3turbo",
backend="pytorch")` and synthesizes each clip independently at temperature
0.8. Existing WAV files are skipped, so synthesis is resumable at clip level.
The output manifest is rewritten after the full input loop; interrupted runs may
have reusable WAVs even if the manifest is incomplete.

`render_timelines.py` creates equal-length 48-kHz mono tracks:

- `user.wav`: ordinary user turns plus overlap events;
- `agent.wav`: assistant speech, interruption cuts, and post-event responses;
- `mixture.wav`: the clipped sum, mainly for listening and optional training.

The current timing defaults are:

| Setting | Value |
|---|---:|
| Turn gap | 0.42 s |
| Interruption reaction latency | 0.32 s |
| Fade at interruption cut | 0.05 s |
| Post-event reply gap | 0.48 s after the event ends |
| Trailing silence | 0.50 s |
| Training window | 24.0 s |
| Preferred event position in window | 14.0 s |

Backchannels do not cut the assistant track. Real and ambiguous interruptions
cut it 320 ms after event onset and apply a 50-ms fade. A post-event assistant
clip begins 480 ms after the user event ends. Each rendered conversation stores
the exact timeline and expected action in `metadata.json`.

Audio QC currently checks readable mono audio, configured sample rate,
synchronized lengths, finite values, non-empty stems, clipping, event energy,
backchannel overlap, interruption stop latency, and assistant silence after the
cut. It does **not** yet run ASR, VAD, speaker-similarity, intelligibility, or
perceptual-quality checks.

Window extraction is causal and event-centred. It shifts a window at recording
boundaries as needed and skips it if another event onset would appear in the
same 24-second window. Full conversations remain the source of truth under
`04_rendered/`; windows under `06_windows/` are derived artifacts.

## Current evidence and artifacts

### Audio proof of concept

`outputs/full_duplex_training/phase1_pilot/index.html` presents four rendered
examples:

- normal turn;
- backchannel;
- real interruption; and
- ambiguous interruption.

Each example has synchronized user, assistant, and mixture WAVs plus metadata.
This was the first proof that separately synthesized VieNeu-TTS clips could be
placed on a common timeline.

### Pipeline fixture

`outputs/full_duplex_training/pipeline_fixture/` contains one six-round
conversation rendered from already existing pilot clips. Structural audio QC
accepted 1/1 conversations. Window extraction produced two 24-second windows
and skipped zero. This validates pipeline plumbing and timing logic, not speech
quality or dataset diversity.

### Matched text-generation experiments

The small attempts produced these results:

| Run | Requested pairs | Valid pairs | Conversations |
|---|---:|---:|---:|
| `matched_generation_pilot` | 1 | 0 | 0 |
| `matched_generation_pilot_v1` | 3 | 1 | 2 |
| `matched_generation_retry_3` | 1 | 1 | 2 |
| `matched_generation_retry_5` | 1 | 1 | 2 |
| `matched_generation_smoke_v2` | 1 | 1 | 2 |

These are overlapping/retry experiments, not five independent dataset shards.
Do not concatenate them blindly.

The only stored matched-review summary is under
`matched_generation_smoke_v2/qc/`: 0/1 single-author and 0/1 role-play
conversations accepted. One rejection was caused by malformed reviewer JSON;
the other review appears to misunderstand who speaks the overlap despite the
review prompt explaining the schema. This makes the present zero-percent result
an unreliable authoring-quality comparison.

## Reproduction commands

### 1. Create a run configuration

```bash
cp fd_data_creation/config.example.json \
  outputs/full_duplex_training/fd_pilot_config.json
```

Review model names before spending money: availability and pricing can change.
Keep the API key in the environment and out of command output.

```bash
export OPENROUTER_API_KEY="$(tr -d '\r\n' < 'docs/open router key.txt')"
```

### 2. Run the matched pilot

The experiment in `PLAN.md` calls for at least 100 matched pairs. Start with a
small paid smoke test whenever prompts or providers change.

```bash
python fd_data_creation/generate.py \
  --config outputs/full_duplex_training/fd_pilot_config.json \
  --output outputs/full_duplex_training/matched_pilot_100 \
  --pairs 100 --start 1 --jobs 2

python fd_data_creation/review.py \
  --config outputs/full_duplex_training/fd_pilot_config.json \
  --input outputs/full_duplex_training/matched_pilot_100/matched_pairs.jsonl \
  --provenance outputs/full_duplex_training/matched_pilot_100/provenance.jsonl \
  --output outputs/full_duplex_training/matched_pilot_100/qc

python fd_data_creation/render_blind_vibe.py \
  --input outputs/full_duplex_training/matched_pilot_100/matched_pairs.jsonl \
  --output outputs/full_duplex_training/matched_pilot_100/vibe_check.html
```

The matched generator is not item-resumable. For a larger run after selecting a
strategy mixture, use `generate_batch.py` and `review_batch.py`:

```bash
python fd_data_creation/generate_batch.py \
  --config outputs/full_duplex_training/fd_pilot_config.json \
  --output outputs/full_duplex_training/production_text_v1 \
  --count 1000 --start 1001 --jobs 4

python fd_data_creation/review_batch.py \
  --config outputs/full_duplex_training/fd_pilot_config.json \
  --input outputs/full_duplex_training/production_text_v1/generated.jsonl \
  --provenance outputs/full_duplex_training/production_text_v1/provenance.jsonl \
  --output outputs/full_duplex_training/production_text_v1/qc \
  --jobs 4
```

### 3. Synthesize accepted conversations on an A100

Use the existing VieNeu-TTS GPU environment. On the known cluster, the project
root is `/home/tuanda67/projects/Full-Duplex-Bench`; verify the active Slurm
allocation rather than relying on an old job ID.

```bash
python speech_training_data/pipeline/synthesize_utterances.py \
  --config outputs/full_duplex_training/fd_pilot_config.json \
  --input outputs/full_duplex_training/production_text_v1/qc/accepted.jsonl \
  --output outputs/full_duplex_training/production_audio_v1/03_utterances \
  --vieneu-root external/VieNeu-TTS
```

Run this command from the repository root. The stage scripts use local imports,
and their manifests store paths as supplied on the command line. For portability
between the cluster and Mac, keep both workspaces at the same relative layout or
rewrite/validate manifest paths after transfer.

### 4. Render, QC, and extract windows

These stages need `numpy` and `soundfile`, not a GPU:

```bash
python speech_training_data/pipeline/render_timelines.py \
  --config outputs/full_duplex_training/fd_pilot_config.json \
  --input outputs/full_duplex_training/production_audio_v1/03_utterances/utterance_manifest.jsonl \
  --output outputs/full_duplex_training/production_audio_v1/04_rendered

python speech_training_data/pipeline/audio_qc.py \
  --config outputs/full_duplex_training/fd_pilot_config.json \
  --input outputs/full_duplex_training/production_audio_v1/04_rendered/rendered_manifest.jsonl \
  --output outputs/full_duplex_training/production_audio_v1/05_audio_qc

python speech_training_data/pipeline/extract_windows.py \
  --config outputs/full_duplex_training/fd_pilot_config.json \
  --input outputs/full_duplex_training/production_audio_v1/05_audio_qc/accepted.jsonl \
  --output outputs/full_duplex_training/production_audio_v1/06_windows

python speech_training_data/pipeline/render_vibe.py \
  --input outputs/full_duplex_training/production_audio_v1/05_audio_qc/accepted.jsonl \
  --output outputs/full_duplex_training/production_audio_v1/vibe_check.html
```

`run_pipeline.py` can orchestrate the older `speech_training_data/pipeline`
text generator through all audio stages, but it does not orchestrate the newer
`fd_data_creation` matched workflow. For the preferred matched workflow, use the
explicit commands above until the orchestrator is updated.

## Validation before scaling

Run the offline tests from repository root in environments containing their
dependencies:

```bash
python -m unittest discover -s speech_training_data/pipeline -p 'test_*.py'
python -m unittest discover -s fd_data_creation -p 'test_*.py'
```

At handoff time, the three `speech_training_data/pipeline` schema tests pass.
The four `fd_data_creation` tests were not executable in the default local
Python because the `openai` package is absent; the failure occurs during test
import, not in an assertion. Use an isolated `uv` environment rather than
installing into the shared interpreter.

Before accepting a production run, verify at minimum:

- requested/generated/reviewed/accepted counts agree;
- conversation IDs are unique and do not overlap held-out evaluation data;
- event-type, topic, style, round-count, author-model, and speaker distributions
  match the intended design;
- every manifest path resolves on the machine running the next stage;
- all audio-QC rejections are explained and sampled by a listener;
- a blind human sample supports the selected single-author/role-play mixture;
- a listening sample covers every event type, voice, and event-position bucket;
- training windows do not contain a second behavioural event; and
- checksums and run configuration are retained with any transferred/released
  shard.

## Known limitations and risks

1. **Text review is not reliable enough yet.** Reviewer JSON can be truncated,
   and at least one stored review misread event roles. Add stronger structured
   output/repair and explicit reviewer calibration examples before scaling.
2. **Matched generation is fragile.** Observed failures include empty/non-JSON
   provider responses, invalid blueprint event positions, and extra/invalid
   round fields. The batch path is resumable, but a failed item file is treated
   as complete until it is deliberately removed or retried under a new output
   root.
3. **No production synthesis has run.** The only assembled full conversation is
   a fixture that reuses pilot clips. It does not validate long-run VieNeu-TTS
   stability, throughput, speaker consistency, or disk requirements.
4. **Audio QC is structural.** It cannot detect mispronunciation, hallucinated
   speech, voice drift, unnatural prosody, or semantic mismatch between text
   and audio.
5. **Fixed speaker identities limit diversity.** Ly and Vĩnh are sufficient for
   plumbing, not for a broadly representative Vietnamese speech corpus. Voice
   expansion requires licensing/provenance checks and speaker-disjoint eval.
6. **Timing is synthetic.** Overlap fractions, 320-ms reaction latency, and
   fixed gaps are rule-based. Their distribution should be widened using
   evidence from natural conversation without leaking benchmark cases.
7. **Model-facing serialization is undecided.** The pipeline produces separate
   user/assistant targets and a mixture plus metadata, but no final shard/index
   contract for a specific trainer. Decide whether the model consumes user-only
   audio, the mixture, prior assistant audio, text/control annotations, or some
   combination before a large render.
8. **Path portability is weak.** JSON manifests can contain workspace-specific
   relative paths. Moving only part of a run between Mac and cluster may break
   downstream stages.
9. **No formal data license manifest exists.** Preserve generator model terms,
   VieNeu-TTS version, reference-voice provenance, prompt/config snapshots, and
   any future noise/RIR licenses before release or training use.

## Recommended next steps

1. Repair reviewer robustness and add regression fixtures for the exact
   user/agent/event ordering that the stored reviewer misunderstood.
2. Run a cheap 4–10-pair smoke test, inspect every output, then run the planned
   100 matched pairs.
3. Blind-review at least 20 matched pairs and compare acceptance, naturalness,
   event correctness, duplication, cost, and latency by strategy.
4. Freeze the chosen generation mixture, prompts, models, seed, schemas, and
   configuration as a versioned text-corpus recipe.
5. Define the downstream trainer's exact sample contract and split policy.
6. Synthesize a 10–20-conversation VieNeu-TTS batch on A100; measure real-time
   factor, failures, peak memory, disk per hour, and resumption after interruption.
7. Add ASR text agreement, VAD/timing, speaker similarity, and sampled listening
   to audio QC before large-scale generation.
8. Expand speaker and timing diversity only after the two-speaker clean baseline
   is reliable; add non-addressed speech and acoustic corruption in later,
   separately auditable phases.

## Suggested reading order

1. `docs/full-duplex-dialogue-training-data-handoff.md` (this document)
2. `fd_data_creation/PLAN.md`
3. `fd_data_creation/README.md`
4. `speech_training_data/README.md`
5. `fd_data_creation/common.py`
6. `speech_training_data/pipeline/schema.py`
7. `speech_training_data/pipeline/synthesize_utterances.py`
8. `speech_training_data/pipeline/render_timelines.py`
9. `speech_training_data/pipeline/audio_qc.py`
10. `speech_training_data/pipeline/extract_windows.py`

