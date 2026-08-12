# Vi-FDB v1.0 colleague handoff

Vietnamese version: [`vi-fdb-v1-colleague-handoff-vi.md`](vi-fdb-v1-colleague-handoff-vi.md)

## Links

- Dataset: <https://huggingface.co/datasets/tuanamz/vi-fdb-v1>
- Evaluation harness: <https://github.com/tuanad121/Full-Duplex-Bench/tree/main/vi_fdb_harness>
- GPT-Realtime reference outputs: <https://huggingface.co/datasets/tuanamz/vi-fdb-v1-gpt-realtime>
- Interactive result explorer: <https://huggingface.co/spaces/tuanamz/vi-fdb-v1-explorer>

The Hugging Face dataset is public and does not require an access request or
authentication. It follows the upstream Full-Duplex-Bench CC BY-NC 4.0 license.
Vi-FDB is independently published and is not an official VinFast dataset.

## What this benchmark measures

Vi-FDB evaluates Vietnamese full-duplex spoken-dialogue behavior: when a system
should wait, start speaking, stop for a real interruption, or continue through a
non-disruptive overlap. It contains 400 synthetic Vietnamese cases across seven
release tasks:

| Task | Cases | Intended behavior |
|---|---:|---|
| Backchannel | 50 | Produce a natural listening backchannel where appropriate |
| Pause handling | 50 | Do not take the turn during a mid-utterance pause |
| Smooth turn-taking | 50 | Respond after a completed user turn |
| Background speech | 50 | Ignore speech not addressed to the system |
| Talking to other | 50 | Ignore a side utterance directed to another person |
| User backchannel | 50 | Continue speaking through a non-disruptive backchannel |
| User interruption | 100 | Stop/yield and follow a changed request; includes standard and paired-control cases |

The package has two subsets:

- `data/pilot_160`: fully exercised end to end with GPT-Realtime, Vietnamese
  ASR, automated judging, and native-speaker corrections.
- `data/expansion_240`: passed structural/audio QC and has complete GPT-Realtime,
  PhoWhisper word-timestamp, Silero VAD, and applicable original-FDB metric artifacts.

Use all 400 event cases for a full submission. The 200 clean controls are paired
comparison conditions and are not additional headline benchmark cases.

## Download

Install the Hugging Face CLI, then download the public dataset:

```bash
hf download tuanamz/vi-fdb-v1 \
  --repo-type dataset \
  --local-dir vi-fdb-v1
```

Clone the harness separately:

```bash
git clone https://github.com/tuanad121/Full-Duplex-Bench.git
cd Full-Duplex-Bench/vi_fdb_harness
uv sync
```

Validate the fully audited pilot package:

```bash
uv run python harness.py validate-dataset \
  --dataset-root /absolute/path/to/vi-fdb-v1/data/pilot_160 \
  --profile original-160
```

Expected result: `ok: true`, 160 samples, and 20 samples in each of the eight
source task/version groups. The release taxonomy merges the two interruption
variants into one 100-case `user_interruption` task; the harness retains their
source methodology internally for compatible scoring.

Validate the expansion separately:

```bash
uv run python harness.py validate-dataset \
  --dataset-root /absolute/path/to/vi-fdb-v1/data/expansion_240 \
  --profile expansion-240
```

## Dataset layout

Each subset has a `manifest.json`. A manifest record points to the case audio and
metadata:

```json
{
  "task": "background_speech",
  "id": "000001",
  "input": "background_speech/000001/input.wav",
  "clean_input": "background_speech/000001/clean_input.wav",
  "metadata": "background_speech/000001/metadata.json",
  "dataset_version": "1.0"
}
```

Important files:

- `input.wav`: event condition sent to the evaluated system.
- `clean_input.wav`: matched control with the overlap replaced by silence;
  present where applicable.
- `metadata.json`: task, primary/event text, event interval, speaker/context
  information, and post-inference evaluation annotations.
- `source_streams/`: unmixed components retained for auditing.
- `index.html` or `vibe_check.html`: convenient dataset review pages.

The event role, source text, and annotated event timestamps are ground truth.
**Never expose them to the evaluated system.** Evaluation and human review may
use them only after inference. Some packaged JSON files retain a legacy
`expected_action` field from dialogue-manager data construction; it is not part
of Vi-FDB scoring and should be ignored.

## Running a model

For a smoke test, begin with one event/clean pair. The included adapter runs
OpenAI Realtime:

```bash
export OPENAI_API_KEY=...

uv run python harness.py run-openai \
  --dataset-root /absolute/path/to/vi-fdb-v1/data/pilot_160 \
  --run-root ../outputs/vi_fdb_v1_0/my_model \
  --node-cli ../v1_v1.5/model_inference/gpt-realtime/cli.js \
  --condition both --jobs 1 --limit 1
```

Install the JavaScript transport once before using that adapter:

```bash
cd ../v1_v1.5/model_inference/gpt-realtime
npm install
```

For another speech-to-speech system, implement an adapter with the same output
contract. The central requirement is one monotonic clock:

1. Finish model/session setup before starting the benchmark clock.
2. Set `t=0` immediately before sending the first input PCM frame.
3. Pace input using absolute source timestamps rather than sending the file as
   fast as possible.
4. Place assistant audio deltas on that same clock and account for audible
   playout, cancellation, and truncation.
5. Save `output.wav` and `output_timing.json`; for paired cases also save
   `clean_output.wav` and `clean_output_timing.json`.
6. Make the output timeline at least as long as the input timeline so silence
   and no-response behavior remain observable.

Validate a completed run:

```bash
uv run python harness.py validate-run \
  --dataset-root /absolute/path/to/vi-fdb-v1/data/pilot_160 \
  --run-root ../outputs/vi_fdb_v1_0/my_model
```

The harness is resumable. Do not pass `--overwrite` unless regenerating completed
cases intentionally.

## Transcription, judging, and review

Install the optional groups needed for Vietnamese ASR and automated judging:

```bash
uv sync --group asr
uv sync --group judge
```

Transcribe assistant output with PhoWhisper word timestamps, then attach
audio-observed outer speech boundaries with Silero VAD:

```bash
uv run python transcribe.py \
  --root ../outputs/vi_fdb_v1_0/my_model \
  --backend phowhisper

uv run python align_phowhisper_vad.py \
  --root ../outputs/vi_fdb_v1_0/my_model
```

Run the role-aware blinded judge. It receives the task definition, ASR output,
timing evidence, and post-inference reference annotations; it does not receive
dialogue-manager action tokens:

```bash
export OPENAI_API_KEY=...

uv run python judge.py \
  --root ../outputs/vi_fdb_v1_0/my_model \
  --asr-backend phowhisper_vad
```

Generate the synchronized audio/transcript review page:

```bash
uv run python report.py \
  --run-root ../outputs/vi_fdb_v1_0/my_model \
  --asr-backend phowhisper_vad
```

Review low-confidence judge decisions, ASR disagreements, no-speech cases, and a
stratified sample from every task. Report both automated and human-corrected
scores and retain the correction file.

## Our completed GPT-Realtime reference result

We ran **GPT-Realtime** on all 400 event cases and 200 clean controls. Semantic
judging uses the role-aware Vietnamese **GPT-4.1-mini** prompt. The pilot
includes 29 explicit native-speaker corrections; the expansion is automated
and should receive the same human audit before a leaderboard claim.

| Task | Pilot (20) | Expansion (30) | Full (50) |
|---|---:|---:|---:|
| Backchannel | 100% | 100% | 100% |
| Pause handling | 40% | 36.7% | 38% |
| Smooth turn-taking | 85% | 73.3% | 78% |
| User interruption — standard | 40% | 80% | 64% |
| Background speech | 30% | 10% | 18% |
| Talking to other | 30% | 26.7% | 28% |
| User backchannel | 100% | 96.7% | 98% |
| User interruption — paired variant | 80% | 96.7% | 90% |
| **Overall** | **63.1% (101/160)** | **65.0% (156/240)** | **64.2% (257/400)** |

Language-independent timing/turn metrics exported to the English FDB-compatible
layout were:

| Original FDB metric | GPT-Realtime result (50 cases) |
|---|---:|
| Pause take-over rate (lower is better) | 48% (24/50) |
| Pause correct-wait rate | 52% (26/50) |
| Smooth-turn take-over rate | 98% (49/50) |
| Smooth-turn latency, conditional on response | 1.000 s |
| Post-interruption response rate | 76% (38/50) |
| Post-interruption latency, conditional on response | 0.662 s |

The semantic score and timing metrics answer different questions. For
example, a system can correctly wait through a pause but then omit the resumed
part of the Vietnamese request; timing credits the wait, while semantic judging
correctly marks the full behavior as a failure.

The clearest pilot weakness was handling speech not addressed to the system:
background speech and talking-to-other both scored 30%. Backchannel behavior was
strong, while pause handling and changed-request interruptions exposed premature
responses, missed interruptions, and incomplete semantic uptake.

## What to report for a new submission

Record at minimum:

- exact Vi-FDB repository revision or commit SHA;
- evaluated subset (`pilot_160`, `expansion_240`, or both);
- model/service version, voice, VAD/turn-detection settings, and run date;
- concurrency and whether event/clean pairs used separate fresh sessions;
- completion counts and any failed/retried samples;
- ASR backend and version;
- automated judge model/prompt version;
- human-review sampling policy and every correction;
- per-task pass counts, overall pass rate, and applicable timing metrics.

When comparing submissions, report both the full score and the same named split.
The pilot is human-corrected while the expansion currently is not.

## Reference artifacts in the dataset repository

The Hugging Face repository includes:

- `evaluation/upstream_metrics.json`: English FDB-compatible timing metrics.
- `evaluation/semantic_parity.json`: pilot, expansion, and full semantic counts.
- `evaluation/interruption_relevance_summary.json`: localized Vietnamese
  interruption-relevance results.
- `data/pilot_160/vibe_check.html`: pilot dataset review page.
- `data/*/manifest.json`: canonical case inventories.

For design rationale, known failure modes, and release limitations, see
`docs/vi-fdb-v1-status-and-findings.md` in the harness repository.
