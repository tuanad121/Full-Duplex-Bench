# Vietnamese Full-Duplex Benchmark v1.0: Status and Findings

## Release status

Vi-FDB currently contains 400 synthetic Vietnamese evaluation cases: 50 cases
for each of eight task/version groups. The original 160-case pilot has completed
the full inference, transcription, automated judging, synchronized review, and
native-speaker review loop. The 240-case expansion has passed structural and
audio/timing validation but has not yet been run through GPT-Realtime or manually
reviewed at the same depth. Therefore the 400-case set is assembled, but the
240-case expansion should still be described as release-candidate data.

The dataset has not yet been uploaded to Hugging Face.

## Scope

The benchmark covers four v1.0 turn-taking tasks and four v1.5 overlap tasks:

- Pause handling
- Smooth turn-taking
- Backchannel generation
- User interruption
- Overlapping user interruption
- User backchannel
- Talking to another person
- Background speech

Scenarios span service/reception assistants, smart homes, in-car assistants, and
human-to-human conversations rather than being restricted to receptionist robots.

## What was reused from English FDB

We reused the benchmark's behavioral taxonomy and task split, not its English
sentences. We also reused these methodological ideas:

- v1.0 task definitions for pause handling, smooth turn-taking, backchanneling,
  and user interruption.
- v1.5 overlap tasks for interruption, user backchannel, talking to others, and
  background speech.
- Paired event and clean audio for v1.5, enabling counterfactual comparison.
- Event timestamps and task-specific metadata.
- ASR-aligned output transcripts as evaluation evidence.
- Timing concepts such as response latency and stop latency.
- Behavioral evaluation of whether the system continued, yielded, stopped, or
  engaged with an overlap event.
- Trailing silence so a streaming model has time to answer, following English
  FDB practice.
- Synchronized user/model timelines and reviewable audio artifacts.

The upstream English harness also contains task-level metrics, Silero-VAD timing,
LLM behavior classification, clean/event comparisons, speech-quality/prosody
features, and paired significance tests. We retained the task and paired-control
logic, but did not treat every upstream metric as a release requirement.

## What was adapted or rebuilt for Vietnamese

- Vietnamese-native text rather than literal translation.
- Vietnamese fillers, politeness particles, backchannels, hesitation, and repair.
- VieNeu-TTS generation with separately synthesized primary and event streams.
- Exact audio mixing and a shared monotonic streaming clock.
- Vietnamese ASR using ChunkFormer, with Whisper/PhoWhisper reserved for disputed
  cases and alignment support.
- A real GPT-Realtime streaming harness that preserves response creation,
  cancellation, arrival, and playout timing.
- A synchronized HTML review page with input, aligned assistant output, ASR, event
  interval, and judge evidence.
- A role-aware judge that hides the expected action and scores observable behavior.
- Vietnamese-specific failure categories: premature turn, missed interruption,
  engaged distractor, no speech, and correct continue/yield/take-turn behavior.

## End-to-end pilot validation

The 160-case pilot produced:

- 160/160 GPT-Realtime event-condition outputs.
- 80/80 clean-control outputs for applicable v1.5 cases.
- 160 event and 80 clean ChunkFormer transcripts.
- 160 automated judge records.
- A synchronized HTML page reviewed by a native Vietnamese speaker.
- 29 explicit examples retained as representative success cases and failure
  modes. Uncommented cases were not implicitly labeled as accepted.

This establishes that the harness can stream benchmark audio into a real
full-duplex system, record aligned output, transcribe it, judge it, and expose
mistakes for human audit.

## Benchmark in action: GPT-Realtime on the 160-case pilot

The first end-to-end run used GPT-Realtime as the evaluated speech-to-speech
system, ChunkFormer for Vietnamese output transcription, and a role-aware
GPT-4.1-mini judge. The table applies the 29 explicit human corrections where
available; all other rows retain the automated verdict. It is therefore a
calibrated pilot result, not yet a final leaderboard score.

| Task | Pass | Total | Pass rate |
|---|---:|---:|---:|
| v1.0 Backchannel | 20 | 20 | 100% |
| v1.0 Pause handling | 8 | 20 | 40% |
| v1.0 Smooth turn-taking | 17 | 20 | 85% |
| v1.0 User interruption | 8 | 20 | 40% |
| v1.5 Background speech | 6 | 20 | 30% |
| v1.5 Talking to other | 6 | 20 | 30% |
| v1.5 User backchannel | 20 | 20 | 100% |
| v1.5 User interruption | 16 | 20 | 80% |
| **Overall** | **101** | **160** | **63.1%** |

### English FDB harness compatibility

The GPT-Realtime pilot is also exported into the original English FDB
evaluation layout. We preserve its language-independent timing rules: at least
one second of detected speech, or more than three ASR chunks, counts as a turn;
negative latency is clipped to zero. The full aligned review stream is sliced
to the event-relative output expected by the upstream scripts.

| Upstream metric | Result |
|---|---:|
| Pause-handling take-over rate (lower is better) | 15% (3/20) |
| Pause-handling correct-wait rate | 85% (17/20) |
| Smooth turn-taking take-over rate | 90% (18/20) |
| Smooth turn-taking latency, conditional on response | 0.773 s |
| User-interruption post-event response rate | 45% (9/20) |
| User-interruption latency, conditional on response | 1.107 s |
| User-interruption relevance (localized English FDB 0–5 scale) | 2.05 / 5 |

These structural metrics complement rather than replace the 101/160 semantic
score. A model can wait through a mid-word pause yet omit the resumed word;
upstream timing credits the wait, while the Vietnamese semantic judge fails the
complete behavior.

The English backchannel JSD is not transferred because its ICC reference
distribution is language- and corpus-specific. English prompt-dependent
semantic metrics are localized to Vietnamese while retaining their task
definitions. Interruption relevance produced eleven scores of 0, one score of
1, and eight scores of 5. At a relevance threshold of 4, all 20 decisions agreed
with the stricter role-aware pass/fail verdict on this pilot. This is a sanity
check, not an assumption that the two metrics are generally identical.

Reproduction commands are provided by
`vi_fdb_harness/export_english_fdb_compat.py` and
`vi_fdb_harness/report_english_fdb_metrics.py`; detailed output is saved in
`outputs/vi_fdb_v1_0/english_fdb_compat/upstream_metrics.json`.

Observed behavior counts across the run were: 40 continue/resume, 24 yield and
follow the new request, 17 take the turn, 12 ignore a distractor, 9 wait then take
the turn, 28 engage a distractor, 12 miss an interruption, 10 produce no speech,
and 8 take a premature turn. Behavior names are descriptive; whether a behavior
passes depends on the task and event role.

The clearest contrast was between backchannel handling, where the model performed
well, and non-addressed speech, where it frequently treated background or side
speech as if it were directed to the assistant. Pause handling and changed-request
interruptions also exposed substantial premature-response and semantic-uptake
failures.

## Main findings

1. **Turn timing must use one clock.** API/WebSocket setup time must not be mixed
   with the audio timeline. Audio should be paced against absolute timestamps.
2. **Judge what the user hears.** An internal cancellation can still result in a
   correct recovery; cancellation alone is not a failure.
3. **Pause placement can change lexical meaning.** Splitting Vietnamese compounds
   such as “yên tĩnh” or “tổng quát” can cause a system to answer an incomplete
   prefix. These are genuine premature-turn failures, not merely ASR mistakes.
4. **Silence is not query completion.** Systems sometimes start on a mid-sentence
   pause and then fail to incorporate the rest of the request.
5. **Interruption success requires semantic uptake.** Stopping is insufficient;
   the response must follow the corrected or replacement request.
6. **Distractor handling is role-sensitive.** Background or side speech remains
   irrelevant even when topically related to the primary request. Acoustic source
   cues help the evaluated system; the judge may use the annotated event role,
   while the evaluated model must not receive that annotation.
7. **Clean/event pairs are valuable.** They show whether a behavioral change was
   caused by the overlap rather than ordinary model variability.
8. **ASR is evidence, not ground truth.** Vietnamese recognition errors and short
   or missing transcripts require conservative rules and selective human review.
9. **Human review found judge boundary errors.** Few-shot counterexamples and
   deterministic checks were added for premature response, missed changed request,
   no speech, and distractor engagement.
10. **Audio review remains necessary.** Schema checks cannot detect pronunciation,
    missing syllables, unnatural pauses, or implausible event placement.

## Remaining release work

Before calling the 400-case package a fully validated public release:

1. Run the 240 expansion cases through the same real-time harness.
2. Transcribe and judge their event and clean outputs.
3. Perform a stratified native-speaker audio/judge audit.
4. Freeze the manifest, schema, task counts, license, checksums, and dataset card.
5. Upload the versioned package and harness to a Hugging Face dataset repository.

The recommended public wording today is: **“Vi-FDB v1.0 release candidate: 400
balanced cases; the 160-case pilot has complete end-to-end system and human
validation, and the 240-case expansion has passed dataset/audio QC.”**
