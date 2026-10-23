# Full-Duplex-Bench session handoff

Updated: 2026-09-03  
Repository: `/Users/tuandinh/repo/Full-Duplex-Bench`  
Branch/HEAD: `main` at `1eec748`

## Executive summary

Work has proceeded along two related tracks:

1. **Vi-FDB v1.0 benchmark and reference evaluation.** We built and published a
   400-case Vietnamese full-duplex benchmark, ran GPT-Realtime on all event
   cases and clean controls, added Vietnamese ASR/judging/review tooling, and
   aligned the language-independent timing metrics with upstream FDB.
2. **Vietnamese semantic dialogue manager.** We built a data-generation,
   transformation, training, evaluation, and failure-analysis pipeline for the
   four dialogue-manager actions used in the paper by Zhang et al. The current
   combined synthetic dataset has 1,716 conversations and 11,932 decisions. A
   Qwen3-0.6B baseline has been trained and evaluated internally.

The current priority is likely the second track: improve the dialogue-manager
baseline and evaluate it on a separately authored, truly unseen QA/QC set.

## 1. Vi-FDB v1.0: current state

### Completed

- Created 400 synthetic Vietnamese event cases across seven public tasks:
  backchannel (50), pause handling (50), smooth turn-taking (50), background
  speech (50), talking to other (50), user backchannel (50), and user
  interruption (100).
- Retained eight source task/version groups internally; the two interruption
  variants are merged into one public 100-case task.
- Included 200 paired clean controls for applicable overlap tasks.
- Fully exercised the 160-case pilot through inference, ASR, judging,
  synchronized review, and 29 explicit native-speaker adjudications.
- Ran the 240-case expansion through GPT-Realtime, PhoWhisper word timestamps,
  Silero VAD alignment, automated GPT-4.1-mini judging, and applicable upstream
  FDB metrics. It still needs the same depth of native-speaker audit as the
  pilot.
- Published the dataset, reference outputs, code, and explorer:
  - Dataset: <https://huggingface.co/datasets/tuanamz/vi-fdb-v1>
  - Reference outputs: <https://huggingface.co/datasets/tuanamz/vi-fdb-v1-gpt-realtime>
  - Explorer: <https://huggingface.co/spaces/tuanamz/vi-fdb-v1-explorer>
  - Harness: `vi_fdb_harness/`

### GPT-Realtime reference result

The run contains 400 event outputs and 200 clean-control outputs. The combined
role-aware semantic result is **257/400 = 64.25%**. Pilot verdicts include the
29 human corrections; expansion verdicts remain automated.

Recommended headline metrics:

| Capability | GPT-Realtime result |
|---|---:|
| Pause handling premature turn-over rate (lower is better) | 48% (24/50) |
| Smooth turn-taking response / conditional mean latency | 98% (49/50) / 1.000 s |
| Standard user-interruption response / conditional mean latency | 76% (38/50) / 0.662 s |
| Backchannel localized success | 100% (50/50) |
| User-backchannel continue-speaking success | 98% (49/50) |
| Background-speech ignore success | 18% (9/50) |
| Talking-to-other ignore success | 28% (14/50) |
| Paired interruption variant yield-and-follow success | 90% (45/50) |

Do not use English ICC backchannel JSD as a Vietnamese headline metric; there
is no Vietnamese human timing distribution. Treat the 64.25% semantic average
as an experimental cross-task summary, not an original-FDB headline measure.

### Important benchmark lessons

- Use one monotonic clock beginning at the first transmitted input frame.
- Pace input against absolute source timestamps to prevent accumulated drift.
- Score what the user hears, including playout and cancellation effects, rather
  than interpreting internal cancellation events as verdicts.
- Timing and semantic correctness must remain separate. A system may wait at
  the correct time but omit the resumed request, or stop for an interruption
  without following the replacement request.
- Event role and expected behavior are post-inference annotations and must never
  be exposed to the evaluated model.
- ASR is evidence rather than ground truth; ambiguous and no-speech cases need
  audio review.

### Authoritative files

- `docs/vi-fdb-v1-colleague-handoff.md`: current English user/colleague guide.
- `docs/vi-fdb-v1-colleague-handoff-vi.md`: current Vietnamese guide.
- `outputs/vi_fdb_v1_0/hf_gpt_realtime_reference/evaluation/summary.json`:
  combined semantic totals.
- `vi_fdb_harness/README.md`: harness commands and contracts.

`docs/vi-fdb-v1-status-and-findings.md` and
`docs/vietnamese-fdb-benchmark-creation-notes.md` are useful historical records
but are stale about release status and expansion completion. Do not use their
old “not uploaded” or “expansion not run” statements as current facts.

## 2. Vietnamese dialogue-manager track

### Objective and action space

The semantic dialogue manager consumes dialogue history, current assistant
state/ongoing speech, and streaming ASR, then predicts exactly one action:

- `<|C-L|>`: continue listening; user query is incomplete.
- `<|S-S|>`: start speaking; user query is complete.
- `<|S-L|>`: stop speaking and listen; real interruption.
- `<|C-S|>`: continue speaking; fake/non-disruptive interruption.

Vi-FDB's 400 cases are held out and were not copied into training data.

### Implemented pipeline

The `dialogue_manager/` package now supports:

- label-first Vietnamese conversation generation;
- author-recipe-style complete dialogue generation and deterministic cuts;
- punctuation/streaming-ASR variants;
- schema, state/action, ongoing-speech, language, duplication, and balance
  validation;
- conversation-held-out splits;
- label-only classifier training;
- native complete-sequence training using original control tokens;
- shared evaluation and HTML failure review.

The main scripts and their roles are summarized in `dialogue_manager/README.md`.
The comparison with the authors' repository is in
`dialogue_manager/docs/paper-repo-comparison.md`.

### Current dataset

`outputs/vi_dm_training/combined_v1/` combines:

- 796 accepted conversations from the initial pilot; and
- 920 accepted conversations from a paid, seven-model OpenRouter expansion.

Totals:

- 1,716 conversations;
- 11,932 decision examples;
- 10,764 training decisions from 1,544 conversations;
- 1,168 internal-eval decisions from 172 conversations;
- zero duplicate conversations/examples and zero train/eval conversation
  overlap.

Class counts are highly imbalanced:

| Action | Count |
|---|---:|
| `<|S-S|>` | 10,537 |
| `<|C-S|>` | 766 |
| `<|S-L|>` | 316 |
| `<|C-L|>` | 313 |

The new expansion used seven explicit paid model families: Gemini, Qwen, GPT,
DeepSeek, Mistral, Llama, and Claude. For a strict paid-only experiment, use
`outputs/vi_dm_training/paid_diverse_1000/paper_format/splits/`; for the largest
baseline, use `outputs/vi_dm_training/combined_v1/paper_format/splits/`.

### Baseline result

The current run is under
`outputs/vi_dm_training/training_runs/qwen3_0.6b_paper_baseline_v1/`.
On the 1,168-example internal split it reached **94.78% accuracy**. This is a
checkpoint-selection validation result, not final QA/QC performance.

The confusion matrix shows the main weaknesses:

- 9/31 true `<|C-L|>` cases were predicted as `<|S-S|>`.
- 20/32 true `<|S-L|>` cases were predicted as `<|C-S|>`.
- 9/77 true `<|C-S|>` cases were predicted as `<|S-L|>`.
- The majority `<|S-S|>` class is strong: 1005/1028 correct.

Accordingly, optimize and select checkpoints using macro-F1 and per-label recall,
not accuracy alone. The most serious issue is real-versus-fake interruption
confusion, especially the poor recall of `<|S-L|>`.

Failure artifacts:

- `outputs/vi_dm_training/training_runs/qwen3_0.6b_paper_baseline_v1/failure_analysis/summary.json`
- `outputs/vi_dm_training/training_runs/qwen3_0.6b_paper_baseline_v1/failure_analysis/predictions.jsonl`
- `outputs/vi_dm_training/training_runs/qwen3_0.6b_paper_baseline_v1/failure_analysis/failure_review.html`

### Recommended next steps

1. Inspect the failure-review page, focusing first on true `<|S-L|>` predicted
   as `<|C-S|>`, then `<|C-L|>` predicted as `<|S-S|>`.
2. Create a separately authored unseen QA/QC set. Do not report the internal
   172-conversation validation split as final generalization performance.
3. Establish macro-F1, per-class precision/recall/F1, and a balanced confusion
   matrix as the primary model-selection report.
4. Compare the current label-first baseline with the author-recipe ablation
   (`repo_recipe_v1_1544`) on the exact same held-out evaluation set.
5. Consider deterministic multi-event augmentation and controlled punctuation/
   ASR variants to increase minority-action density without contaminating eval.
6. Only after offline semantic-DM quality is credible, connect the selected
   checkpoint to the streaming speech stack and test behavioral impact on
   held-out audio.

## 3. Uncommitted working-tree state

The worktree is very dirty. Preserve user work and inspect paths individually
before staging anything.

Meaningful uncommitted source changes include:

- `.gitignore`: ignores a local VieNeu-TTS OpenRouter credential path.
- `v1_v1.5/model_inference/gpt-realtime/cli.js`: substantial rewrite from
  WebRTC to the Realtime WebSocket transport, environment-based API keys,
  24-kHz PCM streaming, absolute-time pacing, cancellation-aware playout, and
  timing JSON output.
- `v1_v1.5/model_inference/gpt-realtime/inference.sh`: invokes the new CLI and
  preserves its timing output.
- `v1_v1.5/model_inference/gpt-realtime/package.json` and lockfile: replace
  `node-fetch`/`wrtc` with `ws`.

There is also extensive tracked churn under
`v1_v1.5/model_inference/gpt-realtime/node_modules/`, apparently caused by
installing dependencies on a different platform. Do not bulk-stage or commit
that directory. Decide separately whether to restore/regenerate it after
confirming no user-owned edits would be lost.

Large untracked trees include `dialogue_manager/`, `fd_data_creation/`,
`outputs/`, `external/`, and documentation/PDF artifacts. Some contain important
research products, while others are generated data, environments, dependencies,
or credentials. Never use `git add .` here. In particular,
`docs/open router key.txt` appears credential-related and must not be committed
or pasted into logs or chat.

## 4. Quick orientation for the next session

Read these in order:

1. `docs/session-handoff-2026-09-03.md` (this file)
2. `docs/vi-fdb-v1-colleague-handoff.md`
3. `dialogue_manager/README.md`
4. `dialogue_manager/docs/paper-repo-comparison.md`
5. `outputs/vi_dm_training/combined_v1/README.md`
6. the baseline failure-analysis summary and HTML page

Before editing, run `git status --short` and scope every change narrowly. Avoid
cleaning, restoring, or staging the existing dirty tree without explicit user
direction.
