# Vietnamese Dialogue Manager

This directory contains the Vietnamese semantic dialogue-manager research
pipeline: synthetic conversation generation, paper-recipe transformations,
label-only and native-sequence training, evaluation, and failure review.

It is intentionally separate from `vi_fdb_harness/`, which is reserved for
creating and evaluating the Vietnamese Full-Duplex Benchmark.

## Main entry points

- `generate_training.py`: semantic label-first conversation generation.
- `generate_repo_dialogues.py`: complete conversations for deterministic cuts.
- `export_paper_format.py`: label-only classifier examples.
- `train_classifier.py`: four-action Qwen classifier training.
- `train_native.py`: original-token, complete-sequence training.
- `evaluate_native.py`: adapt a native checkpoint to the shared four-label eval.

Training artifacts remain under `outputs/vi_dm_training/`; benchmark artifacts
remain under `outputs/vi_fdb_v1_0/`.
