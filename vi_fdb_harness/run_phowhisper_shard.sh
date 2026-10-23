#!/usr/bin/env bash
set -euo pipefail

case "${SLURM_LOCALID:?SLURM_LOCALID is required}" in
  0) start=0 ;;
  1) start=180 ;;
  *) echo "Unexpected SLURM_LOCALID: ${SLURM_LOCALID}" >&2; exit 2 ;;
esac

exec .venv/bin/python transcribe.py \
  --root ../outputs/vi_fdb_v1_0/gpt_realtime_240 \
  --backend phowhisper \
  --start "${start}" \
  --limit 180 \
  --device 0
