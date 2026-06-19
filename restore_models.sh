#!/usr/bin/env bash
# Restore the models the painted UI + voice need, after an instance restart.
#
#   scene art : SDXL  -> /dev/shm/sdxl-base   (public; fetched by export_inpaint.py)
#               ONE model serves EN + ZH: txt2img paints the base scene, and an inpaint
#               pipeline built from the same weights embeds the clue/decoy objects.
#   voice     : chatterbox-turbo (private HF repo) -> /workspace/models/chatterbox-turbo-onnx
#
# Prereq: a Hugging Face token for the (private) voice repo:
#   hf auth login            (or export HF_TOKEN=hf_...)
# NOTE: the old `huggingface-cli` is removed in huggingface_hub >= 1.0 — use `hf`.
# RAM-backed (/dev/shm) art is wiped on restart — re-run this (or restore_after_reboot.sh).
set -euo pipefail

PY="${PYTHON:-/venv/main/bin/python}"; [ -x "$PY" ] || PY=python
HF="${HF_BIN:-/venv/main/bin/hf}"; [ -x "$HF" ] || HF=hf
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep big downloads off the small container overlay (use the RAM disk if present).
export HF_HOME="${HF_HOME:-/dev/shm/hf}"
export TMPDIR="${TMPDIR:-/dev/shm/tmp}"
mkdir -p "$HF_HOME" "$TMPDIR"

TTS_REPO="${TTS_REPO:-Jinyan0924/chatterbox-turbo-onnx}"
TTS_DIR="${TTS_DIR:-/workspace/models/chatterbox-turbo-onnx}"

echo "Fetching SDXL scene model -> /dev/shm/sdxl-base"
"$PY" "${HERE}/export_inpaint.py"

echo "Setting up Wav2Lip (lip-sync talking head)"
bash "${HERE}/setup_wav2lip.sh" || echo "  (Wav2Lip setup failed — talking head will fall back to static portraits)"

# Voice (text-to-speech) for conversation lines. English only (~1 GB). Skip with SKIP_TTS=1.
if [ "${SKIP_TTS:-0}" != "1" ]; then
  echo "Restoring chatterbox-turbo (voice) -> ${TTS_DIR}"
  "$HF" download "${TTS_REPO}" --repo-type model --local-dir "${TTS_DIR}" \
    --include "ve/*" "s3gen_estimator/*" "s3gen_hift/*" "t3_backbone/*"
fi

echo "Done. scene art -> /dev/shm/sdxl-base ; voice -> ${TTS_DIR}"
