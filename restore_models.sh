#!/usr/bin/env bash
# Restore the exported ONNX image models so the painted Phaser UI works without
# re-exporting them. The models are large (SDXL ~7 GB, Hunyuan ~22 GB) and live
# in private Hugging Face repos; download them to the locations config.yaml's
# `images.by_lang` points at:
#   en -> SDXL-Turbo  -> /dev/shm/sdxl-turbo-onnx (RAM-backed: re-run after a restart)
#   zh -> Hunyuan-DiT -> /dev/shm/hunyuan-onnx   (RAM-backed: re-run after a restart)
# Plus the voice model, and an optional INPAINTING model used by gen_textures.py to
# bake the tileable PBR textures for the 3D UI (the inpaint entry is fail-soft — it
# just warns if you haven't exported/uploaded that model yet). After restoring the
# inpaint model, run:  python gen_textures.py --model-dir "$INPAINT_DIR"
#
# Prereq: a Hugging Face token with read access to the repos:
#   huggingface-cli login
#
# Override the repos/dirs with env vars if you forked them (e.g. INPAINT_REPO, INPAINT_DIR).
set -euo pipefail

SDXL_REPO="${SDXL_REPO:-Jinyan0924/sdxl-turbo-onnx}"
HUNYUAN_REPO="${HUNYUAN_REPO:-Jinyan0924/hunyuan-dit-onnx}"
TTS_REPO="${TTS_REPO:-Jinyan0924/chatterbox-turbo-onnx}"
INPAINT_REPO="${INPAINT_REPO:-Jinyan0924/sdxl-inpaint-onnx}"
SDXL_DIR="${SDXL_DIR:-/dev/shm/sdxl-turbo-onnx}"
HUNYUAN_DIR="${HUNYUAN_DIR:-/dev/shm/hunyuan-onnx}"
TTS_DIR="${TTS_DIR:-/workspace/models/chatterbox-turbo-onnx}"
INPAINT_DIR="${INPAINT_DIR:-/dev/shm/sdxl-inpaint-onnx}"

echo "Restoring SDXL-Turbo -> ${SDXL_DIR}"
huggingface-cli download "${SDXL_REPO}" --repo-type model --local-dir "${SDXL_DIR}"

echo "Restoring Hunyuan-DiT -> ${HUNYUAN_DIR}"
huggingface-cli download "${HUNYUAN_REPO}" --repo-type model --local-dir "${HUNYUAN_DIR}"

# Voice (text-to-speech) for conversation lines — config.yaml's audio.by_lang.en
# points here. English only (~1 GB). Skip with SKIP_TTS=1 if you don't want voice.
if [ "${SKIP_TTS:-0}" != "1" ]; then
  echo "Restoring chatterbox-turbo (voice) -> ${TTS_DIR}"
  huggingface-cli download "${TTS_REPO}" --repo-type model --local-dir "${TTS_DIR}" \
    --include "ve/*" "s3gen_estimator/*" "s3gen_hift/*" "t3_backbone/*"
fi

# Inpainting model — feeds gen_textures.py, which bakes the tileable PBR textures for
# the 3D UI (webui/assets/textures/). RAM-backed: re-run after a restart. Skip with
# SKIP_INPAINT=1. Optional & fail-soft: if you haven't exported/uploaded it yet, this
# step warns and continues so the rest of the restore still completes.
if [ "${SKIP_INPAINT:-0}" != "1" ]; then
  echo "Restoring inpainting model -> ${INPAINT_DIR}"
  huggingface-cli download "${INPAINT_REPO}" --repo-type model --local-dir "${INPAINT_DIR}" \
    || echo "  (inpainting model not found at ${INPAINT_REPO} yet — skipping; export+upload it, then re-run / run gen_textures.py)"
fi

echo "Done. EN art -> ${SDXL_DIR} ; ZH art -> ${HUNYUAN_DIR} ; voice -> ${TTS_DIR} ; inpaint -> ${INPAINT_DIR}"
