#!/usr/bin/env bash
# Restore the exported ONNX image models so the painted Phaser UI works without
# re-exporting them. The models are large (SDXL ~7 GB, Hunyuan ~22 GB) and live
# in private Hugging Face repos; download them to the locations config.yaml's
# `images.by_lang` points at:
#   en -> SDXL-Turbo  -> /workspace/models/sdxl-turbo-onnx
#   zh -> Hunyuan-DiT -> /dev/shm/hunyuan-onnx   (RAM-backed: re-run after a restart)
#
# Prereq: a Hugging Face token with read access to the repos:
#   huggingface-cli login
#
# Override the repos with env vars if you forked them.
set -euo pipefail

SDXL_REPO="${SDXL_REPO:-Jinyan0924/sdxl-turbo-onnx}"
HUNYUAN_REPO="${HUNYUAN_REPO:-Jinyan0924/hunyuan-dit-onnx}"
SDXL_DIR="${SDXL_DIR:-/workspace/models/sdxl-turbo-onnx}"
HUNYUAN_DIR="${HUNYUAN_DIR:-/dev/shm/hunyuan-onnx}"

echo "Restoring SDXL-Turbo -> ${SDXL_DIR}"
huggingface-cli download "${SDXL_REPO}" --repo-type model --local-dir "${SDXL_DIR}"

echo "Restoring Hunyuan-DiT -> ${HUNYUAN_DIR}"
huggingface-cli download "${HUNYUAN_REPO}" --repo-type model --local-dir "${HUNYUAN_DIR}"

echo "Done. EN -> ${SDXL_DIR} ; ZH -> ${HUNYUAN_DIR}"
