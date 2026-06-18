#!/usr/bin/env bash
# Bring the browser game (art + voice) back after an instance restart.
#
# What survives a *stop/start* reboot: the /venv/main pip installs, the
# /etc/ld.so.conf.d CUDA-12 registration, /workspace/.env, and (if /workspace
# is a host volume) the voice model. What does NOT: anything under /dev/shm
# (RAM-backed) — i.e. the SDXL and Hunyuan image models. A full *recycle*
# wipes the container filesystem, so the pip stack would need reinstalling
# first (see SETUP_AND_PLAY.md); this script only restores models + config and
# (re)starts the server, all idempotently — safe to run repeatedly.
#
# Usage:  cd /workspace/shifting_the_truths && ./restore_after_reboot.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VENV="${VENV:-/venv/main}"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
ENV_FILE="${ENV_FILE:-/workspace/.env}"
WEB_PORT="${WEB_PORT:-17080}"

# 1) Secrets / keys (ANTHROPIC_API_KEY for the LLM, HF_TOKEN for downloads).
if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARN: HF_TOKEN not set (needed to pull private art/voice models). Put it in $ENV_FILE."
fi

# 2) Re-register the bundled CUDA-12 libs so onnxruntime-gpu finds cublas/cudnn
#    (the torch cu128 wheel ships them under site-packages/nvidia/*/lib). The
#    box is CUDA 13.x; the wheels are CUDA 12 — without this, art/voice fall back.
NV="$($PY -c 'import os,nvidia;print(os.path.dirname(nvidia.__file__))' 2>/dev/null)"
if [ -n "$NV" ] && ls -d "$NV"/*/lib >/dev/null 2>&1; then
  ls -d "$NV"/*/lib > /etc/ld.so.conf.d/onnx-cuda12.conf
  ldconfig
  echo "Registered CUDA-12 libs from $NV"
fi

# 3) torchaudio must match torch's CUDA build (the default PyPI wheel is cu13 and
#    crashes the TTS pipeline against torch cu128). Fix if import fails.
if $PY -c "import torch" >/dev/null 2>&1; then
  if ! $PY -c "import torchaudio" >/dev/null 2>&1; then
    echo "Fixing torchaudio -> cu128 to match torch..."
    $PIP install --no-deps --force-reinstall "torchaudio==2.11.0+cu128" \
      --index-url https://download.pytorch.org/whl/cu128
  fi
fi

# 4) Re-pull the models (idempotent: hf download skips files already present).
#    SDXL + Hunyuan land in /dev/shm (wiped every restart); voice in /workspace.
if [ -n "${HF_TOKEN:-}" ]; then
  ./restore_models.sh || echo "WARN: restore_models.sh reported an error (continuing)."
fi

# 5) (Re)start the web server.
[ -f /tmp/webpy.pid ] && kill "$(cat /tmp/webpy.pid)" 2>/dev/null
sleep 1
source "$VENV/bin/activate"
nohup python web.py > /tmp/web.log 2>&1 & echo $! > /tmp/webpy.pid
for i in $(seq 1 30); do
  curl -sf "http://127.0.0.1:${WEB_PORT}/" -o /dev/null 2>/dev/null && { echo "web up (pid $(cat /tmp/webpy.pid))"; break; }
  sleep 0.5
done

# 6) (Re)open a Cloudflare quick tunnel and print the public URL. All external
#    ports are usually in use on this box, so the authed Caddy edge isn't an
#    option — the quick tunnel is public + ephemeral (no token). For a private
#    link instead, use SSH forwarding (see SETUP_AND_PLAY.md).
CF="${CF:-/opt/instance-tools/bin/cloudflared}"
if [ -x "$CF" ]; then
  pkill -f "cloudflared tunnel --url http://127.0.0.1:${WEB_PORT}" 2>/dev/null
  nohup "$CF" tunnel --url "http://127.0.0.1:${WEB_PORT}" > /tmp/cf_tunnel.log 2>&1 &
  for i in $(seq 1 30); do
    URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cf_tunnel.log | head -1)
    [ -n "$URL" ] && break
    sleep 1
  done
  echo "PLAY AT: ${URL:-<tunnel not ready — check /tmp/cf_tunnel.log>}"
fi
echo "Done."
