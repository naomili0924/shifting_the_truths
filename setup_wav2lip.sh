#!/usr/bin/env bash
# Set up Wav2Lip for the lip-sync talking head (talkgen.py): clone the repo, patch its
# 2020-era librosa call, fetch the GAN + s3fd checkpoints from a public mirror, and install
# opencv. Idempotent — safe to re-run. The repo lives OUTSIDE this game repo (default
# /workspace/Wav2Lip) and isn't committed; this script reconstructs it.
set -uo pipefail
PY="${PYTHON:-/venv/main/bin/python}"; [ -x "$PY" ] || PY=python
PIP="${PIP:-/venv/main/bin/pip}"; [ -x "$PIP" ] || PIP=pip
DIR="${WAV2LIP_DIR:-/workspace/Wav2Lip}"
export HF_HOME="${HF_HOME:-/dev/shm/hf}"
export HF_TOKEN="${HF_TOKEN:-}"

[ -d "$DIR/.git" ] || git clone --depth 1 https://github.com/Rudrabha/Wav2Lip "$DIR"

# librosa >= 0.10 requires keyword args for filters.mel (the repo passes positional).
sed -i 's/librosa.filters.mel(hp.sample_rate, hp.n_fft,/librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,/' "$DIR/audio.py"

"$PIP" install -q opencv-python-headless >/dev/null 2>&1 || true

# Checkpoints (public mirror): wav2lip_gan.pth (~415MB) + s3fd face detector (~85MB).
"$PY" - "$DIR" <<'PYEOF'
import sys, os, shutil
from huggingface_hub import hf_hub_download
DIR = sys.argv[1]; repo = "camenduru/Wav2Lip"
want = {"checkpoints/wav2lip_gan.pth": f"{DIR}/checkpoints/wav2lip_gan.pth",
        "face_detection/detection/sfd/s3fd.pth": f"{DIR}/face_detection/detection/sfd/s3fd.pth"}
for f, dst in want.items():
    if os.path.isfile(dst) and os.path.getsize(dst) > 1_000_000:
        print("have", dst); continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(hf_hub_download(repo, f), dst); print("fetched", dst)
PYEOF

echo "Wav2Lip ready -> $DIR"
