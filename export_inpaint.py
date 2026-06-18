"""export_inpaint.py — fetch the SDXL model used to build the painted UI.

Scenes are built from ONE SDXL checkpoint, run natively via diffusers on the GPU (torch,
fp16) rather than ONNX (onnxruntime-gpu mis-shapes the inpaint UNet; CPU ONNX is too slow).
The base scene is reliable txt2img; the clue/decoy objects are embedded by an inpaint
pipeline built from the SAME weights (diffusers `from_pipe`, no extra VRAM or download).
One English model serves both EN and ZH (rooms.py builds every image prompt in English).

    source /venv/main/bin/activate
    python export_inpaint.py                 # -> /dev/shm/sdxl-base

The model lives in /dev/shm (RAM-backed) — re-run after an instance restart. The download
reuses the HF cache, so it's near-instant if the weights are already cached.
"""
from __future__ import annotations
import os

from huggingface_hub import snapshot_download

MODEL = os.environ.get("INPAINT_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
OUT = os.environ.get("INPAINT_DIR", "/dev/shm/sdxl-base")


def main():
    if os.path.isfile(os.path.join(OUT, "model_index.json")):
        print(f"already present -> {OUT}", flush=True)
        return
    print(f"fetching {MODEL} -> {OUT} ...", flush=True)
    snapshot_download(
        MODEL, local_dir=OUT,
        # fp16 weights + all configs/tokenizers (skip the heavier fp32 duplicates)
        allow_patterns=["*.json", "*.txt", "**/*.json", "**/*.txt",
                        "**/*.fp16.safetensors", "**/*fp16*"],
    )
    print(f"DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
