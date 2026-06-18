"""Bake tileable PBR textures using a STANDARD optimum ONNX inpaint export.
Unlike the idmc inference-driven export (non-standard IO -> produced noise), optimum
exports AND runs the model with consistent IO, so the pipeline actually denoises.
Runs on CPU (avoids the onnxruntime-gpu cuDNN failure on the VAE encoder)."""
import sys, os, argparse
import numpy as np
from PIL import Image, ImageDraw

import transformers
if not hasattr(transformers, "CLIPFeatureExtractor"):
    _LM = type(transformers); _o = _LM.__getattr__
    _LM.__getattr__ = lambda s, n, _o=_o: (s.CLIPImageProcessor if n == "CLIPFeatureExtractor" else _o(s, n))

from optimum.onnxruntime import ORTPipelineForInpainting

MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
OUT_MODEL = "/dev/shm/sd-inpaint-opt-onnx"          # exported ONNX (RAM-backed, re-exports if gone)
OUT_TEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui", "assets", "textures")
MATERIALS = {
    "floor":  "seamless tileable dark grey marble floor tiles with grout lines, top-down flat lay, photorealistic, no objects",
    "wood":   "seamless tileable dark walnut wood grain texture, top-down flat lay, photorealistic, no objects",
    "stone":  "seamless tileable rough grey stone wall texture, top-down flat lay, photorealistic, no objects",
    "fabric": "seamless tileable deep red velvet fabric texture, top-down flat lay, photorealistic, no objects",
    "metal":  "seamless tileable dark brushed iron texture, top-down flat lay, photorealistic, no objects",
}
NEG = "text, watermark, seam, border, objects, people, shadow"


def load_pipe():
    if os.path.exists(os.path.join(OUT_MODEL, "model_index.json")):
        print("loading cached optimum export ...", flush=True)
        return ORTPipelineForInpainting.from_pretrained(OUT_MODEL, provider="CPUExecutionProvider")
    print(f"exporting {MODEL} via optimum (one-time) ...", flush=True)
    pipe = ORTPipelineForInpainting.from_pretrained(MODEL, export=True, provider="CPUExecutionProvider")
    pipe.save_pretrained(OUT_MODEL)
    print("export saved ->", OUT_MODEL, flush=True)
    return pipe


def tileable(pipe, prompt, size, steps, guidance):
    blank = Image.new("RGB", (size, size), (128, 128, 128))
    full = Image.new("L", (size, size), 255)
    base = pipe(prompt=prompt, negative_prompt=NEG, image=blank, mask_image=full,
                num_inference_steps=steps, guidance_scale=guidance, height=size, width=size).images[0]
    rolled = Image.fromarray(np.roll(np.array(base), (size // 2, size // 2), axis=(0, 1)))
    mask = Image.new("L", (size, size), 0); d = ImageDraw.Draw(mask); h = size // 2; b = max(24, size // 16)
    d.rectangle([h - b, 0, h + b, size], fill=255); d.rectangle([0, h - b, size, h + b], fill=255)
    out = pipe(prompt=prompt, negative_prompt=NEG, image=rolled, mask_image=mask,
               num_inference_steps=steps, guidance_scale=guidance, height=size, width=size).images[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.makedirs(OUT_TEX, exist_ok=True)
    pipe = load_pipe()
    mats = MATERIALS if not args.only else {k: MATERIALS[k] for k in args.only.split(",") if k in MATERIALS}
    for cat, prompt in mats.items():
        print(f"  baking {cat} ...", flush=True)
        tileable(pipe, prompt, args.size, args.steps, 7.5).save(os.path.join(OUT_TEX, f"{cat}.png"))
    print("DONE ->", OUT_TEX, flush=True)


if __name__ == "__main__":
    main()
