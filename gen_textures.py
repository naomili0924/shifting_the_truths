# NOTE: superseded by bake_optimum.py. The idmc inference-driven export produced a
# pipeline with non-standard IO that did not denoise (noise output). bake_optimum.py
# uses the standard `optimum` inpaint export, which works. Kept for the custom-model path.
"""
gen_textures.py — bake tileable PBR material textures for the 3D UI using an
INPAINTING diffusion model (exported to ONNX via inference_driven_model_compiler).

Why inpainting: a generated swatch has visible seams when tiled. We roll the image
by half so the seams sit in the centre, paint a mask over those seam lines, and let
the inpainting model repaint them coherently — the result tiles cleanly. That is the
"inpainting strategy" for scenario textures.

Output → webui/assets/textures/<cat>.png  (floor, wood, stone, fabric, metal)
The 3D client (game3d.html) probes /assets/textures/floor.png at startup and, when
present, wraps the floor/walls/furniture in these instead of flat PBR colours. If the
folder is absent it silently falls back to the procedural look — nothing breaks.

USAGE (after you export an inpainting model to ONNX):
    source /venv/main/bin/activate
    python gen_textures.py --model-dir /dev/shm/sdxl-inpaint-onnx --size 1024 --steps 20

The pipeline is loaded exactly like imagegen.py loads SDXL (the vendored
inference_driven_model_compiler ORTDiffusionPipeline auto-detects the inpaint class
from the model's model_index.json). VALIDATE the call signature against your export
the first time — different inpaint exports may name the mask arg `mask_image`.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "webui", "assets", "textures")

# material -> prompt (top-down, flat, tileable surface — no objects/lighting)
MATERIALS = {
    "floor":  "seamless tileable floor of dark grey marble tiles with thin grout lines, top-down flat lay, even lighting, photorealistic, no objects",
    "wood":   "seamless tileable dark walnut wood grain texture, top-down flat lay, even lighting, photorealistic, no objects",
    "stone":  "seamless tileable rough grey stone wall texture, top-down flat lay, even lighting, photorealistic, no objects",
    "fabric": "seamless tileable deep red velvet fabric texture, top-down flat lay, soft, photorealistic, no objects",
    "metal":  "seamless tileable dark brushed iron metal texture, top-down flat lay, photorealistic, no objects",
}
NEG = "text, watermark, seam, border, frame, objects, people, shadow, vignette"


def load_pipe(model_dir, provider, idmc_path):
    if idmc_path and idmc_path not in sys.path:
        sys.path.insert(0, idmc_path)
    # transformers-5 compat (same shim imagegen uses)
    try:
        import transformers
        if not hasattr(transformers, "CLIPFeatureExtractor"):
            _LM = type(transformers); _orig = _LM.__getattr__
            def _a(self, name, _orig=_orig):
                return self.CLIPImageProcessor if name == "CLIPFeatureExtractor" else _orig(self, name)
            _LM.__getattr__ = _a
    except Exception:
        pass
    from inference_driven_model_compiler.optimum.onnxruntime.modeling_diffusion import ORTDiffusionPipeline
    # The ORT VAE encoder returns a ModelOutput(latent_sample=...) which diffusers'
    # retrieve_latents (expects .latent_dist/.latents) can't read — shim it.
    try:
        import torch as _t
        import diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint as _M
        try:
            from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution as _DGD
        except Exception:
            from diffusers.models.vae import DiagonalGaussianDistribution as _DGD
        def _rl(enc, generator=None, sample_mode="sample"):
            if hasattr(enc, "latent_dist"):
                return enc.latent_dist.sample(generator) if sample_mode == "sample" else enc.latent_dist.mode()
            # ORT VAE encoder returns ModelOutput(output=8-ch moments) — wrap + sample
            m = enc.to_tuple()[0] if hasattr(enc, "to_tuple") else (enc[0] if isinstance(enc,(tuple,list)) else enc)
            if _t.is_tensor(m) and m.shape[1] == 8:
                d = _DGD(m); return d.sample(generator) if sample_mode == "sample" else d.mode()
            return m
        _M.retrieve_latents = _rl
    except Exception as e:
        print("retrieve_latents shim skipped:", e)
    print(f"loading inpaint pipeline from {model_dir} (provider={provider}) ...")
    return ORTDiffusionPipeline.from_pretrained(model_dir, provider=provider, export=False)


def seam_mask(size, band):
    """White cross over the centre seam lines (where edges meet after a half-roll)."""
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    h = size // 2
    d.rectangle([h - band, 0, h + band, size], fill=255)   # vertical seam
    d.rectangle([0, h - band, size, h + band], fill=255)   # horizontal seam
    return m


def make_tileable(pipe, prompt, size, steps, guidance):
    # 1) base swatch — inpaint a blank canvas with a full mask (acts as txt2img)
    blank = Image.new("RGB", (size, size), (128, 128, 128))
    full = Image.new("L", (size, size), 255)
    base = pipe(prompt=prompt, negative_prompt=NEG, image=blank, mask_image=full,
                num_inference_steps=steps, guidance_scale=guidance,
                width=size, height=size).images[0]
    # 2) roll by half so seams are centred, then inpaint the seam cross -> tiles cleanly
    rolled = Image.fromarray(np.roll(np.array(base), (size // 2, size // 2), axis=(0, 1)))
    mask = seam_mask(size, max(24, size // 16))
    out = pipe(prompt=prompt, negative_prompt=NEG, image=rolled, mask_image=mask,
               num_inference_steps=steps, guidance_scale=guidance,
               width=size, height=size).images[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="ONNX inpainting model dir (model_index.json inside)")
    ap.add_argument("--idmc-path", default="/workspace")
    ap.add_argument("--provider", default="CUDAExecutionProvider")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--only", default="", help="comma-separated subset of materials")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    pipe = load_pipe(args.model_dir, args.provider, args.idmc_path)
    mats = MATERIALS if not args.only else {k: MATERIALS[k] for k in args.only.split(",") if k in MATERIALS}
    for cat, prompt in mats.items():
        print(f"  • {cat}")
        img = make_tileable(pipe, prompt, args.size, args.steps, args.guidance)
        img.save(os.path.join(OUT, f"{cat}.png"))
    print(f"done -> {OUT}  (reload the 3D UI; it will use them automatically)")


if __name__ == "__main__":
    main()
