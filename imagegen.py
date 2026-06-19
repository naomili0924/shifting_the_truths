"""
imagegen.py — manifest-driven scene/portrait generation for Shifting Truth, built on an
INPAINTING diffusion model (exported to ONNX via optimum; see export_inpaint.py).

Why inpainting: a scene is no longer a flat backdrop. We build it in layers so clue items
and decoy props are painted *into* the picture (realistic, embedded — not separate cards):

  * **Base scene**  = inpaint a grey canvas through a full-white mask (acts as txt2img).
  * **Embed object**= inpaint a white box at the object's (x,y,w,h) with an object prompt,
    onto the evolving canvas; objects accumulate on one image.
  * **Evidence crop** = the object's box cropped out of the finished scene — the pouch
    thumbnail is literally the object as it appears in the room.

Same safety contract as before, so it stays safe to wire into the web server:

  * **Optional & lazy** — the pipeline loads on first real use; if anything is missing
    (no GPU/ONNX, model dir absent, generation error) calls return ``None``/empty and the
    game falls back to a plain backdrop with object chips — exactly the no-art path.
  * **Cached** — keyed by a content hash; identical prompts reuse the cached PNG.
  * **Serialized** — the ORT pipeline is stateful per call, so generation is guarded by a
    lock; callers get concurrency by hiding the batch behind the intro, not by hammering
    the GPU.

Nothing here ever raises into the caller.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading

logger = logging.getLogger("shifting_truth.imagegen")

HERE = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_STYLE = (
    "moody noir painted illustration, dramatic candlelight, deep shadows, "
    "point-and-click adventure game background art, cinematic, atmospheric, "
    "no people, no text"
)
_NEG = "people, person, hands, text, watermark, signature, frame, border, blurry, lowres, deformed"

_TF_PATCHED = False


def _patch_transformers_compat():
    """transformers 5.x renamed CLIPFeatureExtractor -> CLIPImageProcessor, but the
    diffusion runtime still does `from transformers import CLIPFeatureExtractor`. Alias it
    via the lazy module's __getattr__ so the import resolves. Idempotent, best-effort."""
    global _TF_PATCHED
    if _TF_PATCHED:
        return
    try:
        import transformers
        if not hasattr(transformers, "CLIPFeatureExtractor"):
            _LM = type(transformers)
            _orig = _LM.__getattr__

            def _aliased(self, name, _orig=_orig):
                if name == "CLIPFeatureExtractor":
                    return self.CLIPImageProcessor
                return _orig(self, name)

            _LM.__getattr__ = _aliased
        _TF_PATCHED = True
    except Exception as exc:  # noqa: BLE001
        logger.debug("transformers compat patch skipped: %s", exc)


class ImageGen:
    """Lazy, cached, fail-soft wrapper around an ONNX inpainting pipeline."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.model_dir = cfg.get("model_dir", "/dev/shm/sdxl-inpaint")
        self.device = cfg.get("device", "cuda")
        self.dtype = cfg.get("dtype", "float16")
        self.variant = cfg.get("variant", "fp16")
        self.style = cfg.get("style", _DEFAULT_STYLE)
        self.negative = cfg.get("negative", _NEG)
        self.width = int(cfg.get("width", 768))
        self.height = int(cfg.get("height", 768))
        self.steps = int(cfg.get("steps", 20))
        self.guidance = float(cfg.get("guidance", 7.5))
        self.cache_dir = cfg.get("cache_dir") or os.path.join(HERE, "webui", "assets", "cache")

        self._pipe = None       # inpaint pipeline (objects)
        self._txt = None        # txt2img pipeline (base scenes, faces)
        self._load_failed = False
        self._lock = threading.Lock()
        os.makedirs(self.cache_dir, exist_ok=True)

    # ---- availability ------------------------------------------------
    def _can_attempt(self) -> bool:
        return (
            self.enabled
            and not self._load_failed
            and os.path.isfile(os.path.join(self.model_dir, "model_index.json"))
        )

    def available(self) -> bool:
        """True if generation is plausible (config + exported model present).
        Does not force a load — the pipeline import/load happens lazily."""
        return self._can_attempt()

    # ---- pipeline (lazy) ---------------------------------------------
    def _ensure_pipe(self):
        if self._pipe is not None or self._load_failed:
            return self._pipe
        with self._lock:
            if self._pipe is not None or self._load_failed:
                return self._pipe
            try:
                _patch_transformers_compat()
                import torch
                from diffusers import AutoPipelineForText2Image, AutoPipelineForInpainting
                use_cuda = (self.device == "cuda" and torch.cuda.is_available())
                dtype = torch.float16 if (self.dtype == "float16" and use_cuda) else torch.float32
                kw = {"torch_dtype": dtype, "use_safetensors": True}
                if self.variant:
                    kw["variant"] = self.variant
                logger.info("Loading diffusers SDXL pipelines from %s", self.model_dir)
                try:
                    txt = AutoPipelineForText2Image.from_pretrained(self.model_dir, **kw)
                except Exception:
                    kw.pop("variant", None)   # snapshot may lack an fp16 variant
                    txt = AutoPipelineForText2Image.from_pretrained(self.model_dir, **kw)
                dev = "cuda" if use_cuda else "cpu"
                txt = txt.to(dev)
                # Build the inpaint pipeline from the SAME weights (no extra VRAM/download):
                # base scenes come from reliable txt2img; objects are embedded by inpaint.
                inp = AutoPipelineForInpainting.from_pipe(txt).to(dev)
                for p in (txt, inp):
                    try: p.set_progress_bar_config(disable=True)
                    except Exception: pass
                    try: p.safety_checker = None
                    except Exception: pass
                self._txt = txt
                self._pipe = inp
                logger.info("SDXL txt2img + inpaint ready (device=%s, dtype=%s).", dev, dtype)
            except Exception as exc:  # noqa: BLE001 - never propagate
                self._load_failed = True
                logger.warning("Image generation disabled (pipeline load failed): %s", exc)
                self._pipe = None
                self._txt = None
        return self._pipe

    # ---- low-level generation (caller holds self._lock) --------------
    def _gen_txt(self, prompt, steps=None, guidance=None, negative=None):
        """Generate a full image from *prompt* via the txt2img pipeline (base scenes/faces)."""
        out = self._txt(
            prompt=prompt,
            negative_prompt=(self.negative if negative is None else negative),
            num_inference_steps=int(steps or self.steps),
            guidance_scale=float(guidance if guidance is not None else self.guidance),
            width=self.width,
            height=self.height,
        )
        return out.images[0]

    def _run(self, image, mask, prompt, steps=None, guidance=None):
        """One inpaint pass: repaint the white area of *mask* on *image* from *prompt*."""
        out = self._pipe(
            prompt=prompt,
            negative_prompt=self.negative,
            image=image,
            mask_image=mask,
            num_inference_steps=int(steps or self.steps),
            guidance_scale=float(guidance if guidance is not None else self.guidance),
            strength=0.99,
            width=self.width,
            height=self.height,
        )
        return out.images[0]

    def _box_mask(self, box):
        """A feathered white rectangle mask for a normalized (cx, cy, w, h) box."""
        from PIL import Image, ImageDraw, ImageFilter
        W, H = self.width, self.height
        cx, cy, w, h = box
        x0 = int(max(0, (cx - w / 2) * W)); x1 = int(min(W, (cx + w / 2) * W))
        y0 = int(max(0, (cy - h / 2) * H)); y1 = int(min(H, (cy + h / 2) * H))
        if x1 <= x0 or y1 <= y0:
            return None, None
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).rectangle([x0, y0, x1, y1], fill=255)
        m = m.filter(ImageFilter.GaussianBlur(max(4, (x1 - x0) // 12)))
        return m, (x0, y0, x1, y1)

    # ---- cache helpers ----------------------------------------------
    def _hash(self, *parts) -> str:
        key = "|".join(str(p) for p in parts)
        key += f"|{self.style}|{self.width}x{self.height}|s{self.steps}|g{self.guidance}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]

    # ---- public: single image (faces / portraits) -------------------
    def generate(self, prompt: str, negative: str | None = None,
                 style: str | None = None) -> str | None:
        """Paint *prompt* as a full image and return a servable PNG filename, or None on
        any failure. ``style`` overrides the scene style (pass "" for none) — the faces use
        a photoreal style so the talking-head lip-sync looks real."""
        if not self._can_attempt():
            return None
        st = self.style if style is None else style
        full = f"{prompt}, {st}" if st else prompt
        neg = negative if negative is not None else self.negative
        name = self._hash("img", full) + ".png"
        path = os.path.join(self.cache_dir, name)
        if os.path.isfile(path):
            return name
        if self._ensure_pipe() is None:
            return None
        with self._lock:
            if os.path.isfile(path):
                return name
            try:
                img = self._gen_txt(full, negative=neg)
                tmp = path + ".tmp"; img.save(tmp, format="PNG"); os.replace(tmp, path)
                return name
            except Exception as exc:  # noqa: BLE001
                logger.warning("generate() failed for %r: %s", prompt[:60], exc)
                return None

    # ---- public: a composed scene -----------------------------------
    def compose_scene(self, base_prompt: str, objects: list[dict]):
        """Paint a scene: a base backdrop, then each object inpainted into its box.

        ``objects`` items: {"id", "obj_prompt", "x", "y", "w", "h"} (x,y centre and w,h
        are normalized 0..1). Returns ``(scene_filename, {obj_id: crop_filename})``; the
        scene is cached as a whole, and each object's crop is the evidence thumbnail.
        Returns ``(None, {})`` on any failure (caller falls back to a plain backdrop)."""
        if not self._can_attempt():
            return None, {}
        spec = [(o.get("id"), o.get("obj_prompt", ""),
                 round(o.get("x", .5), 3), round(o.get("y", .5), 3),
                 round(o.get("w", .25), 3), round(o.get("h", .25), 3)) for o in objects]
        full_base = f"{base_prompt}, {self.style}" if self.style else base_prompt
        h = self._hash("scene", full_base, json.dumps(spec, sort_keys=True))
        scene_name = h + ".png"
        scene_path = os.path.join(self.cache_dir, scene_name)
        crops = {o.get("id"): f"{h}_{o.get('id')}.png" for o in objects}
        # Fast path: scene + every crop already on disk.
        if os.path.isfile(scene_path) and all(
                os.path.isfile(os.path.join(self.cache_dir, c)) for c in crops.values()):
            return scene_name, crops
        if self._ensure_pipe() is None:
            return None, {}
        with self._lock:
            if os.path.isfile(scene_path) and all(
                    os.path.isfile(os.path.join(self.cache_dir, c)) for c in crops.values()):
                return scene_name, crops
            try:
                canvas = self._gen_txt(full_base)   # reliable txt2img base scene
                boxes = {}
                for o in objects:
                    box = (o.get("x", .5), o.get("y", .5), o.get("w", .25), o.get("h", .25))
                    mask, px = self._box_mask(box)
                    if mask is None:
                        continue
                    op = o.get("obj_prompt") or o.get("name", "an object")
                    op = f"{op}, {self.style}" if self.style else op
                    try:
                        canvas = self._run(canvas, mask, op)
                        boxes[o.get("id")] = px
                    except Exception as exc:  # noqa: BLE001 - one bad object never kills the scene
                        logger.warning("object inpaint failed (%s): %s", o.get("id"), exc)
                tmp = scene_path + ".tmp"; canvas.save(tmp, format="PNG"); os.replace(tmp, scene_path)
                # Crop each successfully-placed object out of the finished scene.
                out_crops = {}
                for oid, c in crops.items():
                    px = boxes.get(oid)
                    cpath = os.path.join(self.cache_dir, c)
                    try:
                        if px:
                            canvas.crop(px).save(cpath + ".tmp", format="PNG")
                            os.replace(cpath + ".tmp", cpath)
                            out_crops[oid] = c
                    except Exception:  # noqa: BLE001
                        pass
                return scene_name, out_crops
            except Exception as exc:  # noqa: BLE001
                logger.warning("compose_scene failed for %r: %s", base_prompt[:60], exc)
                return None, {}


# Per-language registry, configured once by the web layer. With an inpainting backbone one
# English-prompt model serves every language (rooms.py builds prompts in English), but the
# by_lang shape is kept so a language can still override size/style/model if desired.
_CFG: dict | None = None
_INSTANCES: dict[str, ImageGen] = {}


def configure(cfg: dict | None) -> dict | None:
    """Store the image config. Shape:
        {enabled, provider, cache_dir, by_lang: {en: {model_dir, width, height, steps,
         guidance, style}, zh: {...}}}
    Keys outside ``by_lang`` are shared defaults merged into each language."""
    global _CFG, _INSTANCES
    _CFG = dict(cfg or {})
    _INSTANCES = {}
    return _CFG


def instance(lang: str = "en") -> ImageGen | None:
    """Return the (lazily built) ImageGen for *lang*, or None if disabled."""
    if _CFG is None or not _CFG.get("enabled", True):
        return None
    by_lang = _CFG.get("by_lang") or {}
    key = lang if lang in by_lang else ("en" if "en" in by_lang else "_default")
    if key not in _INSTANCES:
        base = {k: v for k, v in _CFG.items() if k != "by_lang"}
        merged = {**base, **(by_lang.get(key, {}))}
        _INSTANCES[key] = ImageGen(merged)
    return _INSTANCES[key]


def dispose() -> None:
    """Release loaded ONNX pipeline(s) to reclaim GPU memory. Safe to call anytime —
    instances rebuild lazily on the next call. Used by the web server to free VRAM when
    no game session is active."""
    global _INSTANCES
    for inst in list(_INSTANCES.values()):
        try:
            inst._pipe = None
            inst._txt = None
            inst._load_failed = False
        except Exception:  # noqa: BLE001
            pass
    _INSTANCES = {}
    try:
        import gc; gc.collect()
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    # Smoke test: paint a base scene + a 2-object composition into the cache.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?",
                    default="a dim wine cellar with an old archive shelf and a tool bench")
    ap.add_argument("--model-dir", default="/dev/shm/sdxl-base")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure({"by_lang": {"en": {"model_dir": args.model_dir}}})
    g = instance("en")
    print("available:", g.available())
    print("face:", g.generate(args.prompt))
    scene, crops = g.compose_scene(args.prompt, [
        {"id": "wrench", "obj_prompt": "a greasy steel wrench on the bench",
         "x": 0.3, "y": 0.7, "w": 0.22, "h": 0.18},
        {"id": "bottle", "obj_prompt": "an empty dusty wine bottle",
         "x": 0.7, "y": 0.72, "w": 0.18, "h": 0.26},
    ])
    print("scene:", scene, "crops:", crops)
