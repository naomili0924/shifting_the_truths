"""
imagegen.py — manifest-driven backdrop/portrait generation for Shifting Truth.

A thin, *optional* layer over the inference-driven SDXL-Turbo ONNX pipeline
exported by the sibling `inference_driven_model_compiler` project. It turns a
text prompt (always built upstream from the room/character **manifest**, never
the other way round) into a PNG on disk and hands back a servable filename.

Design rules that keep it safe to wire into the web server:

  * **Optional & lazy.** The pipeline is loaded on first real use, in a
    background-friendly way. If anything is missing (no GPU, no ONNX, package
    not importable, generation error) every call returns ``None`` and the game
    falls back to a plain backdrop — exactly like the mock-LLM fallback.
  * **Cached.** Output is keyed by a hash of (prompt, size, style). Identical
    prompts — the five fixed faces, a re-rolled-but-identical room — reuse the
    cached PNG, so the marginal cost trends to zero.
  * **Serialized.** The diffusers scheduler the pipeline reuses is stateful per
    call, so generation is guarded by a lock; callers get concurrency by hiding
    the (fast, 1-step) batch behind the intro, not by hammering the GPU.

Nothing here ever raises into the caller.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import threading

logger = logging.getLogger("shifting_truth.imagegen")

HERE = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_STYLE = (
    "moody noir painted illustration, dramatic candlelight, deep shadows, "
    "point-and-click adventure game background art, cinematic, atmospheric, "
    "no people, no text"
)


_TF_PATCHED = False


def _patch_transformers_compat():
    """transformers 5.x removed CLIPFeatureExtractor (renamed to CLIPImageProcessor),
    but the diffusion runtime still does `from transformers import CLIPFeatureExtractor`.
    Alias it via the lazy module's __getattr__ so that import resolves — done here
    (game side), not by editing the sibling pipeline. Idempotent, best-effort."""
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
    """Lazy, cached, fail-soft wrapper around the SDXL-Turbo ONNX pipeline."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.model_dir = cfg.get("model_dir", "/workspace/models/sdxl-turbo-onnx")
        self.idmc_path = cfg.get("idmc_path", "/workspace")
        self.provider = cfg.get("provider", "CUDAExecutionProvider")
        self.style = cfg.get("style", _DEFAULT_STYLE)
        self.width = int(cfg.get("width", 512))
        self.height = int(cfg.get("height", 512))
        self.steps = int(cfg.get("steps", 1))
        self.guidance = float(cfg.get("guidance", 0.0))
        self.cache_dir = cfg.get("cache_dir") or os.path.join(HERE, "webui", "assets", "cache")

        self._pipe = None
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
        """True if generation is plausible (config + model files present).

        Does not force a load — the actual pipeline import/load happens lazily.
        """
        return self._can_attempt()

    # ---- pipeline (lazy) ---------------------------------------------
    def _ensure_pipe(self):
        if self._pipe is not None or self._load_failed:
            return self._pipe
        with self._lock:
            if self._pipe is not None or self._load_failed:
                return self._pipe
            try:
                if self.idmc_path and self.idmc_path not in sys.path:
                    sys.path.insert(0, self.idmc_path)
                _patch_transformers_compat()
                from inference_driven_model_compiler.optimum.onnxruntime.modeling_diffusion import (
                    ORTDiffusionPipeline,
                )
                logger.info("Loading SDXL-Turbo ONNX pipeline from %s", self.model_dir)
                self._pipe = ORTDiffusionPipeline.from_pretrained(
                    self.model_dir, provider=self.provider, export=False
                )
                logger.info("SDXL-Turbo ONNX pipeline ready.")
            except Exception as exc:  # noqa: BLE001 - never propagate
                self._load_failed = True
                logger.warning("Image generation disabled (pipeline load failed): %s", exc)
                self._pipe = None
        return self._pipe

    # ---- cache -------------------------------------------------------
    def _cache_name(self, prompt: str) -> str:
        h = hashlib.sha1(
            f"{prompt}|{self.style}|{self.width}x{self.height}|s{self.steps}|g{self.guidance}".encode("utf-8")
        ).hexdigest()[:20]
        return f"{h}.png"

    def cached(self, prompt: str) -> str | None:
        """Return the servable filename if this prompt is already painted."""
        name = self._cache_name(prompt)
        return name if os.path.isfile(os.path.join(self.cache_dir, name)) else None

    # ---- generation --------------------------------------------------
    def generate(self, prompt: str, negative: str | None = None) -> str | None:
        """Paint *prompt* (style appended) and return a servable PNG filename.

        Returns the cached filename instantly if present, otherwise generates,
        caches, and returns it. Returns ``None`` on any failure — the caller
        must treat a missing image as "no art, use a plain backdrop".
        """
        if not self._can_attempt():
            return None

        full_prompt = f"{prompt}, {self.style}" if self.style else prompt
        name = self._cache_name(full_prompt)
        path = os.path.join(self.cache_dir, name)
        if os.path.isfile(path):
            return name

        pipe = self._ensure_pipe()
        if pipe is None:
            return None

        with self._lock:
            # Re-check the cache inside the lock: another thread may have just
            # painted the same prompt while we waited.
            if os.path.isfile(path):
                return name
            try:
                kwargs = dict(
                    prompt=full_prompt,
                    num_inference_steps=self.steps,
                    guidance_scale=self.guidance,
                    width=self.width,
                    height=self.height,
                )
                if negative:
                    kwargs["negative_prompt"] = negative
                image = pipe(**kwargs).images[0]
                tmp = path + ".tmp"
                image.save(tmp, format="PNG")
                os.replace(tmp, path)  # atomic publish
                return name
            except Exception as exc:  # noqa: BLE001
                logger.warning("Image generation failed for prompt %r: %s", prompt[:60], exc)
                return None


# Per-language registry, configured once by the web layer. Each language maps to
# its own model + params (EN -> SDXL-Turbo, ZH -> Hunyuan-DiT). Pipelines are
# built lazily on first use, so an English game never loads the Chinese model and
# vice-versa.
_CFG: dict | None = None
_INSTANCES: dict[str, ImageGen] = {}


def configure(cfg: dict | None) -> dict | None:
    """Store the image config. Shape:

        {enabled, idmc_path, provider, cache_dir,
         by_lang: {en: {model_dir, width, height, steps, guidance, style}, zh: {...}}}

    Keys outside ``by_lang`` are shared defaults merged into each language. A flat
    config (no ``by_lang``) is treated as the config for every language.
    """
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
    instances rebuild lazily on the next generate(). Used by the web server to free
    VRAM when no game session is active."""
    global _INSTANCES
    for inst in list(_INSTANCES.values()):
        try:
            inst._pipe = None
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
    # Smoke test: paint one backdrop to the cache and print its path.
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="a dim wine cellar with an old archive shelf on the right and a tool bench on the left")
    ap.add_argument("--model-dir", default="/workspace/models/sdxl-turbo-onnx")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure({"by_lang": {"en": {"model_dir": args.model_dir}}})
    g = instance("en")
    print("available:", g.available())
    name = g.generate(args.prompt)
    print("result:", name, "->", os.path.join(g.cache_dir, name) if name else "(none)")
