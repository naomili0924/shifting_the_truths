"""
ttsgen.py — optional, cached text-to-speech for Shifting Truth.

A thin, *optional* layer over the chatterbox-turbo ONNX pipeline
(``ORTChatterboxPipeline`` from the sibling ``inference_driven_model_compiler``
project). It turns a line of dialogue into a WAV on disk and hands back a
servable filename — the exact mirror of ``imagegen.py`` for audio.

Design rules (same as imagegen, so it is safe to wire into the web server):

  * **Optional & lazy.** The pipeline is loaded on first real use. If anything
    is missing (no GPU, no ONNX, package not importable, generation error) every
    call returns ``None`` and the game simply has no voice — exactly like the
    no-art and mock-LLM fallbacks.
  * **Cached.** Output is keyed by a hash of (text, voice). The same line in the
    same voice reuses the cached WAV, so re-asks cost nothing.
  * **Voice cloning, prepared once.** Each voice is a reference clip under
    ``voices_dir``; chatterbox conditioning is prepared once per voice and reused
    for every line in that voice (preparing is the slow part).
  * **Serialized.** The pipeline is stateful (shared conditioning + scheduler),
    so generation is guarded by a lock.

Nothing here ever raises into the caller.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import threading

logger = logging.getLogger("shifting_truth.ttsgen")

HERE = os.path.dirname(os.path.abspath(__file__))

# Emotion -> (temperature, time-stretch rate, pitch semitones, gain, tremor depth).
# chatterbox-turbo ignores exaggeration/cfg, so the feeling is shaped by temperature
# (passed to the model) plus light DSP on the wav: faster+higher+louder reads as
# angry/nervous, slower+lower+softer as sad/cold; nervous adds a faint tremor. Subtle on
# purpose — a hint of how the character feels, not a caricature.
_EMO = {
    "angry":     (0.90, 1.07,  1.0, 1.22, 0.00),
    "nervous":   (0.95, 1.12,  0.8, 1.05, 0.14),
    "defensive": (0.85, 1.05,  0.4, 1.06, 0.00),
    "cold":      (0.70, 0.97, -1.0, 0.90, 0.00),
    "sad":       (0.80, 0.90, -1.5, 0.85, 0.00),
    "calm":      (0.80, 1.00,  0.0, 1.00, 0.00),
}
_EMO_DEFAULT = (0.8, 1.0, 0.0, 1.0, 0.0)

# chatterbox's s3tokenizer ships float64 mel filters (from librosa) that clash
# with float32 STFT magnitudes during voice-cloning. Patch the class once, at
# import time, entirely from here (no edit to the chatterbox package).
_PATCHED = False


def _patch_s3tokenizer():
    global _PATCHED
    if _PATCHED:
        return
    try:
        import torch
        from chatterbox.models.s3tokenizer.s3tokenizer import S3Tokenizer
        _orig = S3Tokenizer.log_mel_spectrogram

        def _safe(self, audio, padding=0):
            if not torch.is_tensor(audio):
                audio = torch.from_numpy(audio)
            audio = audio.float()
            self._mel_filters = self._mel_filters.float()
            return _orig(self, audio, padding)

        S3Tokenizer.log_mel_spectrogram = _safe
        _PATCHED = True
    except Exception as exc:  # noqa: BLE001
        logger.debug("s3tokenizer patch skipped: %s", exc)


class TTSGen:
    """Lazy, cached, fail-soft wrapper around the chatterbox-turbo ONNX pipeline."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.model_dir = cfg.get("model_dir", "/workspace/models/chatterbox-turbo-onnx")
        self.idmc_path = cfg.get("idmc_path", "/workspace")
        self.provider = cfg.get("provider", "CUDAExecutionProvider")
        self.device = cfg.get("device", "cuda")
        self.cache_dir = cfg.get("cache_dir") or os.path.join(HERE, "webui", "assets", "cache", "audio")
        self.voices_dir = cfg.get("voices_dir") or os.path.join(HERE, "webui", "assets", "voices")
        if not os.path.isabs(self.cache_dir):
            self.cache_dir = os.path.join(HERE, self.cache_dir)
        if not os.path.isabs(self.voices_dir):
            self.voices_dir = os.path.join(HERE, self.voices_dir)
        # voice_id -> reference wav filename (resolved against voices_dir)
        self.voices = dict(cfg.get("voices") or {})

        self._pipe = None
        self._load_failed = False
        self._conds: dict[str, object] = {}      # voice_id -> prepared Conditionals
        self._lock = threading.Lock()
        os.makedirs(self.cache_dir, exist_ok=True)

    # ---- availability ------------------------------------------------
    def _model_present(self) -> bool:
        # The chatterbox ONNX bundle is four leaf folders; t3_backbone is the core.
        return os.path.isfile(os.path.join(self.model_dir, "t3_backbone", "model.onnx"))

    def _can_attempt(self) -> bool:
        return self.enabled and not self._load_failed and self._model_present()

    def available(self) -> bool:
        return self._can_attempt()

    def _voice_path(self, voice: str) -> str | None:
        fn = self.voices.get(voice)
        if not fn:
            return None
        p = fn if os.path.isabs(fn) else os.path.join(self.voices_dir, fn)
        return p if os.path.isfile(p) else None

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
                _patch_s3tokenizer()
                from inference_driven_model_compiler.ort_chatterbox import (
                    ORTChatterboxPipeline,
                )
                logger.info("Loading chatterbox-turbo ONNX pipeline from %s", self.model_dir)
                self._pipe = ORTChatterboxPipeline.from_pretrained(
                    self.model_dir, device=self.device, provider=self.provider
                )
                logger.info("chatterbox-turbo ONNX pipeline ready.")
            except Exception as exc:  # noqa: BLE001 - never propagate
                self._load_failed = True
                logger.warning("Voice disabled (TTS pipeline load failed): %s", exc)
                self._pipe = None
        return self._pipe

    def warm(self, voices=None):
        """Pre-load the pipeline and prepare the given voices (default: all
        configured), so the first real line isn't slow. Best-effort, never raises."""
        if not self._can_attempt():
            return
        if self._ensure_pipe() is None:
            return
        for v in (voices or list(self.voices)):
            self._prepare_voice(v)

    def _prepare_voice(self, voice: str):
        """Prepare (and cache) the conditioning for *voice*; return it or None."""
        if voice in self._conds:
            return self._conds[voice]
        path = self._voice_path(voice)
        if path is None:
            return None
        pipe = self._ensure_pipe()
        if pipe is None:
            return None
        try:
            pipe.tts.prepare_conditionals(path)
            self._conds[voice] = pipe.tts.conds
            return self._conds[voice]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Voice prep failed for %r: %s", voice, exc)
            return None

    # ---- cache -------------------------------------------------------
    def _cache_name(self, text: str, voice: str, emotion: str | None = None) -> str:
        # Emotion shapes the audio, so it is part of the cache key — the same line said
        # nervously vs angrily caches apart.
        h = hashlib.sha1(f"{voice}|{emotion or ''}|{text}".encode("utf-8")).hexdigest()[:20]
        return f"{h}.wav"

    def cached(self, text: str, voice: str, emotion: str | None = None) -> str | None:
        name = self._cache_name(text, voice, emotion)
        return name if os.path.isfile(os.path.join(self.cache_dir, name)) else None

    def url_name(self, text: str, voice: str, emotion: str | None = None) -> str:
        """Deterministic servable filename for (text, voice, emotion) — known before the
        WAV exists, so the caller can hand the browser a URL to poll."""
        return self._cache_name(text, voice, emotion)

    # ---- generation --------------------------------------------------
    def _shape(self, y, sr, rate, semis, gain, tremor):
        """Light DSP to colour the delivery with emotion (see _EMO). Fail-soft: on any
        error the unshaped audio is returned."""
        try:
            import numpy as np, librosa
            y = np.asarray(y, dtype="float32")
            if abs(rate - 1.0) > 1e-3:
                y = librosa.effects.time_stretch(y, rate=float(rate))
            if abs(semis) > 1e-3:
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(semis))
            if tremor > 0 and len(y):
                t = np.arange(len(y)) / sr
                y = y * (1.0 - tremor * (0.5 + 0.5 * np.sin(2 * np.pi * 6.5 * t)))
            if abs(gain - 1.0) > 1e-3:
                y = y * float(gain)
            m = float(np.max(np.abs(y))) if len(y) else 0.0
            if m > 0.99:
                y = y * (0.99 / m)               # avoid clipping after gain/shift
            return y.astype("float32")
        except Exception as exc:  # noqa: BLE001
            logger.warning("emotion DSP skipped: %s", exc)
            return y

    def generate(self, text: str, voice: str, emotion: str | None = None) -> str | None:
        """Synthesize *text* in *voice* and return a servable WAV filename.

        *emotion* (nervous/angry/...) colours the delivery so the feeling lands in the
        sound. Returns the cached filename instantly if present; otherwise generates,
        caches, and returns it. Returns ``None`` on any failure — the caller must treat a
        missing clip as "no voice for this line".
        """
        text = (text or "").strip()
        if not text or not self._can_attempt():
            return None

        name = self._cache_name(text, voice, emotion)
        path = os.path.join(self.cache_dir, name)
        if os.path.isfile(path):
            return name

        pipe = self._ensure_pipe()
        if pipe is None:
            return None

        temperature, rate, semis, gain, tremor = _EMO.get(emotion, _EMO_DEFAULT)
        with self._lock:
            if os.path.isfile(path):          # another thread may have just made it
                return name
            try:
                conds = self._prepare_voice(voice)
                if conds is None:
                    return None
                pipe.tts.conds = conds        # select this voice (reuse prepared conds)
                wav = pipe.generate(text, temperature=temperature)   # (1, N) torch @ pipe.sr
                arr = wav.squeeze(0).detach().cpu().numpy()
                arr = self._shape(arr, int(pipe.sr), rate, semis, gain, tremor)
                import soundfile as sf
                tmp = path + ".tmp"
                sf.write(tmp, arr, int(pipe.sr), format="WAV")  # .wav.tmp -> specify format
                os.replace(tmp, path)          # atomic publish
                return name
            except Exception as exc:  # noqa: BLE001
                logger.warning("TTS failed for %r (%r): %s", text[:50], voice, exc)
                return None


# Per-language registry, configured once by the web layer (mirrors imagegen).
# Voice is English-only (chatterbox-turbo) — a language with no entry returns
# None, so e.g. a Chinese game simply has no audio.
_CFG: dict | None = None
_INSTANCES: dict[str, TTSGen] = {}


def configure(cfg: dict | None) -> dict | None:
    global _CFG, _INSTANCES
    _CFG = dict(cfg or {})
    _INSTANCES = {}
    return _CFG


def instance(lang: str = "en") -> TTSGen | None:
    """Return the (lazily built) TTSGen for *lang*, or None if unavailable.

    Unlike imagegen there is no en fallback: a language without its own voice
    config gets no audio (chatterbox-turbo only speaks English)."""
    if _CFG is None or not _CFG.get("enabled", True):
        return None
    by_lang = _CFG.get("by_lang") or {}
    if by_lang and lang not in by_lang:
        return None
    if lang not in _INSTANCES:
        base = {k: v for k, v in _CFG.items() if k != "by_lang"}
        merged = {**base, **(by_lang.get(lang, {}))}
        _INSTANCES[lang] = TTSGen(merged)
    return _INSTANCES[lang]


def dispose() -> None:
    """Release loaded voice pipeline(s) to reclaim GPU memory; rebuilds lazily on
    the next generate(). Called by the web server when no session is active."""
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


def voice_for(name: str, assign: dict | None, default: str) -> str:
    """Map a speaker display name to a voice id (config 'assign'), else default."""
    if assign and name in assign:
        return assign[name]
    return default


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="I was on the terrace when she fell.")
    ap.add_argument("--voice", default="male_a")
    ap.add_argument("--model-dir", default="/workspace/models/chatterbox-turbo-onnx")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure({"by_lang": {"en": {
        "model_dir": args.model_dir,
        "voices": {args.voice: f"{args.voice}.wav"},
    }}})
    g = instance("en")
    print("available:", g.available())
    name = g.generate(args.text, args.voice)
    print("result:", name, "->", os.path.join(g.cache_dir, name) if name else "(none)")
