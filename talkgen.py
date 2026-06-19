"""
talkgen.py — optional, cached lip-sync video for Shifting Truth.

Turns a (portrait image, spoken-line WAV) into a short MP4 of that suspect with their
MOUTH moving in sync with the speech — so a talking suspect, not a static card. A thin
layer over Wav2Lip (cloned at /workspace/Wav2Lip), same safety contract as imagegen/ttsgen:

  * **Optional & lazy** — the Wav2Lip model + face detector load on first real use; if
    anything is missing (no GPU, repo/checkpoints absent, generation error) every call
    returns ``None`` and the UI just shows the static portrait.
  * **Warm** — model + face detector are loaded once and reused; each suspect's face box
    is detected once and cached, so per-reply work is just the mel→frames pass + ffmpeg.
  * **Cached** — output keyed by (portrait, audio); the same line reuses the MP4.
  * **Serialized** — generation is guarded by a lock (the model is stateful on the GPU).

Nothing here ever raises into the caller.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
import tempfile
import threading

logger = logging.getLogger("shifting_truth.talkgen")

HERE = os.path.dirname(os.path.abspath(__file__))
_MEL_STEP = 16
_IMG_SIZE = 96


class TalkGen:
    """Lazy, cached, fail-soft Wav2Lip lip-sync video generator."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.wav2lip_dir = cfg.get("wav2lip_dir", "/workspace/Wav2Lip")
        self.checkpoint = cfg.get("checkpoint") or os.path.join(self.wav2lip_dir, "checkpoints", "wav2lip_gan.pth")
        self.device = cfg.get("device", "cuda")
        self.fps = int(cfg.get("fps", 25))
        self.batch_size = int(cfg.get("batch_size", 128))
        self.out_height = int(cfg.get("out_height", 512))   # downscale portrait for speed
        self.preset = cfg.get("preset", "veryfast")          # x264 speed/quality tradeoff
        self.pads = cfg.get("pads", [0, 12, 0, 0])     # top,bottom,left,right around the face
        self.cache_dir = cfg.get("cache_dir") or os.path.join(HERE, "webui", "assets", "cache", "video")
        if not os.path.isabs(self.cache_dir):
            self.cache_dir = os.path.join(HERE, self.cache_dir)
        self.tmp = cfg.get("tmp_dir") or tempfile.gettempdir()

        self._model = None
        self._detector = None
        self._audio = None           # Wav2Lip's audio module
        self._load_failed = False
        self._boxes: dict[str, tuple] = {}    # portrait path -> (y1,y2,x1,x2)
        self._lock = threading.Lock()
        os.makedirs(self.cache_dir, exist_ok=True)

    # ---- availability ------------------------------------------------
    def _can_attempt(self) -> bool:
        return (self.enabled and not self._load_failed
                and os.path.isfile(self.checkpoint)
                and os.path.isdir(self.wav2lip_dir))

    def available(self) -> bool:
        return self._can_attempt()

    # ---- model + detector (lazy, warm) ------------------------------
    def _ensure(self):
        if (self._model is not None and self._detector is not None) or self._load_failed:
            return self._model
        with self._lock:
            if (self._model is not None and self._detector is not None) or self._load_failed:
                return self._model
            try:
                if self.wav2lip_dir not in sys.path:
                    sys.path.append(self.wav2lip_dir)   # append so it can't shadow site-packages
                import torch
                import audio as w2l_audio               # Wav2Lip/audio.py
                import face_detection                   # Wav2Lip/face_detection
                from models import Wav2Lip              # Wav2Lip/models
                dev = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
                logger.info("Loading Wav2Lip + face detector (device=%s)", dev)
                ckpt = torch.load(self.checkpoint, map_location="cpu")
                model = Wav2Lip()
                model.load_state_dict({k.replace("module.", ""): v
                                       for k, v in ckpt["state_dict"].items()})
                self._model = model.to(dev).eval()
                self._detector = face_detection.FaceAlignment(
                    face_detection.LandmarksType._2D, flip_input=False, device=dev)
                self._audio = w2l_audio
                self._dev = dev
                logger.info("Wav2Lip ready.")
            except Exception as exc:  # noqa: BLE001 - never propagate
                self._load_failed = True
                logger.warning("Lip-sync disabled (Wav2Lip load failed): %s", exc)
                self._model = None
        return self._model

    def warm(self):
        """Load the model + detector AND pay the one-time CUDA/cuDNN init with a dummy
        detect + forward (each is ~10s cold, ~0.03s after) so the first real reply is fast.
        Best-effort, never raises."""
        if not self._can_attempt() or self._ensure() is None:
            return
        try:
            import numpy as np, torch
            # Warm s3fd at the SAME size as real (square, out_height) frames — it
            # re-initialises per input size, so warming at the right size matters.
            oh = self.out_height
            self._detector.get_detections_for_batch(np.zeros((1, oh, oh, 3), np.uint8))
            # face_detection force-sets cudnn.benchmark=True, which makes wav2lip
            # re-autotune (~11s) on every new batch shape — turn it OFF after.
            torch.backends.cudnn.benchmark = False
            ib = torch.zeros(self.batch_size, 6, _IMG_SIZE, _IMG_SIZE, device=self._dev)
            mb = torch.zeros(self.batch_size, 1, 80, _MEL_STEP, device=self._dev)
            with torch.no_grad():
                self._model(mb, ib)
            if self._dev == "cuda":
                torch.cuda.synchronize()
            logger.info("Wav2Lip warmed.")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Wav2Lip warm skipped: %s", exc)

    # ---- per-portrait face box (detected once, cached) --------------
    def _face_box(self, portrait_path, frame):
        if portrait_path in self._boxes:
            return self._boxes[portrait_path]
        import numpy as np
        det = self._detector.get_detections_for_batch(np.array([frame[:, :, ::-1]]))  # BGR->RGB
        rect = det[0] if det else None
        if rect is None:
            return None
        pady1, pady2, padx1, padx2 = self.pads
        y1 = max(0, rect[1] - pady1); y2 = min(frame.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1); x2 = min(frame.shape[1], rect[2] + padx2)
        box = (int(y1), int(y2), int(x1), int(x2))
        self._boxes[portrait_path] = box
        return box

    # ---- cache -------------------------------------------------------
    def _cache_name(self, portrait_path, audio_path) -> str:
        key = f"{os.path.basename(portrait_path)}|{os.path.basename(audio_path)}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20] + ".mp4"

    def cached(self, portrait_path, audio_path) -> str | None:
        name = self._cache_name(portrait_path, audio_path)
        return name if os.path.isfile(os.path.join(self.cache_dir, name)) else None

    def url_name(self, portrait_path, audio_path) -> str:
        return self._cache_name(portrait_path, audio_path)

    # ---- generation --------------------------------------------------
    def generate(self, portrait_path, audio_path) -> str | None:
        """Lip-sync *portrait_path* to *audio_path* -> servable MP4 filename, or None."""
        if not self._can_attempt():
            return None
        if not (portrait_path and audio_path and os.path.isfile(portrait_path)
                and os.path.isfile(audio_path)):
            return None
        name = self._cache_name(portrait_path, audio_path)
        path = os.path.join(self.cache_dir, name)
        if os.path.isfile(path):
            return name
        if self._ensure() is None:
            return None
        with self._lock:
            if os.path.isfile(path):
                return name
            proc = None
            try:
                import numpy as np, cv2, torch
                frame = cv2.imread(portrait_path)
                if frame is None:
                    return None
                # downscale the portrait once (fewer pixels to predict/encode = much faster)
                if frame.shape[0] > self.out_height:
                    sc = self.out_height / frame.shape[0]
                    frame = cv2.resize(frame, (int(round(frame.shape[1] * sc)), self.out_height))
                box = self._face_box(portrait_path, frame)   # may run s3fd (sets benchmark=True)
                if box is None:
                    return None
                torch.backends.cudnn.benchmark = False       # keep wav2lip off the autotune path
                y1, y2, x1, x2 = box
                face = cv2.resize(frame[y1:y2, x1:x2], (_IMG_SIZE, _IMG_SIZE))

                wav = self._audio.load_wav(audio_path, 16000)
                mel = self._audio.melspectrogram(wav)
                if np.isnan(mel.reshape(-1)).sum() > 0:
                    return None
                # chunk the mel to one slice per output frame at self.fps
                mel_chunks = []
                mult = 80.0 / self.fps
                i = 0
                while True:
                    s = int(i * mult)
                    if s + _MEL_STEP > mel.shape[1]:
                        mel_chunks.append(mel[:, mel.shape[1] - _MEL_STEP:])
                        break
                    mel_chunks.append(mel[:, s:s + _MEL_STEP])
                    i += 1

                h, w = frame.shape[:2]
                # one-pass: pipe raw BGR frames straight into ffmpeg (no DIVX intermediate,
                # no double-encode) and mux the audio in the same call.
                tmp_mp4 = path + ".tmp.mp4"
                proc = subprocess.Popen(
                    ["ffmpeg", "-y", "-loglevel", "error",
                     "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                     "-r", str(self.fps), "-i", "-", "-i", audio_path,
                     "-c:v", "libx264", "-preset", self.preset, "-pix_fmt", "yuv420p",
                     "-movflags", "+faststart", "-c:a", "aac", "-shortest", tmp_mp4],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                bs = self.batch_size
                for k in range(0, len(mel_chunks), bs):
                    chunk = mel_chunks[k:k + bs]
                    img_batch = np.asarray([face] * len(chunk))
                    mel_batch = np.asarray(chunk)
                    masked = img_batch.copy(); masked[:, _IMG_SIZE // 2:] = 0
                    img_batch = np.concatenate((masked, img_batch), axis=3) / 255.0
                    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1],
                                                       mel_batch.shape[2], 1])
                    ib = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self._dev)
                    mb = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self._dev)
                    with torch.no_grad():
                        pred = self._model(mb, ib)
                    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
                    for p in pred:
                        f = frame.copy()
                        f[y1:y2, x1:x2] = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                        proc.stdin.write(f.astype(np.uint8).tobytes())
                proc.stdin.close()
                if proc.wait() != 0:
                    logger.warning("ffmpeg failed: %s", proc.stderr.read().decode()[:200])
                    return None
                os.replace(tmp_mp4, path)
                return name
            except Exception as exc:  # noqa: BLE001
                logger.warning("Lip-sync failed (%s): %s",
                               os.path.basename(audio_path), exc)
                try:
                    if proc and proc.poll() is None:
                        proc.kill()
                except Exception:  # noqa: BLE001
                    pass
                return None


# ---- module registry (mirrors imagegen/ttsgen) ----------------------
_CFG: dict | None = None
_INSTANCE: TalkGen | None = None


def configure(cfg: dict | None) -> dict | None:
    global _CFG, _INSTANCE
    _CFG = dict(cfg or {})
    _INSTANCE = None
    return _CFG


def instance() -> TalkGen | None:
    """Return the (lazily built) TalkGen, or None if disabled. Lip-sync is
    language-independent (it only needs a face + a wav), so there's one instance."""
    global _INSTANCE
    if _CFG is None or not _CFG.get("enabled", True):
        return None
    if _INSTANCE is None:
        _INSTANCE = TalkGen(_CFG)
    return _INSTANCE


def dispose() -> None:
    global _INSTANCE
    if _INSTANCE is not None:
        try:
            _INSTANCE._model = None
            _INSTANCE._detector = None
            _INSTANCE._load_failed = False
        except Exception:  # noqa: BLE001
            pass
    _INSTANCE = None
    try:
        import gc; gc.collect()
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", default="/tmp/face.png")
    ap.add_argument("--audio", default="/tmp/test.wav")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure({"cache_dir": "/tmp/talkcache"})
    g = instance()
    print("available:", g.available())
    import time
    t = time.time()
    name = g.generate(args.face, args.audio)
    print("result:", name, "in %.1fs" % (time.time() - t),
          "->", os.path.join(g.cache_dir, name) if name else "(none)")
