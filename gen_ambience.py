"""gen_ambience.py — synthesize an ORIGINAL, royalty-free background ambience loop for the
game's stormy-night noir setting: rain, wind, distant thunder, and a low uneasy drone.

Entirely procedural (numpy) — no samples, no copyright — and crossfaded so it loops
seamlessly. Output: webui/assets/music/ambience.ogg (committed; it's original).

    source /venv/main/bin/activate
    python gen_ambience.py
"""
from __future__ import annotations
import os, subprocess, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "webui", "assets", "music")
SR = 44100
DUR = 70.0          # loop body length (seconds)
SEED = 7


def _norm(x):
    m = float(np.max(np.abs(x)))
    return x / m if m > 0 else x


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    N = int(SR * DUR)
    t = np.arange(N) / SR

    # --- rain: high-passed noise (crude diff), with a slow swell ---
    w = rng.standard_normal(N + 1)
    rain = _norm(np.diff(w))
    rain *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t)
    rain *= 0.18

    # --- wind: brown noise (cumsum) low-passed via moving average, slow swells ---
    b = np.cumsum(rng.standard_normal(N)); b = _norm(b - b.mean())
    k = 400
    wind = np.convolve(b, np.ones(k) / k, mode="same")
    wind = _norm(wind) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.04 * t + 1.0))
    wind *= 0.22

    # --- drone: a low minor chord with slow beating + a quiet eerie upper tone ---
    def tone(f, a, lfo=0.03):
        env = 0.6 + 0.4 * np.sin(2 * np.pi * lfo * t + rng.random() * 6.0)
        vib = 0.4 * np.sin(2 * np.pi * 0.11 * t)               # gentle vibrato
        return a * env * np.sin(2 * np.pi * f * t + vib)
    drone = (tone(55.0, 0.5) + tone(65.41, 0.4) + tone(82.41, 0.35) + tone(110.0, 0.15))
    drone += 0.05 * np.sin(2 * np.pi * 77.78 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.02 * t))
    drone = _norm(drone) * 0.16

    # --- thunder: a few distant low rumbles at random times ---
    thunder = np.zeros(N)
    for _ in range(4):
        start = int(rng.integers(0, max(1, N - SR * 7)))
        dur = int(SR * (3 + rng.random() * 3))
        rum = np.cumsum(rng.standard_normal(dur)); rum = _norm(rum - rum.mean())
        env = np.exp(-np.linspace(0, 4, dur))
        thunder[start:start + dur] += rum * env * 0.6
    thunder = _norm(thunder) * 0.25

    mix = _norm(rain + wind + drone + thunder) * 0.5            # headroom; it's background

    # --- seamless loop: crossfade the tail into the head ---
    xf = int(3 * SR)
    head = mix[:xf].copy(); tail = mix[-xf:].copy()
    body = mix[:-xf].copy()
    fade = np.linspace(0.0, 1.0, xf)
    body[:xf] = tail * (1.0 - fade) + head * fade

    tmp_wav = os.path.join(tempfile.gettempdir(), "ambience.wav")
    import soundfile as sf
    sf.write(tmp_wav, body.astype("float32"), SR)

    out = os.path.join(OUT_DIR, "ambience.ogg")
    # Prefer Ogg/Vorbis (loops cleanly, small); fall back to AAC .m4a if vorbis is absent.
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_wav,
                        "-c:a", "libvorbis", "-q:a", "4", out], check=True)
    except Exception:
        out = os.path.join(OUT_DIR, "ambience.m4a")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_wav,
                        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", out], check=True)
    print("wrote", out, os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    main()
