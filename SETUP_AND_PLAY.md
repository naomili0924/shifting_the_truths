# Setup & Play

## 🌐 Play in the browser (running now, on Claude)

The Phaser web UI is live via a Cloudflare quick tunnel:

> **https://result-dover-successful-brother.trycloudflare.com**

Open it and play (pick language on the start screen → click a suspect to
interrogate, click locations to search, then make your one accusation).

- It runs on **Claude** (NPCs = Haiku, judge = Sonnet), using the key in
  `/workspace/.env`. `web.py` reads the repo's own `config.yaml` (which is set to
  the Anthropic provider) — that file is **unchanged**.
- **Generated art AND voice are ON** — Act backdrops + suspect faces are painted
  by SDXL-Turbo on the GPU, and conversation lines are spoken by chatterbox-turbo
  (see "Generated art & voice" below).
- The tunnel URL is **public (no password)** and **ephemeral** — it dies if the
  instance, `web.py`, or `cloudflared` restarts. For a private connection instead,
  use SSH forwarding from your own machine:
  ```bash
  ssh -p 43007 -L 8080:127.0.0.1:17080 root@107.206.71.138
  # then open http://localhost:8080
  ```

How it's wired (3 things, no repo files changed): `pip install flask`; `web.py`
running on `127.0.0.1:17080`; a `cloudflared` quick tunnel pointing at it (all
external ports were already in use, so the authenticated Caddy edge wasn't an
option). To restart later: `cd /workspace/shifting_the_truths && source
/venv/main/bin/activate && set -a && . /workspace/.env && set +a && python web.py`
then re-run the tunnel `cloudflared tunnel --url http://127.0.0.1:17080`.

### Generated art & voice (how it's wired)

Powered by your `Jinyan0924` HF models + the `naomili0924/inference_driven_model_compiler`
project (cloned to `/workspace/inference_driven_model_compiler`).

**Models** (private HF repos — need `hf auth login`; RAM-backed dirs vanish on restart):
| Repo | → path | Used for |
|---|---|---|
| `Jinyan0924/sdxl-turbo-onnx` | `/dev/shm/sdxl-turbo-onnx` | EN backdrops/faces |
| `Jinyan0924/hunyuan-dit-onnx` | `/dev/shm/hunyuan-onnx` | ZH backdrops |
| `Jinyan0924/chatterbox-turbo-onnx` | `/workspace/models/chatterbox-turbo-onnx` | voice (EN) |

**Runtime deps installed into `/venv/main`:**
- Image: `torch==2.11.0+cu128` (GPU), `diffusers`, `transformers`, `optimum`,
  **`optimum-onnx`**, `onnx`, **`sentencepiece`** (ZH/Hunyuan T5 tokenizer), plus
  `onnxruntime-gpu` (already present). CUDA-12 libs are registered via `ldconfig`
  (the image is CUDA 13.2; the wheels are CUDA 12). EN = SDXL-Turbo (512, 1 step,
  ~1s); ZH = Hunyuan-DiT (1024, 20 steps, ~7–19s/image — noticeably slower).
- Voice: `chatterbox-tts` + its helpers (`librosa`, `resemble-perth`, `einops`,
  `conformer`, `omegaconf`, `s3tokenizer`, `torchaudio`, `antlr4-python3-runtime`,
  `pyloudnorm`) — installed mostly with `--no-deps` **on purpose**: a plain
  `pip install chatterbox-tts` would downgrade `diffusers`/`torch` and break the
  image pipeline. The actual TTS inference runs through the idmc ONNX pipeline.

**First paint is slow (~10s)** as the SDXL pipeline loads onto the GPU; cached
after. Backdrops/faces are painted in a background thread while you read the act
intro. Cache: `webui/assets/cache/*.png` and `.../audio/*.wav`.

**Restart the web server** (e.g. after an instance reboot, once models are
re-downloaded): `kill $(cat /tmp/webpy.pid)` then `cd /workspace/shifting_the_truths
&& source /venv/main/bin/activate && set -a && . /workspace/.env && set +a &&
nohup python web.py > /tmp/web.log 2>&1 & echo $! > /tmp/webpy.pid`. To re-pull all
models after a reboot: `cd /workspace/shifting_the_truths && hf auth login && ./restore_models.sh`.

---

# Play in the terminal (local model, no API key)

The game now runs on a **local Phi-3.5-mini ONNX model on the GPU** — no
`ANTHROPIC_API_KEY` needed. **No file in this repo was modified** (its
`config.yaml` is untouched); a separate run-config outside the repo selects the
local model.

## ▶ Play

```bash
cd /workspace/shifting_the_truths
source /venv/main/bin/activate
python main.py --config /workspace/play.config.yaml
```

Add `--mode developer` to also log the hidden truth to `logs/<session>/developer.jsonl`.
To go back to the Anthropic API, just drop `--config` (uses the repo's own
`config.yaml`, which needs `ANTHROPIC_API_KEY`).

## What I did to get here (5 steps)

1. **Installed** the local runtime into `/venv/main`:
   `uv pip install onnxruntime-genai-cuda huggingface_hub hf_transfer`
2. **Downloaded** the model (~2.3 GB) to RAM-backed `/dev/shm`:
   `microsoft/Phi-3.5-mini-instruct-onnx`, folder `gpu/gpu-int4-awq-block-128`
   → `/dev/shm/phi35-onnx/...`
3. **Fixed the CUDA gap** (runtime is CUDA-12, box is CUDA-13.2): installed the
   `nvidia-*-cu12` lib wheels and registered them with `ldconfig`
   (`/etc/ld.so.conf.d/onnxgenai-cuda12.conf`) so no env var is needed.
4. **Enabled GPU** in the model's `genai_config.json`
   (`provider_options: [{"cuda": {}}]`; original saved as `genai_config.json.cpu.bak`).
5. **Wrote** `/workspace/play.config.yaml` (outside the repo) pointing both the
   `npc` and `judge` agents at the local model.

Verified with a full end-to-end game on the GPU (judge picks the culprit, NPCs
converse, accusation scored, epilogue written) — exit 0, no errors.

## ⚠️ After an instance restart

`/dev/shm` is RAM-backed and is **wiped on restart**. To replay, re-run step 2:

```bash
source /venv/main/bin/activate
python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/Phi-3.5-mini-instruct-onnx', allow_patterns=['gpu/gpu-int4-awq-block-128/*'], local_dir='/dev/shm/phi35-onnx')"
```

then re-do step 4 (set `provider_options` in the fresh `genai_config.json`). The
pip installs (steps 1, 3) and `/workspace/play.config.yaml` survive a restart;
only the `/dev/shm` model and its config edit need redoing.
