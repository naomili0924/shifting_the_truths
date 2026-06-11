# Shifting Truth — three-act playable build

A murder mystery where suspects are LLM agents, the plot re-rolls
every game, and the night unfolds in three designer-authored acts
with timed search and interrogation phases.

## Quick start

```bash
pip install pyyaml
export ANTHROPIC_API_KEY=sk-ant-...
python main.py                    # uses config.yaml (asks language if unset)
python main.py --seed 7           # reproduce a specific plot
python main.py --lang zh          # 用中文游玩
```

No key / no network? Test the plumbing with mock agents: set
`provider: mock` for both agents in config.yaml.

## Play in the browser (web UI)

A point-and-click version with suspect avatars, a chat box, clickable
search locations, an evidence pouch, and the final accusation screen.

```bash
pip install flask                 # one-time (CLI doesn't need it)
python web.py                     # serves http://127.0.0.1:17080
```

Open it locally, or reach a remote box via SSH forwarding
(`ssh -L 8080:127.0.0.1:17080 ...`) or a Cloudflare quick tunnel
(`cloudflared tunnel --url http://127.0.0.1:17080`). Pick the language
on the start screen; click a **suspect** to interrogate them, type in
the **chat box**, click a **location** to search it, then make your one
accusation. It runs without an API key (suspects give canned replies via
the mock provider) and uses real models the moment a key/local model is
configured — `web.py` falls back to mock if a provider can't initialize.

## Painted UI (Phaser + generated art)

The browser UI at `/` is a **Phaser** point-and-click adventure with art
generated on the fly by a local **SDXL-Turbo ONNX** model (exported by the
sibling `inference_driven_model_compiler` project). The original text/click UI
is still available at `/classic`.

**Manifest first, paint second.** The room's searchable structure — which spots
exist, which items hide where — is the manifest (authored in `case.yaml`,
resolved against the rolled plot, and the judge writes each room's visual layout
per run, verified). *Only then* is the image model called with a prompt built
from that manifest. Game logic never depends on what the picture contains, so:

- **Backdrops** are painted per room from the manifest's prompt.
- **Hotspots are chips, not pixels** — each searchable spot is a labeled button
  nudged toward roughly the right spot; if the model painted the desk elsewhere,
  nothing breaks.
- **Items** found in a search are painted as object cards.
- **Five fixed faces** (the suspects) are painted once and cached, reused every
  run — the cast is constant, only the rooms re-roll.

Generation is **hidden behind the scenario intro** (a background thread paints
while you read), **cached** on disk (so the marginal cost trends to zero), and
**entirely optional**: if the model or a GPU is unavailable, generation is
skipped and the UI falls back to plain backdrops — exactly like the mock-LLM
fallback. Art needs `onnxruntime-gpu`; the core game (CLI and classic UI) needs
none of that.

```bash
pip install flask pyyaml
python web.py                 # http://127.0.0.1:17080  (Phaser UI with art)
```

### One model per language

`config.yaml`'s `images.by_lang` maps each language to its own exported ONNX
model — each is loaded lazily, so an English game never loads the Chinese model:

| Language | Model | Location |
|---|---|---|
| English | SDXL-Turbo (512, English prompts) | `/workspace/models/sdxl-turbo-onnx` |
| 中文 | Hunyuan-DiT (1024, Chinese prompts) | `/dev/shm/hunyuan-onnx` |

Both are produced by the sibling `inference_driven_model_compiler` inference-driven
export. They're large (SDXL ~7 GB, Hunyuan ~22 GB), so they aren't in git; restore
them from their (private) Hugging Face repos instead of re-exporting:

```bash
huggingface-cli login          # token with read access to the repos
./restore_models.sh            # downloads both to the paths above
```

The Hunyuan dir is under `/dev/shm` (RAM-backed) — re-run `restore_models.sh`
after an instance restart. If a model is missing, that language just falls back
to plain backdrops.

## Language / 语言

The player picks the language; English stays the default and is never
replaced. Selection order: `--lang en|zh` → `game.lang` in config.yaml
→ an interactive prompt at startup if neither is set.

Each language ships its own authored case (`game.cases` in config:
`case.yaml` for English, `case_zh.yaml` for Chinese). All UI, command
words, LLM prompts, fallback text and logs follow the chosen language;
`i18n.py` holds every string for both. In Chinese you type Chinese
commands too (`查看`、`搜查 <地点>`、`询问 <人名>`、`出示`、`返回`、`下一步`),
and the suspects reply in Chinese. To add a language, add a table to
`i18n.py` and a `case_<lang>.yaml`; no game logic changes.

## Per-agent brains (config.yaml)

Every agent picks its own backend — mix freely:

```yaml
agents:
  npc:                      # plays the five suspects
    provider: anthropic     # anthropic | onnx | mock
    model: claude-haiku-4-5-20251001
  judge:                    # writes deeds & gossip, grades the solve
    provider: anthropic
    model: claude-sonnet-4-6
```

Local inference later: `pip install onnxruntime-genai`, download an
ONNX chat model (e.g. Phi-3.5-mini-instruct-onnx), then:

```yaml
  npc:
    provider: onnx
    onnx_dir: ./models/phi-3.5-mini-instruct-onnx
```

Game logic never changes — only this file does.

## The night (phase order defined per act in case.yaml)

| Act | Scene (yours to author)      | Phases                          |
|-----|------------------------------|---------------------------------|
| 1   | Terrace & courtyard          | search 8 min → talk 12 min      |
| 2   | Study & Diana's room         | search 8 min → talk 12 min      |
| 3   | Cellar: tool room & archive  | search 6 → talk 10 → search 6   |

Then the police arrive and you MUST accuse: who, why, how — one
shot, judged by the same omniscient LLM that wrote the crime.

Between acts the suspects compare notes: the judge deals each NPC
partial hearsay about others' movements — and about what *you've*
been asking. Careless questions travel.

## Commands

Search phase: `look`, `search <spot>` (costs time), `evidence`, `next`
Talk phase: `cast`, `talk <name>`, `notes`, `evidence`, `next`
In conversation: type any question, `show <item>` to confront with
evidence, `back` to walk away.

## Items are plot-aware

You author every item in case.yaml; a `condition` field ties some to
the rolled plot. Searching the railing finds fresh tool marks only
when the killer loosened it; the pill organizer is off only on the
medication-swap roll; the rotted core appears only on the true-
accident roll. Always-present items (Marco's ledger, Sofia's
termination letter, the buried inspection report) are red herrings
or context every playthrough.

## How the LLM agents divide the work

| Job                                   | Agent | Fallback if LLM fails |
|---------------------------------------|-------|------------------------|
| Choosing the culprit & their motive   | judge | deterministic weighted pick |
| Suspect dialogue (lies, tells, memory)| npc   | —                       |
| True deeds for every NPC at game start| judge | deterministic templates |
| Boundary gossip between acts          | judge | deterministic templates |
| Grading WHY and HOW of the accusation | judge | keyword matching        |
| Method/timeline/clues, WHO check, scoring, timers, items | code | (never an LLM job) |

The judge picks the killer and which flaws drive them from the authored
valid combos in the plausibility matrix, so the choice always stays
coherent with method access and clue trails. Every judge output is
validated (JSON shape, on-menu pick, no culprit leaks to innocents)
before use; invalid output falls back, so a flaky model can never break
a running game.

Because an LLM now makes this call, `--seed` alone no longer reproduces
the culprit — it still seeds method, clue and accident rolls. To see a
specific game's hidden truth, run in developer mode (below).

## Launch modes (logging)

Two modes, set by `game.mode` in config.yaml or `--mode` on the CLI.
Both write JSON Lines to `<log_dir>/<session-timestamp-pid>/`; nothing
is ever printed to the player's screen.

```bash
python main.py                      # production (config default)
python main.py --mode developer     # full spoiler layer for debugging
```

| File | production | developer | Contents |
|------|:---------:|:---------:|----------|
| `conversation.jsonl` | ✅ | ✅ | every question & reply, evidence shown, searches + finds, the accusation and verdict |
| `developer.jsonl`    | — | ✅ | the judge's culprit/motive **choice** (+ reasoning, source), the resolved **ground truth**, the generated **deeds**, **boundary gossip**, and a snapshot of **every NPC's memory at each stage** (full system prompt, accumulated hearsay, statements so far) |

Developer mode is how you "track an NPC's memory in each stage" and see
exactly what the judge decided. Production mode keeps a clean transcript
of play with no spoilers. The `logs/` directory is gitignored.

## Files

- `config.yaml` — agent brains, time costs, launch mode, language
- `case.yaml` — the authored case (English): cast, flaws, methods, acts, items
- `case_zh.yaml` — the authored case (Chinese)
- `i18n.py` — every player- and LLM-facing string, per language
- `engine.py` — director roll, judge jobs, prompts, referee, verdict
- `providers.py` — LLMProvider interface: Anthropic / ONNX / mock
- `gamelog.py` — session logging (production / developer modes)
- `main.py` — the act/phase game loop (CLI)
- `web.py` — Flask backend for the browser UI (one session per browser)
- `rooms.py` — visual room manifest: judge-authored layout + image prompts (verified, deterministic fallback)
- `imagegen.py` — optional cached SDXL-Turbo ONNX backdrop/portrait/item painter (graceful no-art fallback)
- `webui/game.html` — the Phaser point-and-click client (painted backdrops, chip hotspots, faces)
- `webui/index.html` — the original single-page web client (served at `/classic`)
