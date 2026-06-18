# Shifting Truth — Code Reading Guide

> The order to read the source so each file only references things you've already
> seen (bottom-up by dependency). Each step is paired with the relevant sections
> of `GAME_MECHANICS.md` and `INFO_FLOW.md` so you can cross-validate as you go.
>
> **The one mental model to hold throughout:** *the engine slices the immutable
> ground truth into narrow, per-job prompts; the judge is stateless and the NPCs
> are siloed.* Keep asking, at every function, "what knowledge does this put in
> front of the LLM, and what does it deliberately withhold?"

---

## Pass 1 — the data model (read the *shape* before the logic)

You can't follow `engine.py` without knowing what a "case" looks like, so start
with the authored data.

**1. `README.md`** — orientation only.

**2. `config.yaml`** *(read fully — short and commented)*
Look for: the `agents.npc` / `agents.judge` split (two independent brains),
`costs` (the time economy), `game.mode`, `game.lang` / `cases`. Skip the
`images:` / `audio:` blocks for now.
→ pairs with **GAME_MECHANICS §1**.

**3. `case.yaml`** *(the most important file to understand — skim structure, don't memorize)*
Trace these keys and ask "who consumes this?":
- `characters[*]` → `role`, `public_story`, `private_story`,
  `flaws[*]` (`motive_seed`, `trigger_vs_victim`, `behavioral_tells`,
  `lie_tendency`), `knows_about_others`
- `plausibility_matrix` → valid `(culprit, flaws, strength)` combos
- `method_options` → `requires_access`
- `special_rolls` → `true_accident` weight
- `acts[*].phases` and `acts[*].spots[*].items[*].condition`
→ pairs with **GAME_MECHANICS §2 & §5.1** and **INFO_FLOW §2**.

**4. `i18n.py`** — **skim, do NOT read top-to-bottom.** It's a ~780-line string
table. Just learn that `t(lang)` returns a dict every other file pulls strings
from. When `engine.py` references `L["npc"]["killer_rules"]` etc., come *back*
here to read that specific string.

---

## Pass 2 — the foundations (small, no game logic)

**5. `providers.py`** (163 lines) — the `LLMProvider.chat()` interface and its
three backends (`anthropic` / `onnx` / `mock`). **Key realization:** every call is
stateless — there is no shared judge memory. Read `MockProvider` to know what
"no network" play looks like.
→ pairs with **INFO_FLOW §0** ("the judge has no memory").

**6. `gamelog.py`** (73 lines) — `conv()` vs `dev_log()`, production vs developer.
Quick read.
→ pairs with **GAME_MECHANICS §10**.

---

## Pass 3 — the core (read in *call order*, not top-to-bottom)

**7. `engine.py`** — follow the order the game actually invokes it:
1. `load_case` (`:27`) → `GroundTruth` dataclass (`:53`) → `culprit_options` (`:67`)
2. `Director.roll` (`:92`) → `_roll_murder` (`:129`) → `_roll_true_accident` (`:171`) → `_distribute_clues` (`:189`)
3. `judge_select_culprit` (`:231`) — note the menu it builds
4. `build_npc_system_prompt` (`:284`) — **the heart of the knowledge model**; read the culprit-vs-innocent branch carefully
5. `referee_check` (`:351`)
6. `item_present` (`:491`)
7. `judge_generate_deeds` (`:529`) and `deal_boundary_gossip` (`:567`)
8. `judge_accusation` (`:430`) → `_who_verdict`, `_grade_with_llm`, the scoring block
→ pairs with **GAME_MECHANICS §2–§9** and **INFO_FLOW §3–§9**, in the same order.

**8. `main.py`** — the orchestrator that ties it together. Read it **outside-in**:
1. `main()` (`:419`) → `Game.__init__` (`:77`) — the 10-step setup
2. `play()` (`:391`) — the act/phase loop (the spine)
3. then drill into `run_search` (`:223`), `run_talk` (`:284`), `npc_reply` (`:169`),
   `system_for` (`:162`), `boundary` (`:360`), `conclusion` (`:371`)
→ pairs with **GAME_MECHANICS §4–§8** and **INFO_FLOW §7**.

> **Stop here.** After files 1–8 you understand the entire game. Everything below
> is the optional, fail-soft multimedia layer — read only when you want to touch
> the web UI or art/voice.

---

## Pass 4 — optional (web + multimedia)

**9. `web.py`** — Flask wrapper; reuses the same engine, one session per browser.
**10. `rooms.py`** — judge-authored room manifest ("manifest first, paint second").
**11. `imagegen.py`** / **12. `ttsgen.py`** — cached, optional ONNX art/voice (mirror each other).
**13. `webui/game.html`** (Phaser UI) / `webui/index.html` (classic UI at `/classic`).

---

## Two tips for the read

- **Keep a developer log open beside the code.** Run
  `python main.py --seed 7 --mode developer` (mock providers are fine — set both
  agents to `provider: mock` in `config.yaml` if you have no API key), then read
  `logs/<session>/developer.jsonl`. You'll see `ground_truth`,
  `culprit_selection`, `deeds`, `boundary_gossip`, and `npc_memory` snapshots —
  these make the abstract flow in `engine.py` concrete.

- **Always ask the withholding question.** For every function that builds a
  prompt, note what ground truth it includes *and what it deliberately leaves
  out*. The three boundaries worth confirming yourself:
  1. the culprit-selection judge is blind to the method (it's rolled after the pick),
  2. the gossip judge is blind to the culprit (their deeds are masked to `(secret)`),
  3. innocents are *told* they don't know the killer (not merely missing it).

---

## The companion docs

- **`GAME_MECHANICS.md`** — step-by-step *what happens* (timers, scoring, item
  visibility, the act/phase loop), with `file:line` references.
- **`INFO_FLOW.md`** — *who knows what, when*: each agent's duties, the knowledge
  in their hands at each stage, and the channels between judge, NPCs, and player.
