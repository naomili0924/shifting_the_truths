# Shifting Truth — Game Mechanics Specification

> A step-by-step description of how the game actually behaves, derived from the
> code. Every step cites the file and line(s) that implement it so you can read
> the source side-by-side and confirm the mechanics are correct.
>
> Scope: the **CLI game** (`main.py`) and the **engine** (`engine.py`) it drives.
> The web UI (`web.py`) reuses the same engine; differences are noted where they
> matter, but this document follows the canonical CLI flow.

---

## 0. Cast of actors (who decides what)

| Actor | Code | Responsibility |
|---|---|---|
| **Director** | `engine.py:80` `Director` | Rolls one playthrough: accident-or-murder, method, motive, secret timeline, clue distribution. Deterministic given a seed. |
| **Judge LLM** | `agents.judge` provider | Picks the culprit+flaws, writes deeds, deals gossip, grades the final accusation, writes the epilogue. Every output validated; falls back to code on failure. |
| **NPC LLM** | `agents.npc` provider | Plays all five suspects in conversation. |
| **Referee** | `engine.py:351` `referee_check` | Cheap regex guard on every NPC reply before the player sees it. |
| **Code (the game itself)** | `main.py`, parts of `engine.py` | Timers, item visibility, WHO verdict, scoring, logging — never an LLM job. |

The division of labor is summarized in the README "How the LLM agents divide the
work" table; this doc traces the mechanics underneath it.

---

## 1. Startup & configuration resolution

Entry point `main()` (`main.py:419`) parses `--config` (default `config.yaml`),
`--seed`, `--mode`, `--lang`, then constructs `Game(...)` and calls `.play()`.

`Game.__init__` (`main.py:77-133`) resolves everything, **in this order**:

1. **Load config** YAML (`main.py:81-83`).
2. **Build providers** for `npc` and `judge` from their config blocks
   (`main.py:85-86` → `provider_from_config`, `providers.py:152`).
   - `provider` is one of `anthropic | onnx | mock`; **defaults to `mock`** if
     absent (`providers.py:154`).
   - `anthropic` requires `ANTHROPIC_API_KEY` in the environment or it raises at
     construction (`providers.py:39-43`).
3. **Resolve language** (`main.py:89` → `resolve_lang`, `main.py:60-66`):
   precedence is `--lang` > `config game.lang` > **interactive startup prompt**.
   Normalized by `normalize_lang` (in `i18n.py`).
4. **Resolve launch mode** (`main.py:95`): `--mode` > `config game.mode` >
   `"production"`. `developer` adds the spoiler log layer.
5. **Load the case file** for the chosen language (`main.py:98` →
   `resolve_case_path`, `main.py:69-74`): `game.cases[lang]` > `game.case_<lang>`
   > `game.case` > `"case.yaml"`. `load_case` (`engine.py:27`) requires the keys
   `scenario, victim, characters, timeline_skeleton, plausibility_matrix,
   method_options` or raises.
6. **Resolve seed** (`main.py:99-100`): `--seed` > `config game.seed` > `None`.
7. **Roll the playthrough** (`main.py:104`, see §2).
8. **Generate deeds** for every NPC (`main.py:113`, see §3).
9. **Build each NPC's base system prompt** (`main.py:116-121`, see §3).
10. Initialize the mutable game state (`main.py:122-131`, see §3).

> **Cross-validation note:** the order matters. The culprit is chosen *during*
> the roll in step 7, before deeds/prompts in steps 8-9, so deeds and prompts can
> reference who the killer is.

---

## 2. The Director's roll — building ground truth

`self.gt = self.director.roll(selector=self._select_culprit)` (`main.py:104`).
The roll produces a `GroundTruth` dataclass (`engine.py:53-61`):

```
is_murder, culprit, active_flaws, method, motive, secret_timeline, distributed_clues
```

### 2.1 Accident special-roll first

`Director.roll` (`engine.py:92-107`):

1. Iterate `case["special_rolls"]`. For `true_accident`, draw `rng.random()`; if
   it is `< weight` (case authored at `0.1`, i.e. **10%**), the night is a *true
   accident* → `_roll_true_accident` (`engine.py:171-182`) and the selector is
   **never consulted**. (`engine.py:104-105`)
2. Otherwise it is a murder; call the `selector()` to get `(culprit, flaws)` and
   run `_roll_murder` (`engine.py:106-107`).

> So the culprit choice (LLM or code) only happens on the murder branch — ~90% of
> games.

### 2.2 Culprit selection (the judge LLM's job)

The selector is `Game._select_culprit` (`main.py:136-141`), which calls
`judge_select_culprit` (`engine.py:231-278`):

1. Build the menu of **valid (culprit, flaws) combos** from the plausibility
   matrix via `culprit_options` (`engine.py:67-77`). Only `strong` (weight 3) and
   `medium` (weight 2) rolls are eligible; `weak` (weight 0) combos are **excluded
   entirely** (`engine.py:73-76`).
2. Present each combo with its authored `motive_seed`s to the judge LLM as a
   numbered menu (`engine.py:246-251`).
3. Call `judge.chat(...)` and `_extract_json` the reply (`engine.py:255-257`).
4. **Validate**: the picked name must resolve via `_match_name` (`engine.py:218`)
   and the `(culprit, flaws)` set must exactly match an on-menu combo
   (`engine.py:259-263`). If anything fails → `except` branch.
5. On success: `source="llm"` + rationale. On any failure: deterministic weighted
   pick (`rng.choices` weighted by combo weight), `source="fallback"`
   (`engine.py:267-271`).
6. Attach the resolved `motive_seeds` and `triggers` for the chosen flaws
   (`engine.py:273-277`).

The selection (including `source` and `rationale`) is dev-logged
(`main.py:139-140`).

> **Cross-validation note:** because an LLM makes this call, `--seed` alone does
> **not** reproduce the culprit (README "Items are plot-aware" / "Launch modes").
> The seed still fixes method, clues, accident roll, and the fallback pick.

### 2.3 Murder roll details

`_roll_murder` (`engine.py:129-169`):

1. **Culprit/flaws**: from the selector; if `None`, fall back to
   `_weighted_culprit` (`engine.py:122-127`).
2. **Method** (`engine.py:138-146`): filter `method_options` to those the culprit
   can access (`requires_access == ["any"]` or contains the culprit), then weighted
   `rng.choices`. The universal method `push_in_the_dark` is reachable by everyone.
3. **Motive** (`engine.py:148-159`): woven from the chosen flaws' `motive_seed`s
   and `trigger_vs_victim`s using localized connective strings
   (`motive_head/and/mid/also/tail`). Pure string assembly — no LLM.
4. **Secret timeline** (`engine.py:162`, `_secret_timeline` `engine.py:184-187`):
   localized template list keyed by `method_id` (or `default`), formatted with
   culprit + victim names.
5. **Clue distribution** (`engine.py:168`, see §2.5).

### 2.4 Accident roll details

`_roll_true_accident` (`engine.py:171-182`): `is_murder=False`, `culprit=None`,
`active_flaws=["guilty_concealment"]`, no method. The motive/timeline come from
localized accident templates, formatted with victim `v` and the **concealer** `e`
(the character carrying the `guilty_concealment` flaw — `concealer_name`,
`engine.py:42-47`).

### 2.5 Clue distribution (the fairness rule)

`_distribute_clues` (`engine.py:189-207`):

- Pick the eligible witnesses: all characters except the culprit (and, on the
  accident roll, also except the concealer) (`engine.py:195-202`).
- `rng.sample` up to **2** witnesses (`engine.py:203`).
- Give each one a method-specific (or accident-specific) sighting clue, formatted
  with culprit/victim/concealer names (`engine.py:204-207`).

> **Fairness invariant (engine.py:191-193 comment):** every playthrough plants at
> least two independent threads pointing at the truth. Confirm `method_sightings`
> in `i18n.py` has an entry for every method id, or `_distribute_clues` will
> `KeyError` (`engine.py:198`).

---

## 3. Pre-game setup (after the roll)

1. **Names list** `self.names` (`main.py:112`) — character order from the case.
2. **Deeds** `self.deeds` (`main.py:113`) via `judge_generate_deeds`
   (`engine.py:529-560`):
   - Judge LLM writes 1-4 true, mundane deed lines per NPC.
   - **Validation** (`engine.py:542-557`): keys must be exactly the cast; each
     innocent gets 1-4 lines; no confession phrases; the culprit's name must not
     appear next to a kill verb. The culprit's own deeds are blanked (their truth
     lives in the secret block).
   - On any failure → `default_deeds` (`engine.py:507-526`), deterministic
     templates. Dev-logged (`main.py:115`).
3. **NPC base system prompts** `self.base_prompt` (`main.py:116-121`) via
   `build_npc_system_prompt` (`engine.py:284-345`). Each prompt contains: intro,
   setting, the public "accident" story, the character's role/public/private
   story, the fixed timeline skeleton, what they know about others **plus any
   distributed clue aimed at them** (`engine.py:312-313`), their flaws (tells +
   triggers + lie tendencies), their deeds, and then either:
   - **culprit block** (`engine.py:328-337`): the real motive, the secret
     timeline, and the killer rules; **or**
   - **innocent block** (`engine.py:338-342`): innocent instructions.
   - Plus shared style rules.
4. **Mutable state** (`main.py:122-131`):
   - `self.extras[name]` — accumulated hearsay (gossip), appended between acts.
   - `self.histories[name]` — full chat message list per NPC.
   - `self.questions_log[name]` — raw player questions per NPC (feeds gossip).
   - `self.evidence` — list of found items `{name, found_text}`.
   - `self.searched` — set of spot ids already searched.
   - `self._alias` — reverse map of every localized command token → canonical verb
     (`main.py:129-131`), so EN and ZH command words both resolve.

> **Cross-validation note — live system prompt:** `system_for(name)`
> (`main.py:162-167`) returns `base_prompt[name]` **plus** any accumulated
> `extras` (hearsay) appended under a "new memories" header. So the prompt grows
> across acts; it is recomputed on every NPC turn.

---

## 4. The night — act/phase loop

`Game.play()` (`main.py:391-416`):

1. Print title, setting, and the public accident text (`main.py:392-397`).
2. Snapshot NPC memory at `game_start` (dev only) (`main.py:400`).
3. For each act in `case["acts"]` (`main.py:401-415`):
   - Set `self.cur_act` for log context (`main.py:402`).
   - Print act header + `scene_intro` (`main.py:404-406`).
   - For each phase in `act["phases"]` (`main.py:407-412`): dispatch by
     `phase["type"]` → `run_search` (search) else `run_talk`. The phase's `time`
     is the minute budget.
   - Snapshot NPC memory at `act<N>_end` (dev only) (`main.py:413`).
   - If not the last act, run the boundary gossip (`main.py:414-415`, see §7).
4. After all acts → `conclusion()` (`main.py:416`, see §8).

The authored act/phase shape (from `case.yaml`):

| Act | Phases (type, minutes) | Spots |
|---|---|---|
| 1 | search 8 → talk 12 | railing, stairs, body, table |
| 2 | search 8 → talk 12 | desk, bedroom, fireplace |
| 3 | search 6 → talk 10 → search 6 | tool_room, archive |

> Act 3 has the search→talk→search shape, so phase order is data-driven, not
> hardcoded. Confirm by reading `acts[*].phases` in `case.yaml`.

---

## 5. SEARCH phase

`run_search(act, minutes)` (`main.py:223-281`). `minutes` is the phase budget;
**search cost** is `costs.search` (config default **2**) (`main.py:225`).

Loop until `minutes <= 0` or the player types `next`:

- Parse `verb` + `rest` from the line; canonicalize the verb (`main.py:230-231`).
- `next` → return early (`main.py:233`).
- `look` → list every spot with a "(searched)" tag if already done
  (`main.py:235-239`).
- `evidence` → show the pouch (`main.py:240-241`).
- `help` → print search help (`main.py:242-243`).
- `search <spot>` (`main.py:246-279`):
  1. Match `rest` against spot `id` or `name` substring (`main.py:248-252`). No
     match / no target → "search where?" (no time lost) (`main.py:253-255`).
  2. **If `minutes < cost`** → "no time" and the phase ends (`main.py:256-258`).
  3. **Deduct cost** (`main.py:259`). *(Time is charged even if the spot was
     already searched — see next step.)*
  4. If the spot id is already in `self.searched` → "already searched", no items
     (`main.py:260-262`).
  5. Mark searched; compute visible items via `item_present(item, gt)`
     (`main.py:263-265`, see §5.1).
  6. If none → print the spot's `empty_text` (`main.py:266-267`). Else print each
     found item and **append it to `self.evidence`** (`main.py:269-276`).
  7. Log a `search` conversation event with the finds (`main.py:277-278`).
- Anything else → "unknown command" (`main.py:280`).

> **Cross-validation note — double-search cost:** because step 3 deducts before
> step 4 checks `already searched`, re-searching a spot **still costs time** but
> yields nothing. Verify you intend that (it's a mild anti-stalling rule).

### 5.1 Item visibility is plot-aware

`item_present(item, gt)` (`engine.py:491-499`):

- No `condition` → always present.
- `condition.method` → present only if `gt.is_murder` and the rolled method id
  matches.
- `condition.accident` → present only when `is_murder` matches the negation
  (i.e. accident-roll items appear only on the accident roll).
- `condition.culprit` → present only if that named culprit was chosen.

> This is the "Items are plot-aware" mechanic: e.g. fresh tool marks on the
> railing appear only on the loosened-railing roll; always-present items are red
> herrings/context. Cross-check each item's `condition` in `case.yaml`.

---

## 6. TALK phase

`run_talk(minutes)` (`main.py:284-357`). `minutes` is the phase budget;
`question` cost (default **1**) and `show` cost (default **1**) from `costs`
(`main.py:285-286`).

Outer loop (choosing what to do):

- `next` → return (`main.py:293-294`).
- `cast` → list suspects (`main.py:296-297`).
- `evidence` → show pouch (`main.py:299-300`).
- `notes` → dump every suspect's prior statements (`main.py:302-303`,
  `show_notes` `main.py:209-220`).
- `help` → talk help (`main.py:305-306`).
- `talk <name>` (`main.py:308-355`): resolve the name (`resolve_name`,
  `main.py:185-192`; accepts a number, or a substring of a name). Enter the
  **conversation sub-loop**.

### 6.1 Conversation sub-loop (`main.py:314-354`)

While `minutes > 0`, prompt `You → <name>`:

- `back` (no argument) → leave the conversation (`main.py:320-321`).
- `show <item>` (`main.py:322-342`):
  1. Find an evidence item whose name contains the token (`main.py:323-326`). No
     match → "no item", no time lost (`main.py:327-328`).
  2. **Deduct show cost** (`main.py:330`).
  3. Build the "I present X" message and get the NPC reply via `npc_reply`
     (`main.py:331-335`). Provider errors are caught and shown
     (`main.py:336-338`).
  4. Log a `show_evidence` event (`main.py:339-341`).
- **Any other text = a question** (`main.py:343-352`):
  1. **Deduct question cost** (`main.py:343`).
  2. Append to `questions_log[name]` (`main.py:344`) — this is what later
     **travels as gossip**.
  3. `npc_reply` and print; errors caught (`main.py:345-350`).
  4. Log a `question` event with `minutes_left` (`main.py:351-352`).
- If `minutes <= 0` mid-conversation, break out of both loops (`main.py:353-354`).

> **Cross-validation note — `show` vs `talk` token collision:** in the
> conversation sub-loop, only `back` and `show <item>` are special; **everything
> else is a question**, including words like `cast`/`notes`. Those outer commands
> are not available once you're talking to someone — confirm that's intended.

### 6.2 How an NPC reply is produced — `npc_reply` (`main.py:169-183`)

1. Append the player's message as a `user` turn to that NPC's history
   (`main.py:170`).
2. Call `npc_llm.chat(system_for(name), histories[name])` — note `system_for`
   includes accumulated hearsay (`main.py:171-172`).
3. **Referee** the reply (`referee_check`, `engine.py:351-361`):
   - If it matches `leak_rx` (mentions "system prompt", "language model",
     "instructions say", "as an ai") → leak hint.
   - If the NPC is **not** the culprit but the reply matches `confession_rx`
     (`I killed/murdered/pushed`, `it was me`) → confession hint.
4. If a hint is returned, append the bad reply + an `[OUT OF CHARACTER
   CORRECTION]` user turn and **regenerate once** (`main.py:174-181`).
5. Append the final reply as an `assistant` turn and return it (`main.py:182`).

> **Cross-validation note:** the referee runs **once** (single regeneration); it
> is a cheap regex guard, not a semantic check. A culprit *is* allowed to confess
> (only innocents are corrected) — confirm `gt.culprit != npc_name` at
> `engine.py:358`.

---

## 7. Between acts — boundary gossip

`boundary(act_no)` (`main.py:360-368`) runs after every act except the last
(`main.py:414`). It calls `deal_boundary_gossip` (`engine.py:567-614`):

1. Summarize **what the player asked** each NPC (last 3 questions, truncated)
   (`engine.py:576-578`) and each NPC's deeds (`engine.py:579-581`).
2. Judge LLM writes ≤2 hearsay lines per NPC (`engine.py:589-592`).
3. **Validation** (`engine.py:594-601`): keep ≤2 lines per known name; drop any
   line where the culprit's name sits next to a kill verb (`bad` regex,
   `engine.py:584-588`); require at least one surviving line.
4. On failure → deterministic fallback (`engine.py:603-613`): each NPC gets a
   generic "I saw <other>" slip, plus, if the player pressed someone, a "they kept
   asking <target> about <q>" line.
5. Back in `boundary`, each line is prefixed with the localized "hearsay" marker
   and appended to `extras[name]` (`main.py:365-367`), so it enters that NPC's
   live system prompt next act (§3 note). Dev-logged (`main.py:368`).

> This is the "careless questions travel" mechanic: the gossip explicitly folds in
> a summary of *your* questioning, so pressing one suspect about another can leak
> to the cast. Cross-check `gossip_user` / `gossip_asked` strings in `i18n.py`.

---

## 8. The forced accusation (endgame)

`conclusion()` (`main.py:371-388`):

1. Print divider + intro + the cast (`main.py:372-374`).
2. Prompt **WHO** (loops until non-empty), then **WHY** (free text), then **HOW**
   (free text) (`main.py:375-379`).
3. Log the `accusation` (`main.py:380`).
4. Grade via `judge_accusation` (`main.py:382-383`, see §9).
5. Print the verdict, log it, snapshot final NPC memory (`main.py:384-388`).

> One shot — there is no retry path in the code. Confirm: `conclusion` is called
> once at the end of `play()` (`main.py:416`).

---

## 9. Scoring — `judge_accusation` (`engine.py:430-485`)

### 9.1 WHO verdict (code, not LLM)

`_who_verdict(gt, accused, lang)` (`engine.py:381-388`):

- **Murder**: true if the culprit's full name, a ≥2-char substring of it, or any
  >2-char name-part appears in the accusation text (bidirectional matching)
  (`engine.py:384-387`).
- **Accident**: true if the text contains an accident keyword (`accident`, `no
  one`, `nobody`, `noone`, `railing`, `rot`) (`engine.py:388`).

### 9.2 WHY/HOW grade (judge LLM, with keyword fallback)

`_grade_with_llm(gt, reasoning, how, provider, lang)` (`engine.py:402-427`):

1. Assemble the hidden truth (culprit, motive, flaws, method, secret timeline)
   (`engine.py:406-413`).
2. Ask the judge to grade WHY (`motive`) and HOW (`method`) (`engine.py:418-421`).
3. Accept only if JSON parses and `motive ∈ {correct, partial, wrong}`
   (`engine.py:422-424`); default `method` to `not_stated`.
4. On failure → `_keyword_motive_fallback` (`engine.py:391-399`): count how many
   flaw-id word-stems appear in the player's reasoning; ≥2 → correct, 1 → partial,
   0 → wrong; method forced to `not_stated`.

### 9.3 Score assembly (`engine.py:435-447`)

```
score  = 50 if WHO correct else 0
score += {correct: 35, partial: 18}.get(motive, 0)     # WHY
score += {correct: 15, partial:  8}.get(method, 0)     # HOW
if not WHO_ok: score = min(score, 49)                  # WHO gates the top tiers
```

> **Cross-validation note — the WHO gate (engine.py:444-446):** even a perfect
> why/how cannot lift a wrong arrest above 49, so it can never read as "the right
> arrest." Max possible is `50+35+15 = 100`.

### 9.4 Rating, verdict card, epilogue

- **Rating** is the first label whose cutoff `score` meets (`engine.py:446`),
  from `EN["ratings"]`:

  | score ≥ | rating |
  |---|---|
  | 90 | FLAWLESS — the who, the why, the how |
  | 70 | CASE CLOSED |
  | 50 | THE RIGHT ARREST, THE WRONG STORY |
  | 25 | SO CLOSE — right room, wrong face |
  | 0 | MISCARRIAGE — an innocent in handcuffs |

- **Verdict card** (`engine.py:454-463`): rating, score, WHO line (accused vs the
  real truth label), WHY label+comment, HOW label.
- **Consequence** branch (`engine.py:465-471`): correct arrest / wrong-murder /
  accident-misread.
- **Epilogue** (`engine.py:473-484`): the judge LLM (as narrator) writes a closing
  passage from the real motive + secret timeline + consequence; on error a
  templated error line is used. Final output = card + epilogue (`engine.py:485`).

---

## 10. Logging (`gamelog.py`)

- Mode normalized by `normalize_mode` (`gamelog.py:31-32`): anything starting
  "dev" → developer, else production.
- Logs go to `<log_dir>/<YYYYmmdd-HHMMSS-pid>/` (`gamelog.py:42-52`).
- **`conversation.jsonl`** (both modes) via `conv(...)` — session start, searches,
  questions, evidence shown, accusation, verdict.
- **`developer.jsonl`** (developer only) via `dev_log(...)` — ground truth
  (`main.py:105-110`), culprit selection (`main.py:140`), deeds (`main.py:115`),
  boundary gossip (`main.py:368`), and full NPC memory snapshots at each stage
  (`log_npc_memory`, `main.py:143-154`).
- Nothing is ever printed to the player's screen; both writers swallow IO errors
  (`gamelog.py:60-64`).

---

## 11. Fail-soft summary (what happens when an LLM misbehaves)

Every judge job is validated and has a deterministic fallback, so a flaky model
can never break a running game:

| Job | Validated by | Fallback |
|---|---|---|
| Culprit/flaws pick | on-menu match (`engine.py:259-263`) | weighted code pick (`engine.py:267-271`) |
| Deeds | shape + no-confession checks (`engine.py:542-557`) | `default_deeds` templates (`engine.py:560`) |
| Boundary gossip | ≤2 lines, no kill-verb leak (`engine.py:594-601`) | generic slip templates (`engine.py:603-613`) |
| WHY/HOW grade | JSON + enum (`engine.py:422-424`) | keyword match (`engine.py:427`) |
| NPC reply | regex referee (`engine.py:351-361`) | single regeneration (`main.py:174-181`) |
| Provider down at runtime | try/except in talk loop (`main.py:336-338`, `:348-350`) | error line shown, game continues |
| Epilogue | — | templated error line (`engine.py:482-483`) |

WHO verdict, scoring, timers, and item visibility are **pure code** — never an
LLM decision.

---

## 12. Quick verification checklist

Run these to confirm the spec against live behavior:

```bash
# 1. Deterministic plumbing, no network (mock both agents in config.yaml):
python main.py --seed 7 --mode developer --lang en
#   -> read logs/<session>/developer.jsonl: ground_truth, culprit_selection
#      (source should be "fallback" with mock), deeds, boundary_gossip, npc_memory

# 2. Confirm the accident branch (~10%): loop seeds until is_murder=false,
#    or temporarily raise special_rolls weight in case.yaml.

# 3. Confirm the WHO gate: accuse the wrong person with a perfect why/how
#    -> score must cap at 49 and rating must be "SO CLOSE" or "MISCARRIAGE".
```
