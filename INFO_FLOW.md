# Shifting Truth — Information Flow & Knowledge Model

> Companion to `GAME_MECHANICS.md`. That document traces *what happens*; this one
> traces *who knows what, when* — the duties of each agent at each stage, the
> exact knowledge in their hands, and the channels by which information moves
> between the judge, the NPCs, and the player. Every claim cites `file:line`.

---

## 0. The four knowledge holders

| Holder | Persistence | Sees the hidden truth? | Implemented as |
|---|---|---|---|
| **Ground truth** | created once at roll, immutable | — (it *is* the truth) | `GroundTruth` dataclass, `engine.py:53` |
| **Judge LLM** | **stateless across jobs** — knows only what each job's prompt passes in | varies per job (see §3-§8) | `agents.judge` provider; each job is one `judge.chat(...)` |
| **NPC LLM** (×5 personas) | **per-NPC persistent history**; never sees another NPC's history | only the culprit persona sees motive+secret; innocents are told they do *not* know | `agents.npc` provider; `self.histories[name]`, `self.base_prompt[name]` |
| **Player** | accumulates evidence + transcript in their own head | no — must infer | the human; their tools are `evidence`, `notes`, the transcript |

> **Critical design fact #1 — the judge has no memory.** "The judge" is just a
> provider; each duty (select, deeds, gossip, grade, epilogue) is an *independent*
> `chat()` call (`providers.py:45`). It knows **only** the text that specific call
> puts in front of it. So "what the judge knows" is different at every stage, and
> deliberately limited so it cannot leak (e.g. the gossip job is **not told who
> the killer is**, §6).
>
> **Critical design fact #2 — NPCs are siloed.** Each suspect has its own message
> history (`self.histories[name]`, `main.py:123`). NPC A never sees the player's
> conversation with NPC B. The *only* cross-NPC channel is **boundary gossip**,
> which the judge writes and the engine injects into the listener's prompt (§6).

---

## 1. The information-flow timeline (overview)

```
AUTHORING (case.yaml, i18n.py)         static knowledge baseline
        │
        ▼
[ROLL]  Director.roll()                creates GROUND TRUTH
        │   ├─ accident? (10%)         engine.py:103-105
        │   └─ else murder:
        │        ├─ judge picks culprit+flaws ........ JUDGE JOB #0  (§3)
        │        ├─ code rolls method/motive/timeline  (§2)
        │        └─ code distributes 2 clues to innocents (§2)
        ▼
[SETUP] judge writes deeds ............................ JUDGE JOB #1  (§4)
        build each NPC's system prompt (knowledge packet) (§5)
        ▼
[PLAY]  per act: SEARCH ⇄ TALK
        │   player ↔ NPC turns ......................... NPC JOB      (§7)
        │   referee guards each reply
        ▼
[BOUNDARY] between acts: judge deals gossip ........... JUDGE JOB #2  (§6)
        │   (folds in a summary of the player's questions)
        ▼   ...repeat acts...
[END]   forced accusation
        │   code grades WHO ............................ (§8.1)
        │   judge grades WHY/HOW ....................... JUDGE JOB #3 (§8.2)
        │   judge writes epilogue ..................... JUDGE JOB #4 (§8.3)
        ▼
        verdict shown
```

---

## 2. Authoring → Ground Truth (what knowledge exists)

**Before the roll**, the only knowledge is *authored* in `case.yaml` /
`i18n.py`: the cast, each character's role + public story + private story +
flaws (with `motive_seed`, `trigger_vs_victim`, `behavioral_tells`,
`lie_tendency`), `knows_about_others`, the shared `timeline_skeleton`, the
`method_options`, and the `plausibility_matrix` of valid (culprit, flaws) combos.

**The roll** (`Director.roll`, `engine.py:92`) turns that into one game's
**ground truth** (`engine.py:53-61`):

- `is_murder`, `culprit`, `active_flaws`, `method`, `motive` (woven string,
  `engine.py:148-159`), `secret_timeline` (`engine.py:184-187`),
  `distributed_clues` (`engine.py:189-207`).

This object is **the** source of truth. Nobody but the developer log ever sees it
whole. It is fed into the agents only in carefully sliced portions, described
below.

> Clue distribution (`engine.py:203`) hands **2 innocent NPCs** one true sighting
> clue each — this is the only ground-truth fact baked directly into an innocent's
> knowledge (see §5).

---

## 3. JUDGE JOB #0 — choose the culprit

**When:** during the roll, murder branch only (`main.py:136-141` →
`judge_select_culprit`, `engine.py:231`).

**What the judge is GIVEN** (`select_system` + `select_user`):
- The victim's name.
- A **numbered menu** of valid `(culprit, flaws, strength, motive_seeds)` combos —
  *strong* and *medium* only (`engine.py:246-251`).
- Nothing else. **At this moment the method, secret timeline and clues do not yet
  exist** — they are rolled *after* the judge returns (`engine.py:106-107`,
  `_roll_murder`). So the judge picks blind to the mechanism.

**Duty:** "Pick the most dramatically compelling option and vary your choice
across games." Return strict JSON `{culprit, flaws, rationale}`.

**What the judge RETURNS / what flows out:** `(culprit, flaws)` → the selector →
the director, which then rolls method/motive/timeline/clues around that choice.

**Validation & fallback:** the pick must resolve to a real name and exactly match
an on-menu combo (`engine.py:259-263`); otherwise a weighted code pick
(`engine.py:267-271`). Logged as `culprit_selection` with `source` (`main.py:140`).

> Information boundary: the rationale is "for the designer's eyes only" — it goes
> to the dev log, never to a player or another agent.

---

## 4. JUDGE JOB #1 — write every NPC's true deeds

**When:** setup, right after the roll (`main.py:113` → `judge_generate_deeds`,
`engine.py:529`).

**What the judge is GIVEN** (`deeds_system` + `deeds_user`):
- The list of character **names**.
- The **motive** string and the **secret events** string (`engine.py:535-537`).
- The shared fixed beats (blackout 9:05, partial power 9:28, the fall 9:41).
- It is told it "knows the full hidden truth" and must write each character's TRUE
  memory of the 9:05-9:41 blackout window.

**Duty / rules enforced by the prompt:** exactly 3 second-person facts per
character; **innocents must NOT know who caused the fall and must NOT witness the
crime**; each innocent gets one unaccounted-for gap; facts must be mutually
consistent; **the culprit's entry must be an empty list** (their truth lives in
the secret block instead).

**What flows out:** `deeds[name]` → stored on the Game (`main.py:113`) and folded
into each NPC's system prompt (§5). The culprit's deeds are forced empty
(`engine.py:557`).

**Validation & fallback:** key set must equal the cast; 1-4 lines each; no
confession phrases; culprit's name must not sit beside a kill verb
(`engine.py:542-557`). On failure → `default_deeds` templates (`engine.py:560`).
Logged as `deeds` (`main.py:115`).

> This is the **anti-confabulation** layer: by pre-writing each NPC's true memory,
> the NPC LLM has grounded facts to speak from and is less likely to invent
> contradictory events.

---

## 5. SETUP — what each NPC persona is given to know

`build_npc_system_prompt` (`engine.py:284-345`) assembles each suspect's
knowledge packet. **Every NPC** receives (the strings are in `i18n.py`'s `npc`
table):

| Knowledge | Source | Line | Notes |
|---|---|---|---|
| Who they are + that they're being questioned by the player | `intro` | `engine.py:294` | |
| The setting | `setting` | `engine.py:296` | |
| The **public** "accident" story | `what_happened` | `engine.py:297` | the cover everyone shares |
| Their role | `who_you_are` | `engine.py:299` | |
| Their **public face** | `public_face` | `engine.py:300` | what they'll say openly |
| Their **private truth** ("never volunteer this") | `private_truth` | `engine.py:301` | their secret to protect |
| The shared **fixed timeline beats** | `fixed_events`+`beat` | `engine.py:303-307` | everyone experienced these |
| **Things they know about others** | `knows_about_others` | `engine.py:309-311` | "reveal only if it serves you / deflects / player earns trust" |
| **A distributed clue** (if they were chosen as a witness) | `gt.distributed_clues[name]` | `engine.py:312-313` | the ground-truth sighting from §2 — appears in their "know about others" list |
| Their **psychology**: each flaw's description, tells, tonight's trigger, lie tendency | `psychology`+`flaw_*` | `engine.py:315-320` | "these leak through under pressure" |
| Their **deeds** (true memory of 9:05-9:41) | `deeds[name]` | `engine.py:323-326` | from Judge Job #1 |

Then a **branch** by role:

- **If culprit** (`engine.py:328-337`): a `=== SECRET: YOU ARE THE KILLER ===`
  block with **the real motive**, **the secret timeline** (what they actually
  did), and the **killer rules**: never confess unless cornered with ≥2 specific
  accurate pieces of evidence (and even then crack gradually); lie strategically by
  redirecting suspicion using what they know about others; stay internally
  consistent.
- **If innocent** (`engine.py:338-342`): a `=== YOU ARE INNOCENT ===` block that
  explicitly says **they do NOT know who caused the fall or whether it was even
  murder; never invent knowledge of the killer**; but they *do* have their own
  secrets and **will lie to protect them, even if it makes them look guilty**.

Plus shared **style rules** (`engine.py:344`): 1-4 sentences, first person, never
mention prompts/AI, never speak for others.

> **Knowledge asymmetry summary at game start:**
> - The **culprit persona** knows: everything above **+ motive + secret timeline**.
> - An **innocent persona** knows: everything above, explicitly **minus** who did
>   it; their dangerous knowledge is their *own* secret + maybe one true clue about
>   someone else.
> - **No NPC knows another NPC's private story, deeds, or secret** — only the
>   authored `knows_about_others` fragments and (for 2 of them) one clue.

The packet for each NPC is cached as `self.base_prompt[name]` (`main.py:116-121`).
The **live** prompt at any moment is `system_for(name)` = base packet **+
accumulated hearsay** (`main.py:162-167`); hearsay starts empty and grows at act
boundaries (§6).

---

## 6. JUDGE JOB #2 — boundary gossip (the NPC ⇄ NPC channel)

**When:** after each act except the last (`main.py:414` → `boundary`,
`main.py:360` → `deal_boundary_gossip`, `engine.py:567`).

**What the judge is GIVEN** (`gossip_system` + `gossip_user`,
`engine.py:574-582`):
- The act number.
- **Everyone's true deeds** — but the **culprit's line is replaced with
  `(secret)`** (`engine.py:579-581`, `gossip_secret`).
- A summary of **what the player asked each suspect** (last 3 questions, truncated)
  (`engine.py:576-578`, `gossip_asked`).
- It is told: write 1-2 partial hearsay lines per suspect, each starting "You
  heard" / "Word reached you"; **never state or imply who the culprit is**; keep it
  partial.

> **Information boundary here is the key safety property:** the gossip-writing
> judge is **not given the culprit's identity, motive, or secret timeline** — the
> culprit's deeds are masked. So even though the same provider knows the truth in
> *other* jobs, *this* job structurally cannot leak it (and a regex strips any
> culprit-name+kill-verb line anyway, `engine.py:584-588, 597`).

**Duty:** decide what partial fragments each suspect "picked up" — both about
others' movements and about **what the journalist has been asking**.

**What flows out and to whom:** each line is prefixed with the hearsay marker and
appended to `self.extras[listener]` (`main.py:365-367`), which `system_for`
splices into that listener's system prompt next act (`main.py:164-166`). So:

```
player's questions to NPC A ──► questions_log[A]
                                      │  (summarized)
deeds of all NPCs (culprit masked) ──┤
                                      ▼
                              judge writes gossip
                                      │
                          extras[B] += "you heard ..."
                                      ▼
                  NPC B's prompt next act now contains it
```

This is the mechanism behind **"careless questions travel"**: press Sofia about
Marco in Act 1, and Marco may "hear" about it in Act 2.

**Validation & fallback:** ≤2 lines/name, drop culprit-leak lines, require ≥1
surviving line (`engine.py:594-601`); else deterministic slips + a "they kept
asking X" line (`engine.py:603-613`). Logged as `boundary_gossip` (`main.py:368`).

---

## 7. PLAY — the player ⇄ NPC channel (per turn)

**When:** every question or `show` during a TALK phase (`main.py:314-352`).

### 7.1 What the player knows / accumulates
- Starts with: the **public accident story** + setting + cast (printed in `play`,
  `main.py:392-397`).
- Gains **evidence** by searching (plot-aware items, §5/§5.1 of GAME_MECHANICS) →
  `self.evidence` (`main.py:274-276`), viewable with `evidence`.
- Gains **statements** from suspects → stored per NPC in `histories`, reviewable
  with `notes` (`main.py:209-220`).
- Everything else is **inference** — the game never tells the player the truth.

### 7.2 What flows INTO an NPC on a turn — `npc_reply` (`main.py:169-183`)
- **System prompt:** `system_for(name)` = base knowledge packet (§5) + all hearsay
  accumulated so far (§6).
- **Messages:** that NPC's **entire prior history with the player**
  (`self.histories[name]`) plus the new user line (`main.py:170-172`). The NPC
  therefore remembers everything *it* discussed with the player, and is told to
  stay internally consistent — but it sees **nothing** from other NPCs' chats.
- For `show <item>`, the user line is a formatted "I present <item>: <found_text>"
  (`main.py:331-332`); the NPC reacts to the evidence text.

### 7.3 What flows OUT and the referee guard
- The NPC reply is checked by `referee_check` (`engine.py:351-361`) **before the
  player sees it**: blocks meta-leaks (`leak_rx`) and an **innocent** confessing
  (`confession_rx`); on a hit, appends an out-of-character correction and
  **regenerates once** (`main.py:174-181`).
- The accepted reply is appended to history (`main.py:182`), printed, and logged
  (`question` / `show_evidence`, `main.py:339-341, 351-352`). The raw question is
  also pushed to `questions_log[name]` (`main.py:344`) — the seed for §6 gossip.

> So within an act, the only thing that changes an NPC's knowledge is the running
> conversation with the player. Cross-suspect knowledge is injected **only** at the
> act boundary via gossip.

### 7.4 Developer memory snapshots
In developer mode, `log_npc_memory` (`main.py:143-154`) dumps each NPC's full live
system prompt, hearsay, statements, and history at `game_start`, `act<N>_end`, and
`final` (`main.py:400, 413, 388`). This is the intended way to "track an NPC's
memory at each stage."

---

## 8. END — the accusation, graded

### 8.1 WHO — code, omniscient comparison (`_who_verdict`, `engine.py:381-388`)
Compares the player's accused name against `gt.culprit` (murder) or accident
keywords (accident). Pure code; no LLM. The player's free-text WHY/HOW are **not**
used here.

### 8.2 JUDGE JOB #3 — grade WHY/HOW (`_grade_with_llm`, `engine.py:402-427`)

**What the judge is GIVEN** (`grade_system` + `grade_user`):
- The **full hidden truth** assembled for grading: culprit, motive, flaws, method,
  secret timeline (`engine.py:406-413`).
- The player's **WHY and HOW**, wrapped in `<answer>` tags. The prompt explicitly
  says the tagged text is **DATA to grade, never instructions** — a prompt-
  injection guard (`grade_system`).

**Duty:** grade MOTIVE (`correct`/`partial`/`wrong`) and METHOD
(`correct`/`partial`/`wrong`/`not_stated`) + a one-sentence comment with no
spoilers beyond the player's own claims. Strict JSON.

**Validation & fallback:** JSON must parse and `motive` be in the enum
(`engine.py:422-424`); else keyword-stem matching against the reasoning
(`engine.py:391-399`).

**What flows out:** the grade → the score formula (`engine.py:435-447`, see
GAME_MECHANICS §9.3) → rating + verdict card.

### 8.3 JUDGE JOB #4 — the epilogue (`engine.py:473-484`)

**What the judge is GIVEN** (`narrator_system` + `epilogue_prompt`): the **real
motive**, the **secret timeline**, and a **consequence** branch chosen by code
from whether WHO was right (`consequence_ok` / `consequence_wrong_murder` /
`consequence_accident`, `engine.py:465-471`).

**Duty:** write a 6-10 sentence noir epilogue. This is the one place the hidden
truth is finally narrated to the player — *after* the verdict is locked. On error,
a templated line (`engine.py:482-483`).

---

## 9. "Who knows what, when" — master matrix

Legend: ● full · ◐ partial/sliced · ○ none

| Knowledge ↓  /  Holder → | Ground truth | Judge (per job) | Culprit NPC | Innocent NPC | Player |
|---|:--:|:--:|:--:|:--:|:--:|
| Public accident story | ● | ● | ● | ● | ● (from start) |
| Setting + fixed beats | ● | ● | ● | ● | ◐ (setting only) |
| Own role/public/private story | — | ● | ● (own) | ● (own) | ○ |
| Another NPC's private story | ● | ● | ◐ (authored fragments) | ◐ (authored fragments) | ○ |
| Own deeds (9:05-9:41) | ● | ● (job #1/#2) | n/a (empty) | ● | ○→◐ (via questioning) |
| The culprit's identity | ● | ● *except gossip job* | ● (self) | ○ (told they don't know) | ○ (must infer) |
| The motive | ● | ● (#0 seeds, #1, #3, #4) | ● | ○ | ○ (states a guess at end) |
| The method | ● | ◐ (#1 via secret; #3 full) | ● | ○ | ○ (states a guess at end) |
| The secret timeline | ● | ● (#1,#3,#4) | ● | ○ | ○ |
| Distributed sighting clues | ● | ● | (n/a) | ◐ (2 of them, 1 each) | ◐ (only if the NPC tells them) |
| The player's questions to others | — | ◐ (gossip job summary) | ◐ (via hearsay next act) | ◐ (via hearsay next act) | ● (their own) |
| Found evidence items | (defined) | ○ | ◐ (only what's `show`n to them) | ◐ (only what's `show`n) | ● (the pouch) |
| Another NPC's live conversation | — | ○ | ○ | ○ | ● (only their own transcripts) |

> The most important asymmetries to verify against the code:
> 1. **Innocents are *told* they don't know the killer** (`engine.py:339-342`) —
>    they aren't merely missing it; the prompt forbids inventing it.
> 2. **The gossip job is the one judge call kept ignorant of the culprit**
>    (`engine.py:579-581`) — confirm the culprit's deeds are masked to `(secret)`.
> 3. **NPCs are siloed**; the gossip channel (§6) is the *only* path by which one
>    suspect learns anything about another's movements or about the player's
>    questioning — and it arrives only at act boundaries.

---

## 10. Interaction channels (the edges of the graph)

| From → To | Channel | Code | When |
|---|---|---|---|
| Author → Ground truth | the roll | `engine.py:92-207` | once |
| Ground truth → Judge | sliced into each job's prompt | `engine.py:235,535,582,406,473` | per job |
| Judge → Ground truth | culprit/flaws pick | `engine.py:106`, `main.py:136-141` | roll |
| Ground truth → Innocent NPC | 2 distributed clues | `engine.py:203`, `engine.py:312-313` | setup |
| Judge → NPC | deeds (true memory) | `engine.py:529`, `main.py:113` | setup |
| Player → NPC | questions / shown evidence | `main.py:169-183` | every turn |
| NPC → Player | reply (after referee) | `main.py:182`, `engine.py:351` | every turn |
| Player → (gossip) → NPC | questions summarized, re-injected as hearsay | `questions_log`→`engine.py:576`→`extras`, `main.py:344,365` | act boundary |
| NPC ⇄ NPC | judge-written gossip about deeds | `engine.py:567`, `main.py:365-367` | act boundary |
| Player → Judge | the WHY/HOW answer (as graded data) | `engine.py:402-421` | endgame |
| Judge → Player | grade + epilogue (truth finally revealed) | `engine.py:436-485` | endgame |
| Everything → dev log | spoiler layer | `main.py:105,115,140,368`, `gamelog.py:71` | developer mode |
```
