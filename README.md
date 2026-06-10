# Shifting Truth — three-act playable build

A murder mystery where suspects are LLM agents, the plot re-rolls
every game, and the night unfolds in three designer-authored acts
with timed search and interrogation phases.

## Quick start

```bash
pip install pyyaml
export ANTHROPIC_API_KEY=sk-ant-...
python main.py                    # uses config.yaml
python main.py --seed 7           # reproduce a specific plot
```

No key / no network? Test the plumbing with mock agents: set
`provider: mock` for both agents in config.yaml.

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
the culprit — it still seeds method, clue and accident rolls. To see or
reproduce a specific game's hidden truth, read the developer debug log
(`game.debug_log` in config.yaml, default `judge_debug.jsonl`): one JSON
line per game records the judge's culprit + motive pick (and its
reasoning) plus the fully resolved ground truth. It's written to disk
only, never shown to the player.

## Files

- `config.yaml` — agent brains + time costs
- `case.yaml` — the authored case: cast, flaws, methods, acts, items
- `engine.py` — director roll, judge jobs, prompts, referee, verdict
- `providers.py` — LLMProvider interface: Anthropic / ONNX / mock
- `main.py` — the act/phase game loop
