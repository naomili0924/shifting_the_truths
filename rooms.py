"""
rooms.py — the visual *manifest* layer for Shifting Truth's painted UI.

Manifest first, paint second. The searchable structure of a scene (its spots and
plot-conditional items) is authored in case.yaml and resolved against the rolled
ground truth elsewhere; this module turns one act into a **room manifest** the
renderer can paint and place chips from:

    {
      "act": 1,
      "title": "...",
      "prompt": "<english SDXL backdrop prompt, atmosphere only>",
      "spots": { "<spot_id>": {"x": 0.0..1, "y": 0.0..1}, ... }   # chip anchors
    }

The judge LLM may author the manifest each run (varied per-run backdrops),
passed through a verification gate that guarantees it references exactly the
act's real spot ids; if the judge is absent or its output is invalid, a
deterministic manifest is used so the game always renders. Either way the chips
are buttons driven by the manifest — never anchored to pixels in the painting.
"""

from __future__ import annotations

from engine import _extract_json  # reuse the tolerant JSON extractor


# ---- chip anchors (normalized 0..1 over the backdrop) -------------------------

def _grid_positions(spot_ids: list[str]) -> dict[str, dict]:
    """Spread spots across the lower portion of the frame, where hotspots sit."""
    n = max(len(spot_ids), 1)
    cols = min(n, 4)
    pos: dict[str, dict] = {}
    for i, sid in enumerate(spot_ids):
        col, row = i % cols, i // cols
        x = (col + 0.5) / cols
        y = min(0.58 + 0.18 * row, 0.9)
        pos[sid] = {"x": round(x, 3), "y": round(y, 3)}
    return pos


# ---- prompt builders ---------------------------------------------------------

_STYLE_NOTE = "wide establishing shot, no people, no text"


def deterministic_backdrop_prompt(case: dict, act: dict) -> str:
    setting = (case.get("scenario", {}) or {}).get("setting", "").strip()
    # First sentence of the setting is enough atmosphere for the backdrop.
    setting_lead = setting.split(".")[0].strip() if setting else ""
    title = act.get("title", "").strip()
    spot_names = [s.get("name", s.get("id", "")) for s in act.get("spots", [])]
    spots_clause = ", ".join(n for n in spot_names if n)
    parts = [p for p in (title, setting_lead) if p]
    head = ". ".join(parts) if parts else "A dim interior at night"
    return f"{head}. In view: {spots_clause}. {_STYLE_NOTE}"


def default_manifest(case: dict, act: dict) -> dict:
    spot_ids = [s["id"] for s in act.get("spots", [])]
    return {
        "act": act.get("act"),
        "title": act.get("title", ""),
        "prompt": deterministic_backdrop_prompt(case, act),
        "spots": _grid_positions(spot_ids),
    }


def item_prompt(item: dict) -> str:
    """English prompt for a single evidence item (a close-up object study)."""
    name = item.get("name", "an object")
    detail = (item.get("found_text", "") or "").split(".")[0].strip()
    base = f"A single evidence object: {name}"
    if detail:
        base += f" — {detail}"
    return (
        base
        + ". Close-up still life of one object, centered, dramatic candlelit noir "
        "lighting, plain dark background, painted illustration, no text"
    )


def portrait_prompt(char: dict) -> str:
    """English portrait prompt for one of the fixed faces (stable across runs)."""
    name = char.get("character", "a guest")
    role = char.get("role", "")
    look = (char.get("public_story", "") or "").split(".")[0].strip()
    desc = ", ".join(p for p in (role, look) if p)
    return (
        f"Character portrait of {name}"
        + (f", {desc}" if desc else "")
        + ". Head and shoulders, dramatic candlelit noir lighting, neutral "
        "expression, plain dark background, painted illustration"
    )


# ---- judge job: author the room's visual manifest (verified) -----------------

def judge_room_layout(case: dict, act: dict, judge, lang: str = "en") -> dict:
    """Ask the judge to write this act's backdrop prompt + chip anchors.

    Verified: the returned spot ids must be exactly the act's spot ids and every
    (x, y) must be in range. Any deviation, or any judge error, falls back to the
    deterministic manifest — the renderer always gets a valid manifest.
    The output is internal (drives the English image prompt + chip placement),
    so it is language-independent regardless of the game language.
    """
    spots = act.get("spots", [])
    spot_ids = [s["id"] for s in spots]
    if not spot_ids:
        return default_manifest(case, act)

    spot_lines = "\n".join(
        f'- id "{s["id"]}": {s.get("name", s["id"])}' for s in spots
    )
    setting = (case.get("scenario", {}) or {}).get("setting", "")
    system = (
        "You are the art director for a noir murder-mystery point-and-click "
        "adventure. Given a scene and its searchable spots, write ONE English "
        "image prompt for a painted background (atmosphere only — no people, no "
        "text) and place each spot at a normalized (x, y) position in [0,1] where "
        "it would plausibly appear in that painting (x: 0 left .. 1 right, y: 0 "
        "top .. 1 bottom). Reply with ONLY JSON of the form: "
        '{"prompt": "<background prompt>", "spots": {"<id>": {"x": <0..1>, '
        '"y": <0..1>}}}.'
    )
    user = (
        f"SCENE TITLE: {act.get('title', '')}\n"
        f"SETTING: {setting}\n"
        f"SEARCHABLE SPOTS:\n{spot_lines}\n\n"
        "Use every spot id exactly once. Place spots where their objects sit in "
        "the painting you describe."
    )
    try:
        raw = judge.chat(system, [{"role": "user", "content": user}], max_tokens=500)
        data = _extract_json(raw)
        assert data and isinstance(data.get("prompt"), str) and data["prompt"].strip()
        in_spots = data.get("spots") or {}
        assert set(in_spots.keys()) == set(spot_ids)
        out_spots = {}
        for sid, p in in_spots.items():
            x = min(max(float(p["x"]), 0.05), 0.95)
            y = min(max(float(p["y"]), 0.2), 0.92)
            out_spots[sid] = {"x": round(x, 3), "y": round(y, 3)}
        prompt = data["prompt"].strip()
        if _STYLE_NOTE.split(",")[0] not in prompt:
            prompt = f"{prompt}, {_STYLE_NOTE}"
        return {
            "act": act.get("act"),
            "title": act.get("title", ""),
            "prompt": prompt,
            "spots": out_spots,
        }
    except Exception:
        return default_manifest(case, act)
