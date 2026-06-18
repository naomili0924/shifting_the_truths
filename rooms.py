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

from engine import _extract_json, item_present  # reuse tolerant JSON + plot gate


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


def object_prompt(name: str, found_text: str = "") -> str:
    """English prompt for ONE object embedded *in* the scene (not a plain card).

    Unlike item_prompt (a close-up still life on a dark background), this describes the
    object as it sits in the room, so the inpaint blends it into the backdrop."""
    detail = (found_text or "").split(".")[0].strip()
    base = f"{name}"
    if detail:
        base += f", {detail}"
    return (base + ", a single realistic object resting in the scene, detailed, "
            "candlelit noir lighting, in focus, no text")


# ---- two sub-scenes per act, with embedded collectible + decoy objects --------

# Plot-irrelevant props: inspectable ("touch and check") but never collectible. A small
# deterministic pool fits the noir-estate setting; each scene draws a couple by index, so
# the same run is reproducible and no LLM call is needed. Each entry is
# (en_name, en_flavor, zh_name, zh_flavor): the player sees the localized name/flavor, but
# the image prompt is always built from the English name (SDXL is English-prompted).
_DECOYS = [
    ("an empty wine bottle", "An empty estate vintage. Dust, a tide-line of old red, nothing more.",
     "一只空酒瓶", "庄园的陈年空瓶，积着灰，瓶壁一圈暗红的酒痕，仅此而已。"),
    ("a guttered candle stub", "Burned to the holder during the blackout. Just wax and a cold wick.",
     "一截烧尽的蜡烛", "停电时烧到了底座，只剩蜡油和冷掉的烛芯。"),
    ("a stack of yellowed ledgers", "Decades of wine accounts. Tedious, and none of them tonight's business.",
     "一摞发黄的账簿", "几十年的酒庄账目，枯燥乏味，与今晚无关。"),
    ("a coil of weathered rope", "Garden rope, stiff with damp. It has hung here for years.",
     "一卷风化的绳子", "花园里的绳子，受潮发硬，在这儿挂了好些年。"),
    ("a chipped porcelain cup", "Cold tea, a lipstick ghost on the rim. Someone's, long before tonight.",
     "一只缺口的瓷杯", "冷掉的茶，杯沿一抹口红印，是某人的，远在今晚之前。"),
    ("a dusty oil lamp", "Unlit, wick dry. A relic from before the estate had wiring.",
     "一盏积灰的油灯", "没点着，灯芯干枯。庄园通电之前留下的旧物。"),
    ("a withered potted fern", "Past saving. The estate stopped tending small things a while ago.",
     "一盆枯萎的蕨", "已经救不活了。庄园早就顾不上这些小东西。"),
    ("a folded estate map", "A tourist's map of the vineyard. Pretty, and useless to you.",
     "一张折叠的庄园地图", "游客用的酒庄地图，挺好看，对你没用。"),
]


def _layout(n: int) -> list[dict]:
    """Spread n objects across the lower-middle of the frame with varied sizes, so they
    read as 'placed in the room'. Returns normalized {x, y, w, h} centres + sizes."""
    if n <= 0:
        return []
    cols = min(n, 3)
    out = []
    for i in range(n):
        col, row = i % cols, i // cols
        x = (col + 0.5) / cols
        y = 0.6 + 0.16 * row
        # alternate sizes a little so it doesn't look like a grid of identical boxes
        w = 0.26 if i % 2 == 0 else 0.20
        h = 0.24 if i % 3 else 0.30
        out.append({"x": round(min(max(x, 0.12), 0.88), 3),
                    "y": round(min(y, 0.85), 3), "w": w, "h": h})
    return out


def scene_manifests(case: dict, act: dict, gt, lang: str = "en",
                    en_case: dict | None = None, decoys_per_scene: int = 2) -> list[dict]:
    """Build the act's TWO sub-scenes, each with embedded objects.

    Splits the act's spots in half (scene A / scene B). Each scene's objects are the
    *present* clue items in its spots (kind 'collectible', pickable) plus a couple of
    plot-irrelevant decoys (kind 'decoy', inspect-only). Game logic never depends on the
    painted pixels — positions are anchors for hotspot chips, exactly like the old chips.

    Player-facing text (names, found_text, flavor) is in the game language; every IMAGE
    prompt (base + objects) is built from the English case (``en_case``, matched by item
    id / act number), because SDXL is English-prompted — so a Chinese game still paints
    good art. ``en_case`` defaults to ``case`` (an English game)."""
    en_case = en_case or case
    act_no = act.get("act")
    en_act = next((a for a in en_case.get("acts", []) if a.get("act") == act_no), act)
    en_items = {it["id"]: it for s in en_act.get("spots", [])
                for it in s.get("items", [])}                 # id -> English item (for prompts)
    setting = (en_case.get("scenario", {}) or {}).get("setting", "").strip()
    setting_lead = setting.split(".")[0].strip() if setting else ""
    act_title_en = en_act.get("title", "").strip()

    spots = act.get("spots", [])
    half = (len(spots) + 1) // 2
    groups = [spots[:half], spots[half:]]                     # single-spot acts -> [spot],[]

    scenes = []
    for gi, group in enumerate(groups):
        sid = f"{act_no}-{'ab'[gi]}"
        # Base prompt = ATMOSPHERE ONLY (English). We deliberately do NOT list spot/item
        # names: clue objects are inpainted in afterwards, and naming e.g. "Diana's body"
        # in the backdrop prompt conflicts with the no-people style. A coherent location
        # is all the base needs.
        head = ". ".join(p for p in (act_title_en, setting_lead) if p) or "A dim interior at night"
        prompt = f"{head}. A detailed atmospheric establishing shot of the location, {_STYLE_NOTE}"

        # collectibles = present clue items in this group's spots (display localized;
        # image prompt from the English item with the same id)
        objects = []
        for s in group:
            for item in s.get("items", []):
                if not item_present(item, gt):
                    continue
                en_it = en_items.get(item["id"], item)
                objects.append({
                    "id": item["id"], "kind": "collectible",
                    "name": item["name"], "found_text": item.get("found_text", ""),
                    "spot_id": s["id"],
                    "obj_prompt": object_prompt(en_it["name"], en_it.get("found_text", "")),
                })
        # decoys = a couple of plot-irrelevant props, chosen deterministically per scene
        for k in range(decoys_per_scene):
            en_name, en_flavor, zh_name, zh_flavor = _DECOYS[(act_no * 2 + gi + k) % len(_DECOYS)]
            objects.append({
                "id": f"{sid}-decoy{k}", "kind": "decoy",
                "name": zh_name if lang == "zh" else en_name,
                "flavor": zh_flavor if lang == "zh" else en_flavor,
                "obj_prompt": object_prompt(en_name, en_flavor),
            })

        # place every object
        for obj, pos in zip(objects, _layout(len(objects))):
            obj.update(pos)

        scenes.append({"id": sid, "act": act_no, "title": act.get("title", ""),
                       "prompt": prompt, "objects": objects})
    return scenes


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
