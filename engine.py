"""
engine.py — case loading, the director (per-playthrough roll),
knowledge packet compilation, a v0 referee, and the accusation judge.

Everything here is deterministic given a random seed, so a
playthrough is reproducible for debugging (--seed 42).

All player- and LLM-facing text is localized: functions take a `lang`
(default "en") and read their strings from i18n.t(lang). Character names
(victim, the concealer) come from the case file, so each language ships
its own case and the engine itself is language-agnostic.
"""

from __future__ import annotations
import random
import re
import yaml
from dataclasses import dataclass, field

from providers import LLMProvider
from i18n import t


# ----------------------------------------------------------------
# Case loading
# ----------------------------------------------------------------
def load_case(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        case = yaml.safe_load(f)
    for key in ("scenario", "victim", "characters", "timeline_skeleton",
                "plausibility_matrix", "method_options"):
        if key not in case:
            raise ValueError(f"Case file missing required key: {key}")
    return case


def victim_name(case: dict) -> str:
    """Victim's name, read from the case (engine hard-codes no names)."""
    return case["victim"]["name"]


def concealer_name(case: dict) -> str | None:
    """The character who buried the inspection report (has guilty_concealment)."""
    for ch in case["characters"]:
        if any(fl["id"] == "guilty_concealment" for fl in ch["flaws"]):
            return ch["character"]
    return None


# ----------------------------------------------------------------
# Ground truth produced by the director's roll
# ----------------------------------------------------------------
@dataclass
class GroundTruth:
    is_murder: bool
    culprit: str | None           # None => true accident
    active_flaws: list[str]
    method: dict | None
    motive: str
    secret_timeline: list[str]
    distributed_clues: dict[str, list[str]] = field(default_factory=dict)


# Valid (culprit, flaws) combos authored in the plausibility matrix.
# Returned as (culprit, flaws, strength, weight); shared by the code
# fallback picker and the judge-selection prompt so both stay in sync.
def culprit_options(case: dict) -> list[tuple]:
    strengths = {"strong": 3, "medium": 2, "weak": 0}
    opts = []
    for entry in case["plausibility_matrix"]:
        for roll in entry["valid_rolls"]:
            strength = roll.get("strength", "weak")
            w = strengths.get(strength, 0)
            if w > 0:
                opts.append((entry["culprit"], list(roll["flaws"]),
                             strength, w))
    return opts


class Director:
    """Rolls one playthrough from the authored case."""

    def __init__(self, case: dict, seed: int | None = None, lang: str = "en"):
        self.case = case
        self.rng = random.Random(seed)
        self.lang = lang
        self.L = t(lang)
        self.victim = victim_name(case)
        self.concealer = concealer_name(case)

    # -- public ---------------------------------------------------
    def roll(self, selector=None) -> GroundTruth:
        """Roll one playthrough.

        `selector`, if given, is a no-arg callable returning
        (culprit, flaws) for the murder branch — this is how the judge
        LLM gets to choose. When omitted, a deterministic weighted code
        pick is used (used by tests and as the offline fallback). The
        accident special-roll is decided first, so the selector is only
        consulted when the night is actually a murder.
        """
        special = self.case.get("special_rolls", [])
        for sp in special:
            if sp["id"] == "true_accident" and self.rng.random() < float(sp.get("weight", 0)):
                return self._roll_true_accident(sp)
        culprit, flaws = selector() if selector else (None, None)
        return self._roll_murder(culprit, flaws)

    # -- internals --------------------------------------------------
    def _char(self, name: str) -> dict:
        for ch in self.case["characters"]:
            if ch["character"] == name:
                return ch
        raise KeyError(name)

    def _flaw(self, char: dict, flaw_id: str) -> dict:
        for fl in char["flaws"]:
            if fl["id"] == flaw_id:
                return fl
        raise KeyError(flaw_id)

    def _weighted_culprit(self) -> tuple[str, list[str]]:
        """Deterministic weighted pick from the plausibility matrix."""
        opts = culprit_options(self.case)
        culprit, flaws, _strength, _w = self.rng.choices(
            opts, weights=[o[3] for o in opts], k=1)[0]
        return culprit, flaws

    def _roll_murder(self, culprit: str | None = None,
                     flaws: list[str] | None = None) -> GroundTruth:
        L = self.L
        # 1) The culprit + flaw combo. Supplied by the judge LLM via the
        #    selector; falls back to a weighted code pick if absent.
        if culprit is None or flaws is None:
            culprit, flaws = self._weighted_culprit()
        char = self._char(culprit)

        # 2) Pick a method the culprit has access to.
        methods = [
            m for m in self.case["method_options"]
            if m["requires_access"] == ["any"] or culprit in m["requires_access"]
        ]
        # Weighted pick: the universally-accessible method (push) is
        # reachable by every culprit and otherwise swamps the others.
        mweights = [float(m.get("weight", 1)) for m in methods]
        method = self.rng.choices(methods, weights=mweights, k=1)[0]

        # 3) Weave the motive from the rolled flaws' seeds + triggers.
        seeds, triggers = [], []
        for fid in flaws:
            fl = self._flaw(char, fid)
            seeds.append(fl["motive_seed"])
            triggers.append(fl["trigger_vs_victim"])
        motive = (
            L["motive_head"].format(c=culprit, v=self.victim)
            + L["motive_and"].join(seeds)
            + L["motive_mid"] + L["motive_also"].join(triggers)
            + L["motive_tail"]
        )

        # 4) Culprit's secret timeline for this method.
        secret = self._secret_timeline(culprit, method["id"])

        gt = GroundTruth(
            is_murder=True, culprit=culprit, active_flaws=flaws,
            method=method, motive=motive, secret_timeline=secret,
        )
        self._distribute_clues(gt)
        return gt

    def _roll_true_accident(self, sp: dict) -> GroundTruth:
        L = self.L
        v, e = self.victim, (self.concealer or "")
        gt = GroundTruth(
            is_murder=False, culprit=None,
            active_flaws=["guilty_concealment"], method=None,
            motive=L["accident_motive"].format(v=v, e=e),
            secret_timeline=[s.format(v=v, e=e)
                             for s in L["accident_timeline"]],
        )
        self._distribute_clues(gt)
        return gt

    def _secret_timeline(self, culprit: str, method_id: str) -> list[str]:
        tmpls = self.L["secret_timeline"]
        chosen = tmpls.get(method_id, tmpls["default"])
        return [s.format(c=culprit, v=self.victim) for s in chosen]

    def _distribute_clues(self, gt: GroundTruth) -> None:
        """Give 1-2 innocent NPCs a sighting clue tied to the method.

        Fairness rule: every playthrough plants at least two
        independent threads pointing at the truth.
        """
        names = [c["character"] for c in self.case["characters"]]
        innocents = [n for n in names if n != gt.culprit]
        if gt.is_murder:
            templates = self.L["method_sightings"][gt.method["id"]]
        else:
            templates = self.L["accident_clues"]
            # In the accident roll the concealer is the culprit-of-record.
            innocents = [n for n in innocents if n != self.concealer]
        witnesses = self.rng.sample(innocents, k=min(2, len(innocents)))
        for w, tmpl in zip(witnesses, templates):
            clue = tmpl.format(c=gt.culprit or "", v=self.victim,
                               e=self.concealer or "")
            gt.distributed_clues.setdefault(w, []).append(clue)


# ----------------------------------------------------------------
# Judge job #0: the judge LLM chooses the killer and their motive.
#
# The judge picks from the authored valid (culprit, flaws) combos, so
# every choice stays coherent with method access and clue trails. On any
# failure (no API key, bad JSON, off-menu pick) it falls back to the
# deterministic weighted code pick, so the game always starts.
# ----------------------------------------------------------------
def _match_name(token: str, names: list[str]) -> str | None:
    token = (token or "").strip().lower()
    if not token:
        return None
    for n in names:                       # exact full-name match
        if token == n.lower():
            return n
    for n in names:                       # substring match (en + zh)
        if token in n.lower() or n.lower() in token:
            return n
    return None


def judge_select_culprit(case: dict, judge: LLMProvider,
                         rng: random.Random, lang: str = "en") -> dict:
    """Ask the judge to pick (culprit, flaws). Returns a dict with
    culprit/flaws plus debug fields (source, rationale, motive_seeds,
    triggers); always valid even on failure. Logging is the caller's job."""
    L = t(lang)
    opts = culprit_options(case)
    names = [c["character"] for c in case["characters"]]
    v = victim_name(case)
    flaw_text = {c["character"]: {f["id"]: f for f in c["flaws"]}
                 for c in case["characters"]}

    # Present the menu of valid combos with their authored motive seeds.
    sep = "；" if lang == "zh" else "; "
    menu_lines = []
    for i, (cul, flaws, strength, _w) in enumerate(opts, 1):
        seeds = sep.join(flaw_text[cul][f]["motive_seed"] for f in flaws)
        menu_lines.append(L["select_menu"].format(
            i=i, c=cul, flaws=flaws, strength=strength, seeds=seeds))
    system = L["select_system"].format(v=v)
    user = L["select_user"].format(menu="\n".join(menu_lines))

    result = None
    try:
        raw = judge.chat(system, [{"role": "user", "content": user}],
                         max_tokens=250)
        data = _extract_json(raw)
        assert data
        cul = _match_name(str(data.get("culprit", "")), names)
        flaws = [str(f).strip() for f in data.get("flaws", [])]
        match = next((o for o in opts
                      if o[0] == cul and set(o[1]) == set(flaws)), None)
        assert match
        result = {"culprit": match[0], "flaws": match[1],
                  "source": "llm",
                  "rationale": str(data.get("rationale", ""))[:300]}
    except Exception:
        cul, flaws, _strength, _w = rng.choices(
            opts, weights=[o[3] for o in opts], k=1)[0]
        result = {"culprit": cul, "flaws": flaws, "source": "fallback",
                  "rationale": L["select_fallback_rationale"]}

    by_id = flaw_text[result["culprit"]]
    result["motive_seeds"] = [by_id[f]["motive_seed"]
                              for f in result["flaws"] if f in by_id]
    result["triggers"] = [by_id[f]["trigger_vs_victim"]
                          for f in result["flaws"] if f in by_id]
    return result


# ----------------------------------------------------------------
# Knowledge packets -> NPC system prompts
# ----------------------------------------------------------------
def build_npc_system_prompt(case: dict, char: dict, gt: GroundTruth,
                            deeds: list[str] | None = None,
                            lang: str = "en") -> str:
    L = t(lang)
    P = L["npc"]
    name = char["character"]
    is_culprit = (gt.culprit == name)
    sc = case["scenario"]
    v = victim_name(case)

    lines = [
        P["intro"].format(name=name, v=v),
        "",
        P["setting"].format(setting=sc["setting"]),
        P["what_happened"].format(accident=sc["the_accident"]),
        "",
        P["who_you_are"].format(role=char["role"]),
        P["public_face"].format(public=char["public_story"]),
        P["private_truth"].format(private=char["private_story"]),
        "",
        P["fixed_events"],
    ]
    for beat in case["timeline_skeleton"]:
        lines.append(P["beat"].format(time=beat["time"], beat=beat["beat"]))
    lines.append("")
    lines.append(P["know_others"])
    for k in char.get("knows_about_others", []):
        lines.append(P["bullet"].format(x=k))
    for clue in gt.distributed_clues.get(name, []):
        lines.append(P["bullet"].format(x=clue))
    lines.append("")
    lines.append(P["psychology"])
    for fl in char["flaws"]:
        lines.append(P["flaw_desc"].format(
            description=fl["description"], tells=fl["behavioral_tells"]))
        lines.append(P["flaw_tonight"].format(trigger=fl["trigger_vs_victim"]))
        lines.append(P["flaw_lie"].format(lie=fl["lie_tendency"]))
    lines.append("")

    if deeds:
        lines.append(P["deeds_header"])
        lines += [P["bullet"].format(x=d) for d in deeds]
        lines.append("")

    if is_culprit:
        lines += [
            P["killer_header"],
            P["killer_truth"].format(motive=gt.motive),
            P["killer_did"],
            *[P["bullet"].format(x=s) for s in gt.secret_timeline],
            "",
            P["killer_rules_head"],
            *P["killer_rules"],
        ]
    else:
        lines += [
            P["innocent_header"],
            *[s.format(v=v) for s in P["innocent_lines"]],
        ]

    lines += ["", P["style_header"], *P["style_rules"]]
    return "\n".join(lines)


# ----------------------------------------------------------------
# Referee v0 — cheap output check before the player sees a reply.
# ----------------------------------------------------------------
def referee_check(reply: str, npc_name: str, gt: GroundTruth,
                  lang: str = "en") -> str | None:
    """Return a regeneration hint if the reply is invalid, else None."""
    L = t(lang)
    ref = L["referee"]
    if re.search(ref["leak_rx"], reply, re.IGNORECASE):
        return ref["leak_hint"]
    if gt.culprit != npc_name and re.search(
            ref["confession_rx"], reply, re.IGNORECASE):
        return ref["confess_hint"]
    return None


# ----------------------------------------------------------------
# Accusation judge — the one-shot endgame.
# ----------------------------------------------------------------
import json as _json


def _extract_json(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return _json.loads(text[start:end + 1])
    except _json.JSONDecodeError:
        return None


def _who_verdict(gt: GroundTruth, accused: str, lang: str = "en") -> bool:
    a = accused.strip().lower()
    if gt.is_murder:
        cul = gt.culprit.lower()
        # Bidirectional: the player may type a full or partial name.
        return cul in a or (len(a) >= 2 and a in cul) or \
            any(p in a for p in cul.split() if len(p) > 2)
    return any(w in a for w in t(lang)["who_accident_keywords"])


def _keyword_motive_fallback(gt: GroundTruth, reasoning: str,
                             lang: str = "en") -> dict:
    words = []
    for fid in gt.active_flaws:
        words += [w for w in fid.split("_") if len(w) > 3]
    hits = sum(1 for w in words if w in reasoning.lower())
    verdict = "correct" if hits >= 2 else "partial" if hits == 1 else "wrong"
    return {"motive": verdict, "method": "not_stated",
            "comment": t(lang)["keyword_comment"]}


def _grade_with_llm(gt: GroundTruth, reasoning: str, how: str,
                    provider: LLMProvider, lang: str = "en") -> dict:
    L = t(lang)
    g = L["grade_truth"]
    truth = [
        g["culprit"].format(c=gt.culprit or g["nobody"]),
        g["motive"].format(motive=gt.motive),
        g["flaws"].format(flaws=", ".join(gt.active_flaws)),
        g["method"].format(method=(gt.method or {}).get(
            "description", g["method_accident"])),
        g["secret"].format(secret=" | ".join(gt.secret_timeline)),
    ]
    system = L["grade_system"]
    user = L["grade_user"].format(
        truth="\n".join(truth), why=reasoning,
        how=how or L["grade_not_stated"])
    try:
        raw = provider.chat(system, [{"role": "user", "content": user}],
                            max_tokens=250)
        verdict = _extract_json(raw)
        if verdict and verdict.get("motive") in ("correct", "partial", "wrong"):
            verdict.setdefault("method", "not_stated")
            return verdict
    except Exception:
        pass
    return _keyword_motive_fallback(gt, reasoning, lang)


def judge_accusation(case: dict, gt: GroundTruth, accused: str,
                     reasoning: str, how: str,
                     provider: LLMProvider, lang: str = "en") -> str:
    L = t(lang)
    v = victim_name(case)
    who_ok = _who_verdict(gt, accused, lang)
    grade = _grade_with_llm(gt, reasoning, how, provider, lang)
    motive, method = grade["motive"], grade.get("method", "not_stated")

    score = (50 if who_ok else 0)
    score += {"correct": 35, "partial": 18}.get(motive, 0)
    score += {"correct": 15, "partial": 8}.get(method, 0)
    # WHO gates the top ratings: a wrong accusation must never read as
    # "the right arrest" just because the why/how fit the real killer.
    if not who_ok:
        score = min(score, 49)
    rating = next(label for cut, label in L["ratings"] if score >= cut)
    rating = rating.format(v=v)

    truth_label = (gt.culprit if gt.is_murder else
                   L["truth_label_accident"].format(e=concealer_name(case) or ""))

    who_word = L["who_correct"] if who_ok else L["who_wrong"]
    glabels = L["grade_labels"]
    card = [
        rating,
        L["card_score"].format(score=score),
        L["card_who"].format(ok=who_word, accused=accused.strip(),
                             truth=truth_label),
        L["card_why"].format(motive=glabels.get(motive, motive.upper()),
                             comment=grade.get("comment", "")),
        L["card_how"].format(method=glabels.get(method,
                                                method.upper().replace("_", " "))),
    ]

    if who_ok:
        consequence = L["consequence_ok"]
    elif gt.is_murder:
        consequence = L["consequence_wrong_murder"].format(
            accused=accused, c=gt.culprit)
    else:
        consequence = L["consequence_accident"]

    epilogue_prompt = L["epilogue_prompt"].format(
        motive=gt.motive, secret=" ".join(gt.secret_timeline),
        consequence=consequence)
    try:
        epilogue = provider.chat(
            L["narrator_system"],
            [{"role": "user", "content": epilogue_prompt}],
            max_tokens=420,
        )
    except Exception as e:  # pragma: no cover
        epilogue = L["epilogue_error"].format(e=e)

    return "\n".join(card) + "\n\n" + epilogue


# ----------------------------------------------------------------
# Acts: item visibility depends on the rolled plot.
# ----------------------------------------------------------------
def item_present(item: dict, gt: GroundTruth) -> bool:
    cond = item.get("condition") or {}
    if "method" in cond:
        return gt.is_murder and gt.method["id"] == cond["method"]
    if "accident" in cond:
        return (not gt.is_murder) == bool(cond["accident"])
    if "culprit" in cond:
        return gt.culprit == cond["culprit"]
    return True


# ----------------------------------------------------------------
# Judge job #1: write true deeds for EVERY NPC (anti-confabulation).
# LLM-written when the judge provider cooperates; deterministic
# fallback otherwise, so the game always starts.
# ----------------------------------------------------------------
def default_deeds(case: dict, gt: GroundTruth,
                  rng: random.Random, lang: str = "en") -> dict[str, list[str]]:
    L = t(lang)
    v = victim_name(case)
    places, doings = L["deeds_places"], L["deeds_doings"]
    reasons = [r.format(v=v) for r in L["deeds_reasons"]]
    deeds: dict[str, list[str]] = {}
    for ch in case["characters"]:
        name = ch["character"]
        if name == gt.culprit:
            deeds[name] = []  # culprit's deeds live in the secret block
            continue
        p1, p2 = rng.sample(places, 2)
        deeds[name] = [
            L["deeds_line1"].format(p1=p1, doing=rng.choice(doings)),
            L["deeds_line2"].format(mm=rng.choice(L["deeds_mm"]),
                                    reason=rng.choice(reasons)),
            L["deeds_line3"].format(p2=p2),
        ]
    return deeds


def judge_generate_deeds(case: dict, gt: GroundTruth,
                         judge: LLMProvider,
                         rng: random.Random, lang: str = "en") -> dict[str, list[str]]:
    L = t(lang)
    names = [c["character"] for c in case["characters"]]
    system = L["deeds_system"]
    user = L["deeds_user"].format(
        names=", ".join(names), motive=gt.motive,
        secret=" | ".join(gt.secret_timeline))
    try:
        raw = judge.chat(system, [{"role": "user", "content": user}],
                         max_tokens=900)
        data = _extract_json(raw)
        assert data and set(data.keys()) == set(names)
        for n, lines in data.items():
            assert isinstance(lines, list)
            if n != gt.culprit:
                assert 1 <= len(lines) <= 4
                joined = " ".join(map(str, lines))
                low = joined.lower()
                assert ("i killed" not in low and "我杀" not in joined
                        and "是我干的" not in joined)
                if gt.culprit:
                    first = gt.culprit.split()[0]
                    assert not re.search(
                        re.escape(first) +
                        r"\s*(killed|pushed|murdered|杀|推|谋杀|害死)",
                        joined, re.IGNORECASE)
        data[gt.culprit] = [] if gt.culprit else data.get(gt.culprit, [])
        return {n: list(map(str, v)) for n, v in data.items()}
    except Exception:
        return default_deeds(case, gt, rng, lang)   # the safety net


# ----------------------------------------------------------------
# Judge job #2: boundary gossip. Partial truths about others' deeds
# plus word of what the player has been asking - dealt between acts.
# ----------------------------------------------------------------
def deal_boundary_gossip(case: dict, gt: GroundTruth, act_no: int,
                         deeds: dict[str, list[str]],
                         questions_log: dict[str, list[str]],
                         judge: LLMProvider,
                         rng: random.Random, lang: str = "en") -> dict[str, list[str]]:
    L = t(lang)
    names = [c["character"] for c in case["characters"]]
    system = L["gossip_system"]
    qsep = "；" if lang == "zh" else "; "
    qsum = "\n".join(
        L["gossip_asked"].format(n=n, qs=qsep.join(q[:70] for q in qs[-3:]))
        for n, qs in questions_log.items() if qs) or L["gossip_none"]
    deeds_str = "\n".join(
        f"{n}: {' | '.join(d) if d else L['gossip_secret']}"
        for n, d in deeds.items())
    user = L["gossip_user"].format(act=act_no, deeds=deeds_str, qsum=qsum)
    bad = None
    if gt.culprit:
        bad = re.compile(
            re.escape(gt.culprit) +
            r"\s*(killed|pushed|murdered|did it|杀|推|谋杀|害死|干的)",
            re.IGNORECASE)
    try:
        raw = judge.chat(system, [{"role": "user", "content": user}],
                         max_tokens=700)
        data = _extract_json(raw)
        assert data
        out = {}
        for n in names:
            lines = [str(s) for s in data.get(n, [])][:2]
            lines = [s for s in lines if not (bad and bad.search(s))]
            if lines:
                out[n] = lines
        assert out
        return out
    except Exception:
        out = {}
        for n in names:
            other = rng.choice([m for m in names if m != n])
            lines = [L["gossip_fallback_slip"].format(other=other)]
            asked = [m for m in names
                     if m != n and questions_log.get(m)]
            if asked:
                tgt = rng.choice(asked)
                q = rng.choice(questions_log[tgt])[:60]
                lines.append(L["gossip_fallback_press"].format(t=tgt, q=q))
            out[n] = lines[:2]
        return out
