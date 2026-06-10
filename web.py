"""
web.py — an interactive browser UI for Shifting Truth.

A thin Flask layer over the existing engine: it holds one game session
per browser (sid), exposes the same actions the CLI has — search a spot,
question a suspect, show evidence, end a phase, accuse — as JSON, and
serves a single-page app (webui/index.html).

Run:
    source /venv/main/bin/activate
    pip install flask pyyaml
    python web.py                 # serves http://127.0.0.1:17080

No API key? It still runs: providers that can't initialize (e.g. the
Anthropic provider with no key) fall back to the mock provider, so the
UI is fully clickable — the suspects just give canned replies until a
real model is configured.
"""

from __future__ import annotations
import os
import uuid

import yaml
from flask import Flask, request, jsonify, send_from_directory

from providers import provider_from_config, MockProvider
from i18n import t, normalize_lang
from engine import (
    load_case, Director, build_npc_system_prompt, referee_check,
    judge_accusation, item_present, judge_generate_deeds,
    deal_boundary_gossip, judge_select_culprit, victim_name,
)

HERE = os.path.dirname(os.path.abspath(__file__))
AVATAR_COLORS = ["#b4654a", "#4a6fa5", "#5a8a5a", "#8a5a8a", "#a5904a"]
AVATAR_EMOJI = ["💼", "🔑", "🔬", "🥃", "📋"]
# Real-time countdown: how many real seconds each in-fiction minute of a
# phase budget is worth. The client ticks this down live and auto-ends
# the phase at zero (~20 min total game at the default).
SECONDS_PER_MIN = 20


def _safe_provider(cfg_block):
    """Build a provider, falling back to mock if it can't initialize."""
    try:
        p = provider_from_config(cfg_block)
        return p, isinstance(p, MockProvider)
    except Exception:
        return MockProvider(), True


class WebGame:
    """One playable session, driven by API calls instead of stdin."""

    def __init__(self, cfg: dict, lang: str):
        self.lang = normalize_lang(lang)
        self.L = t(self.lang)
        self.costs = cfg.get("costs", {})
        gcfg = cfg["game"]

        self.npc_llm, mock_npc = _safe_provider(cfg["agents"]["npc"])
        self.judge_llm, _ = _safe_provider(cfg["agents"]["judge"])
        self.mock = mock_npc

        cases = gcfg.get("cases") or {}
        case_path = (cases.get(self.lang) or gcfg.get(f"case_{self.lang}")
                     or gcfg.get("case", "case.yaml"))
        self.case = load_case(os.path.join(HERE, case_path))
        self.victim = victim_name(self.case)

        self.director = Director(self.case, seed=gcfg.get("seed"), lang=self.lang)
        self.gt = self.director.roll(selector=self._select)
        self.names = [c["character"] for c in self.case["characters"]]
        self.deeds = judge_generate_deeds(
            self.case, self.gt, self.judge_llm, self.director.rng, self.lang)
        self.base_prompt = {
            c["character"]: build_npc_system_prompt(
                self.case, c, self.gt,
                deeds=self.deeds.get(c["character"]), lang=self.lang)
            for c in self.case["characters"]
        }
        self.extras = {n: [] for n in self.names}
        self.histories = {n: [] for n in self.names}
        self.chat = {n: [] for n in self.names}       # display log {who,text}
        self.questions_log = {n: [] for n in self.names}
        self.evidence = []                             # {name, found_text}
        self._have = set()
        self.searched = set()
        self.found = {}                                # spot_id -> [items]
        self.verdict = None

        # Flatten acts -> phases into one timeline.
        self.plan = []
        for act in self.case["acts"]:
            for phi, phase in enumerate(act["phases"]):
                self.plan.append({
                    "act": act, "phase": phase,
                    "last_of_act": phi == len(act["phases"]) - 1,
                })
        self.pi = 0

        self.avatars = {
            n: {"initial": n[0],
                "color": AVATAR_COLORS[i % len(AVATAR_COLORS)],
                "emoji": AVATAR_EMOJI[i % len(AVATAR_EMOJI)]}
            for i, n in enumerate(self.names)
        }

    # ---- director selector (judge picks culprit) --------------------
    def _select(self):
        sel = judge_select_culprit(
            self.case, self.judge_llm, self.director.rng, self.lang)
        return sel["culprit"], sel["flaws"]

    # ---- NPC plumbing ----------------------------------------------
    def _system_for(self, name):
        sysp = self.base_prompt[name]
        if self.extras[name]:
            sysp += ("\n\n" + self.L["ui"]["new_memories"] + "\n" +
                     "\n".join(f"  - {x}" for x in self.extras[name]))
        return sysp

    def _npc_reply(self, name, user_text):
        self.histories[name].append({"role": "user", "content": user_text})
        reply = self.npc_llm.chat(self._system_for(name), self.histories[name])
        hint = referee_check(reply, name, self.gt, self.lang)
        if hint:
            self.histories[name].append({"role": "assistant", "content": reply})
            self.histories[name].append(
                {"role": "user",
                 "content": f"[OUT OF CHARACTER CORRECTION: {hint}]"})
            reply = self.npc_llm.chat(self._system_for(name), self.histories[name])
        self.histories[name].append({"role": "assistant", "content": reply})
        return reply

    # ---- phase helpers ---------------------------------------------
    def _cur(self):
        return self.plan[self.pi] if self.pi < len(self.plan) else None

    def phase_kind(self):
        cur = self._cur()
        if cur:
            return cur["phase"]["type"]          # "search" | "talk"
        return "over" if self.verdict else "accuse"

    # ---- actions ----------------------------------------------------
    def search(self, spot_id):
        cur = self._cur()
        if not cur or cur["phase"]["type"] != "search":
            return {"error": "not_search"}
        spot = next((s for s in cur["act"]["spots"] if s["id"] == spot_id), None)
        if not spot:
            return {"error": "bad_spot"}
        if spot_id in self.searched:
            return {"error": self.L["web"]["already_searched"]}
        self.searched.add(spot_id)
        found = [i for i in spot.get("items", []) if item_present(i, self.gt)]
        self.found[spot_id] = [{"name": i["name"], "found_text": i["found_text"]}
                               for i in found]
        for i in found:
            if i["name"] not in self._have:
                self._have.add(i["name"])
                self.evidence.append({"name": i["name"],
                                      "found_text": i["found_text"]})
        return {"ok": True}

    def talk(self, name, text):
        cur = self._cur()
        if not cur or cur["phase"]["type"] != "talk":
            return {"error": "not_talk"}
        if name not in self.names:
            return {"error": "bad_name"}
        if not text.strip():
            return {"error": "empty"}
        self.questions_log[name].append(text)
        self.chat[name].append({"who": "you", "text": text})
        reply = self._npc_reply(name, text)
        self.chat[name].append({"who": "npc", "text": reply})
        return {"ok": True, "reply": reply}

    def show(self, name, item_name):
        cur = self._cur()
        if not cur or cur["phase"]["type"] != "talk":
            return {"error": "not_talk"}
        if name not in self.names:
            return {"error": "bad_name"}
        item = next((e for e in self.evidence if e["name"] == item_name), None)
        if not item:
            return {"error": "no_item"}
        msg = self.L["ui"]["evidence_present"].format(
            name=item["name"], text=item["found_text"])
        self.chat[name].append({"who": "you", "text": "▸ " + item["name"]})
        reply = self._npc_reply(name, msg)
        self.chat[name].append({"who": "npc", "text": reply})
        return {"ok": True, "reply": reply}

    def next_phase(self):
        gossip = None
        cur = self._cur()
        if cur:
            was_last = cur["last_of_act"]
            self.pi += 1
            if was_last and self.pi < len(self.plan):
                gossip = self._boundary(cur["act"]["act"])
        return gossip

    def _boundary(self, act_no):
        gossip = deal_boundary_gossip(
            self.case, self.gt, act_no, self.deeds,
            self.questions_log, self.judge_llm, self.director.rng, self.lang)
        prefix = self.L["ui"]["hearsay_prefix"]
        for n, lines in gossip.items():
            self.extras[n].extend(f"{prefix}{s}" for s in lines)
        return [{"name": n, "avatar": self.avatars[n], "lines": lines}
                for n, lines in gossip.items()]

    def accuse(self, who, why, how):
        if self.pi < len(self.plan):
            return {"error": "not_finished"}
        self.verdict = judge_accusation(
            self.case, self.gt, who, why, how, self.judge_llm, self.lang)
        return {"ok": True}

    # ---- serialization ---------------------------------------------
    def to_dict(self):
        cur = self._cur()
        kind = self.phase_kind()
        act = cur["act"] if cur else None
        phase_seconds = (int(cur["phase"]["time"]) * SECONDS_PER_MIN
                         if cur else 0)
        spots = []
        if act:
            for s in act["spots"]:
                spots.append({
                    "id": s["id"], "name": s["name"],
                    "empty_text": s.get("empty_text", self.L["web"]["found_nothing"]),
                    "searched": s["id"] in self.searched,
                    "found": self.found.get(s["id"], []),
                })
        cast = []
        for n in self.names:
            ch = next(c for c in self.case["characters"] if c["character"] == n)
            cast.append({
                "name": n, "role": ch["role"], "avatar": self.avatars[n],
                "chat": self.chat[n],
                "statements": sum(1 for m in self.chat[n] if m["who"] == "npc"),
            })
        return {
            "lang": self.lang,
            "web": self.L["web"],
            "mock": self.mock,
            "victim": self.victim,
            "scenario": {
                "title": self.case["scenario"]["title"],
                "setting": self.case["scenario"]["setting"],
                "the_accident": self.case["scenario"]["the_accident"],
            },
            "phase_kind": kind,
            "act_num": act["act"] if act else None,
            "act_title": act["title"] if act else None,
            "act_intro": act.get("scene_intro", "") if act else "",
            "phase_index": self.pi + 1,
            "phase_total": len(self.plan),
            "phase_seconds": phase_seconds,
            "spots": spots,
            "cast": cast,
            "evidence": self.evidence,
            "verdict": self.verdict,
        }


# ----------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------
app = Flask(__name__, static_folder=None)
GAMES: dict[str, WebGame] = {}


def _load_cfg():
    with open(os.path.join(HERE, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _game():
    sid = request.headers.get("X-Session")
    if not sid:
        sid = (request.get_json(silent=True) or {}).get("sid")
    return sid, GAMES.get(sid)


@app.get("/")
def index():
    return send_from_directory(os.path.join(HERE, "webui"), "index.html")


@app.get("/<path:path>")
def static_files(path):
    return send_from_directory(os.path.join(HERE, "webui"), path)


@app.post("/api/new")
def api_new():
    data = request.get_json(silent=True) or {}
    lang = normalize_lang(data.get("lang", "en"))
    sid = uuid.uuid4().hex
    GAMES[sid] = WebGame(_load_cfg(), lang)
    state = GAMES[sid].to_dict()
    state["sid"] = sid
    return jsonify(state)


def _respond(g, extra=None):
    state = g.to_dict()
    if extra:
        state.update(extra)
    return jsonify(state)


@app.post("/api/search")
def api_search():
    sid, g = _game()
    if not g:
        return jsonify({"error": "no_session"}), 404
    r = g.search((request.get_json(silent=True) or {}).get("spot_id", ""))
    return _respond(g, {"action": r})


@app.post("/api/talk")
def api_talk():
    sid, g = _game()
    if not g:
        return jsonify({"error": "no_session"}), 404
    d = request.get_json(silent=True) or {}
    r = g.talk(d.get("name", ""), d.get("text", ""))
    return _respond(g, {"action": r})


@app.post("/api/show")
def api_show():
    sid, g = _game()
    if not g:
        return jsonify({"error": "no_session"}), 404
    d = request.get_json(silent=True) or {}
    r = g.show(d.get("name", ""), d.get("item", ""))
    return _respond(g, {"action": r})


@app.post("/api/next")
def api_next():
    sid, g = _game()
    if not g:
        return jsonify({"error": "no_session"}), 404
    gossip = g.next_phase()
    return _respond(g, {"gossip": gossip})


@app.post("/api/accuse")
def api_accuse():
    sid, g = _game()
    if not g:
        return jsonify({"error": "no_session"}), 404
    d = request.get_json(silent=True) or {}
    r = g.accuse(d.get("who", ""), d.get("why", ""), d.get("how", ""))
    return _respond(g, {"action": r})


if __name__ == "__main__":
    port = int(os.environ.get("ST_WEB_PORT", "17080"))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
