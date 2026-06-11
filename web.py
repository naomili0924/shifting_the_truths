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
import datetime
import json
import os
import threading
import uuid

import yaml
from flask import Flask, request, jsonify, send_from_directory

import imagegen
import rooms
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

# The five fixed faces are the same people in every language, so portraits are
# painted from the canonical (English) character descriptions and keyed by cast
# index — one stable, globally-cached face set regardless of the game language.
try:
    _ENG_CHARS = load_case(os.path.join(HERE, "case.yaml")).get("characters", [])
except Exception:
    _ENG_CHARS = []


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
        self.names = [c["character"] for c in self.case["characters"]]
        self.images_cfg = cfg.get("images") or {}

        # The heavy work — the judge's culprit pick, the per-NPC deeds, the system
        # prompts and the room manifests — used to run here, blocking /api/new for
        # a minute. It now happens in a background thread so the request returns
        # instantly and the setup hides behind the scenario intro. Placeholders
        # keep to_dict() safe before setup completes.
        self.ready = False
        self.gt = None
        self.deeds = {}
        self.base_prompt = {}
        self.room_manifests = {}

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

        self._img_lock = threading.Lock()
        self.backdrops: dict[int, str | None] = {}   # act_num -> filename
        self.faces: dict[str, str | None] = {}        # npc name -> filename
        self.item_images: dict[str, str | None] = {}  # item name -> filename
        self.images_done = False
        gen = imagegen.instance()
        self.art_enabled = bool(gen and gen.available())

        # Everything heavy runs off the request path, behind the intro.
        threading.Thread(target=self._setup, daemon=True).start()

    # ---- background setup (judge work) then painting ----------------
    def _setup(self):
        """Roll the plot, write deeds + prompts + room manifests, then paint.

        Runs in a daemon thread; the client shows the scenario intro and polls
        /api/state until ``setup_done`` (and the first backdrop) is ready.
        """
        # 1) The culprit roll is the only judge step the first (search) phase needs
        #    — item_present() depends on it — so do it first and mark the game
        #    playable as soon as it (and the instant deterministic manifest) is set.
        try:
            self.gt = self.director.roll(selector=self._select)
        except Exception:
            self.gt = self.director.roll()
        use_judge = bool(self.images_cfg.get("judge_layout"))
        self.room_manifests = {
            act["act"]: (
                rooms.judge_room_layout(self.case, act, self.judge_llm, self.lang)
                if use_judge else rooms.default_manifest(self.case, act)
            )
            for act in self.case["acts"]
        }
        self.ready = True

        # 2) Paint the rooms/faces/items (backdrop the intro waits for).
        if self.art_enabled:
            self._paint_all()

        # 3) Deeds + NPC system prompts are only needed in the talk phase (after
        #    the timed search phase), so generate them after the game is playable.
        try:
            self.deeds = judge_generate_deeds(
                self.case, self.gt, self.judge_llm, self.director.rng, self.lang)
        except Exception:
            self.deeds = {}
        self.base_prompt = {
            c["character"]: build_npc_system_prompt(
                self.case, c, self.gt,
                deeds=self.deeds.get(c["character"]), lang=self.lang)
            for c in self.case["characters"]
        }

    # ---- background art painting -----------------------------------
    def _paint_all(self):
        gen = imagegen.instance()
        if not gen:
            return
        # Backdrops, in play order so the first room is ready first.
        for act in self.case["acts"]:
            manifest = self.room_manifests.get(act["act"], {})
            fn = gen.generate(manifest.get("prompt", "")) if manifest else None
            with self._img_lock:
                self.backdrops[act["act"]] = fn
        # The five fixed faces, keyed by cast index to the canonical descriptions.
        for i, name in enumerate(self.names):
            src = _ENG_CHARS[i] if i < len(_ENG_CHARS) else None
            fn = gen.generate(rooms.portrait_prompt(src)) if src else None
            with self._img_lock:
                self.faces[name] = fn
        # Items that are actually present in this rolled plot (manifest first).
        for act in self.case["acts"]:
            for spot in act.get("spots", []):
                for item in spot.get("items", []):
                    if not item_present(item, self.gt):
                        continue
                    fn = gen.generate(rooms.item_prompt(item))
                    with self._img_lock:
                        self.item_images[item["name"]] = fn
        with self._img_lock:
            self.images_done = True

    def _image_state(self):
        cur = self._cur()
        act_num = cur["act"]["act"] if cur else None
        with self._img_lock:
            backdrop = self.backdrops.get(act_num) if act_num else None
            faces = dict(self.faces)
            done = self.images_done
        manifest = self.room_manifests.get(act_num, {}) if act_num else {}
        return {
            "art_enabled": self.art_enabled,
            "images_done": done,
            "backdrop": f"/assets/cache/{backdrop}" if backdrop else None,
            "faces": {n: (f"/assets/cache/{fn}" if fn else None)
                      for n, fn in faces.items()},
            "chips": manifest.get("spots", {}),
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
        if not self.ready:
            return {"error": "loading"}
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
        if not self.ready:
            return {"error": "loading"}
        cur = self._cur()
        if not cur or cur["phase"]["type"] != "talk":
            return {"error": "not_talk"}
        if name not in self.names:
            return {"error": "bad_name"}
        if name not in self.base_prompt:
            return {"ok": True,
                    "reply": self.L["web"].get("suspects_arriving",
                                               "The suspects are still gathering — try again in a moment.")}
        if not text.strip():
            return {"error": "empty"}
        self.questions_log[name].append(text)
        self.chat[name].append({"who": "you", "text": text})
        reply = self._npc_reply(name, text)
        self.chat[name].append({"who": "npc", "text": reply})
        return {"ok": True, "reply": reply}

    def show(self, name, item_name):
        if not self.ready:
            return {"error": "loading"}
        cur = self._cur()
        if not cur or cur["phase"]["type"] != "talk":
            return {"error": "not_talk"}
        if name not in self.names:
            return {"error": "bad_name"}
        if name not in self.base_prompt:
            return {"ok": True,
                    "reply": self.L["web"].get("suspects_arriving",
                                               "The suspects are still gathering — try again in a moment.")}
        item = next((e for e in self.evidence if e["name"] == item_name), None)
        if not item:
            return {"error": "no_item"}
        msg = self.L["ui"]["evidence_present"].format(
            name=item["name"], text=item["found_text"])
        self.chat[name].append({"who": "you", "text": "▸ " + item["name"]})
        reply = self._npc_reply(name, msg)
        self.chat[name].append({"who": "npc", "text": reply})
        return {"ok": True, "reply": reply}

    def hint(self):
        """Last-resort help: return up to (2 - found) UNSEARCHED spots in the
        current search that actually contain a present item, so the player can
        reach at least two clues. Reveals only the minimum needed."""
        if not self.ready:
            return {"spots": []}
        cur = self._cur()
        if not cur or cur["phase"]["type"] != "search":
            return {"spots": []}
        act = cur["act"]
        found_ct = sum(len(self.found.get(s["id"], [])) for s in act["spots"])
        need = max(0, 2 - found_ct)
        if need <= 0:
            return {"spots": []}
        targets = []
        for s in act["spots"]:
            if s["id"] in self.searched:
                continue
            if any(item_present(i, self.gt) for i in s.get("items", [])):
                targets.append(s["id"])
            if len(targets) >= need:
                break
        return {"spots": targets}

    def next_phase(self):
        if not self.ready:
            return None
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
        if not self.ready:
            return {"error": "loading"}
        if self.pi < len(self.plan):
            return {"error": "not_finished"}
        self.verdict = judge_accusation(
            self.case, self.gt, who, why, how, self.judge_llm, self.lang)
        return {"ok": True}

    def rate(self, stars, comment):
        """Record a player's plot rating with the hidden ground truth, so
        the designer can see which generated plots land well."""
        try:
            stars = max(0, min(5, int(stars)))
        except (TypeError, ValueError):
            stars = 0
        rec = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "lang": self.lang, "stars": stars,
            "comment": str(comment or "")[:1000],
            "is_murder": self.gt.is_murder, "culprit": self.gt.culprit,
            "flaws": self.gt.active_flaws,
            "method": self.gt.method["id"] if self.gt.method else None,
            "motive": self.gt.motive,
            "verdict_head": (self.verdict or "").splitlines()[0] if self.verdict else None,
        }
        path = os.path.join(HERE, "logs", "ratings.jsonl")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return {"ok": True, "stars": stars}

    # ---- serialization ---------------------------------------------
    def to_dict(self):
        cur = self._cur()
        kind = self.phase_kind()
        act = cur["act"] if cur else None
        phase_seconds = (int(cur["phase"]["time"]) * SECONDS_PER_MIN
                         if cur else 0)
        def _with_img(items):
            out = []
            for it in items:
                fn = self.item_images.get(it["name"])
                out.append({**it,
                            "image": f"/assets/cache/{fn}" if fn else None})
            return out

        spots = []
        if act:
            for s in act["spots"]:
                spots.append({
                    "id": s["id"], "name": s["name"],
                    "empty_text": s.get("empty_text", self.L["web"]["found_nothing"]),
                    "searched": s["id"] in self.searched,
                    "found": _with_img(self.found.get(s["id"], [])),
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
            "setup_done": self.ready,
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
            "evidence": _with_img(self.evidence),
            "verdict": self.verdict,
            "images": self._image_state(),
        }


# ----------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------
app = Flask(__name__, static_folder=None)
GAMES: dict[str, WebGame] = {}


def _load_cfg():
    with open(os.path.join(HERE, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Configure the (optional) image generator once at import, from config.yaml.
try:
    _CFG = _load_cfg()
    _imgcfg = dict(_CFG.get("images") or {})
    if _imgcfg.get("cache_dir") and not os.path.isabs(_imgcfg["cache_dir"]):
        _imgcfg["cache_dir"] = os.path.join(HERE, _imgcfg["cache_dir"])
    imagegen.configure(_imgcfg)
except Exception:
    imagegen.configure({"enabled": False})


def _game():
    sid = request.headers.get("X-Session")
    if not sid:
        sid = (request.get_json(silent=True) or {}).get("sid")
    return sid, GAMES.get(sid)


@app.get("/")
def index():
    # Phaser point-and-click UI with generated art is the primary experience.
    return send_from_directory(os.path.join(HERE, "webui"), "game.html")


@app.get("/classic")
def classic():
    # The original text/click single-page UI is still available.
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


@app.post("/api/state")
def api_state():
    """Lightweight poll — returns current state (used to await painted art)."""
    sid, g = _game()
    if not g:
        return jsonify({"error": "no_session"}), 404
    return _respond(g)


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


@app.post("/api/hint")
def api_hint():
    sid, g = _game()
    if not g:
        return jsonify({"error": "no_session"}), 404
    return jsonify(g.hint())


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


@app.post("/api/rate")
def api_rate():
    sid, g = _game()
    if not g:
        return jsonify({"error": "no_session"}), 404
    d = request.get_json(silent=True) or {}
    return jsonify(g.rate(d.get("stars"), d.get("comment", "")))


if __name__ == "__main__":
    port = int(os.environ.get("ST_WEB_PORT", "17080"))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
