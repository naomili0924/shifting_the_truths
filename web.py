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
import time
import uuid

import yaml
from flask import Flask, request, jsonify, send_from_directory

import imagegen
import ttsgen
import talkgen
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


def _safe_provider(cfg_block):
    """Build a provider, falling back to mock if it can't initialize."""
    try:
        p = provider_from_config(cfg_block)
        return p, isinstance(p, MockProvider)
    except Exception:
        return MockProvider(), True


# The NPC begins each reply with an emotion tag ([nervous]/[angry]/...); we strip it from
# the text and colour the VOICE instead, so the feeling lands in the sound, not in
# unspeakable stage directions. chatterbox-turbo ignores exaggeration/cfg, so the actual
# shaping (rate/pitch/gain/tremor + temperature) lives in ttsgen by emotion name.
_EMOTIONS = {"nervous", "angry", "defensive", "cold", "sad", "calm"}


def _split_emotion(text):
    """Pull a leading [emotion] tag off an NPC reply. Returns (emotion|None, rest)."""
    import re as _re
    m = _re.match(r"\s*[\[\(（【]\s*(nervous|angry|defensive|cold|sad|calm)\s*"
                  r"[\]\)）】]\s*", text or "", _re.I)
    if m:
        return m.group(1).lower(), (text or "")[m.end():]
    return None, (text or "")


def _clean_reply(text):
    """Drop any leftover stage directions so the chat shows (and voices) only spoken
    words: (parentheticals), *actions*, and stray [bracketed] notes."""
    import re as _re
    t = text or ""
    t = _re.sub(r"[\(（][^)）]*[\)）]", " ", t)   # (stage directions)
    t = _re.sub(r"\*[^*]+\*", " ", t)                          # *actions*
    t = _re.sub(r"[\[【][^\]】]*[\]】]", " ", t)   # [leftover tags]
    t = _re.sub(r"\s+", " ", t).strip()
    return t


_VOICE_MAX = 130


def _voice_text(text, max_chars=_VOICE_MAX):
    """Reduce a line to a short, speakable 'taste' for TTS (the full text is still
    displayed). chatterbox is autoregressive, so long lines take 15-20s; voicing
    the first sentence or two keeps it snappy. Strips parenthetical/bracketed
    stage directions (narration, not speech)."""
    import re as _re
    t = _re.sub(r"[\(\[][^)\]]*[\)\]]", " ", text or "")   # drop (stage directions)
    t = _re.sub(r"\s+", " ", t).strip()
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "),
              cut.rfind("。"), cut.rfind("！"), cut.rfind("？"))
    return (cut[:end + 1] if end > 40 else cut).strip()


class WebGame:
    """One playable session, driven by API calls instead of stdin."""

    def __init__(self, cfg: dict, lang: str):
        self.last_seen = time.time()        # for idle-session eviction
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
        # English case too — image prompts are always English (SDXL), even for a zh game.
        en_path = cases.get("en") or gcfg.get("case", "case.yaml")
        self.en_case = (self.case if self.lang == "en"
                        else load_case(os.path.join(HERE, en_path)))
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
        self.scene_manifests = {}      # act_num -> [scene manifest dicts] (built on roll)

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
        self.faces: dict[str, str | None] = {}        # npc name -> filename
        self.item_images: dict[str, str | None] = {}  # item name -> scene-crop filename (evidence thumb)
        self.scenes: dict[int, list] = {}             # act_num -> [{id,title,backdrop,objects}]
        self.acts_painted: set = set()                # act nums whose scenes are painted
        self.collected: set = set()                   # collectible obj ids picked up
        gen = imagegen.instance(self.lang)            # one inpaint model serves EN + ZH
        self.art_enabled = bool(gen and gen.available())

        # Optional voice (chatterbox-turbo ONNX, English only). Suspects get a
        # gender-matched voice; the player's questions get the narrator voice.
        audcfg = cfg.get("audio") or {}
        self.voices_assign = audcfg.get("assign") or {}
        self.player_voice = audcfg.get("player_voice", "narrator")
        self.default_voice = audcfg.get("default_voice", self.player_voice)
        tgen = ttsgen.instance(self.lang)
        self.audio_enabled = bool(tgen and tgen.available())
        # Lip-sync talking head needs a face (art) + the reply wav (audio) + Wav2Lip.
        vgen = talkgen.instance()
        self.video_enabled = bool(vgen and vgen.available()
                                  and self.art_enabled and self.audio_enabled)

        # Everything heavy runs off the request path, behind the intro. The image and
        # voice models load SEQUENTIALLY inside _setup — loading both torch models at the
        # same time races on CUDA (meta-tensor error) — so there is no separate warm thread.
        threading.Thread(target=self._setup, daemon=True).start()

    def _warm_audio(self):
        if not self.audio_enabled:
            return
        g = ttsgen.instance(self.lang)
        if g:
            try:
                g.warm(self._voices_in_use())
            except Exception:  # noqa: BLE001 - warming is best-effort
                pass

    def _voices_in_use(self):
        vs = {self.player_voice, self.default_voice}
        vs.update(self.voices_assign.get(n) for n in self.names)
        return [v for v in vs if v]

    def _enqueue_tts(self, text, voice, emotion=None):
        """Servable URL for (text, voice, emotion); start background synthesis if not yet
        cached. The emotion picks chatterbox params so the feeling lands in the sound.
        Returns the URL immediately (browser polls until it appears), or None if voice is
        off / unavailable."""
        if not self.audio_enabled or not (text or "").strip():
            return None
        g = ttsgen.instance(self.lang)
        if g is None:
            return None
        emo = emotion if emotion in _EMOTIONS else None
        name = g.url_name(text, voice, emotion=emo)
        if not g.cached(text, voice, emotion=emo):
            threading.Thread(target=g.generate, args=(text, voice),
                             kwargs={"emotion": emo}, daemon=True).start()
        return f"/assets/cache/audio/{name}"

    def _enqueue_talk(self, text, voice, emotion, name):
        """Servable URL for the lip-sync talking-head video of *name* speaking *text* in
        *emotion*. Spawns background gen (reply wav -> Wav2Lip video). Returns the URL
        immediately (browser polls until it lands), or None if video is off/unavailable."""
        if not self.video_enabled or not (text or "").strip():
            return None
        tv = talkgen.instance()
        tg = ttsgen.instance(self.lang)
        gen = imagegen.instance(self.lang)
        if tv is None or tg is None or gen is None:
            return None
        with self._img_lock:
            portrait_fn = self.faces.get(name)
        if not portrait_fn:
            return None
        emo = emotion if emotion in _EMOTIONS else None
        portrait_path = os.path.join(gen.cache_dir, portrait_fn)
        wav_path = os.path.join(tg.cache_dir, tg.url_name(text, voice, emotion=emo))
        vid_name = tv.url_name(portrait_path, wav_path)
        if not tv.cached(portrait_path, wav_path):
            threading.Thread(target=self._make_talk,
                             args=(text, voice, emo, portrait_path, wav_path),
                             daemon=True).start()
        return f"/assets/cache/video/{vid_name}"

    def _make_talk(self, text, voice, emotion, portrait_path, wav_path):
        """Ensure the reply wav exists, then lip-sync it onto the portrait (background)."""
        tg = ttsgen.instance(self.lang)
        if tg and not os.path.isfile(wav_path):
            tg.generate(text, voice, emotion=emotion)
        tv = talkgen.instance()
        if tv and os.path.isfile(wav_path):
            tv.generate(portrait_path, wav_path)

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
        # Resolve each act into TWO sub-scenes with embedded collectible + decoy objects.
        # Cheap (no painting) and depends on the rolled plot (which clues are present).
        self.scene_manifests = {
            act["act"]: rooms.scene_manifests(self.case, act, self.gt, self.lang,
                                              en_case=self.en_case)
            for act in self.case["acts"]
        }
        self.ready = True

        # 2) Deeds + NPC system prompts (what the TALK phase needs) are a judge API call
        #    with no GPU use, so run them in their OWN thread — NOT gated behind the slow
        #    GPU painting + voice-warm below. Otherwise talk stays "still gathering" for
        #    ~30-45s after the art is ready and the player's first question vanishes.
        threading.Thread(target=self._setup_npcs, daemon=True).start()

        # 3) Paint Act 1's scenes behind the intro; faces once; then prefetch later acts
        #    in the background (SDXL is slow, so we don't paint all acts up front).
        first = self.case["acts"][0]["act"]
        if self.art_enabled:
            self._paint_faces()
        self._paint_act(first)
        # Warm the voice model only AFTER the image model has loaded — loading both torch
        # models concurrently races on CUDA. Then warm the lip-sync model (also sequential).
        self._warm_audio()
        if self.video_enabled:
            vg = talkgen.instance()
            if vg:
                try:
                    vg.warm()
                except Exception:  # noqa: BLE001
                    pass
        threading.Thread(target=self._paint_rest, args=(first,), daemon=True).start()

    def _setup_npcs(self):
        """Judge-written deeds + each NPC's system prompt (talk-phase readiness). Runs in
        its own thread, concurrent with painting, so suspects are ready to answer fast."""
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
    def _paint_faces(self):
        """Paint the five fixed suspect faces once (from this game's localized
        character descriptions). Best-effort; missing faces fall back to initials."""
        gen = imagegen.instance(self.lang)
        if not gen:
            return
        # Build the portrait prompt from the ENGLISH character descriptions (SDXL is
        # English-prompted), with a photo style/negative so faces are photoreal.
        chars = self.en_case.get("characters", [])
        for i, name in enumerate(self.names):
            src = chars[i] if i < len(chars) else None
            # Match the face gender to the (gender-matched) voice — the case has no gender
            # field, so derive it from the voice id (male_* / female_*).
            vid = str(ttsgen.voice_for(name, self.voices_assign, self.default_voice))
            gender = "man" if vid.startswith("male") else "woman"
            fn = (gen.generate(rooms.portrait_prompt(src, gender=gender),
                               negative=rooms.PHOTO_NEG, style="") if src else None)
            with self._img_lock:
                self.faces[name] = fn

    def _paint_act(self, act_no):
        """Compose this act's two sub-scenes (base backdrop + embedded objects). Records
        each collectible's scene-crop as its evidence thumbnail. Safe with no art (scenes
        are still registered with backdrop=None so the UI shows object chips on a plain
        backdrop). Idempotent per act."""
        if act_no in self.acts_painted:
            return
        gen = imagegen.instance(self.lang)
        painted = []
        for man in self.scene_manifests.get(act_no, []):
            backdrop, crops = (gen.compose_scene(man["prompt"], man["objects"])
                               if gen else (None, {}))
            if crops:
                for obj in man["objects"]:
                    if obj.get("kind") == "collectible" and obj["id"] in crops:
                        with self._img_lock:
                            self.item_images[obj["name"]] = crops[obj["id"]]
            painted.append({**man, "backdrop": backdrop})
        with self._img_lock:
            self.scenes[act_no] = painted
            self.acts_painted.add(act_no)

    def _paint_rest(self, first_act):
        """Prefetch the remaining acts' scenes in the background so they're ready by the
        time the player gets there (each act's talk phase buys painting time)."""
        for act in self.case["acts"]:
            if act["act"] != first_act:
                self._paint_act(act["act"])

    def _image_state(self):
        cur = self._cur()
        act_num = cur["act"]["act"] if cur else None
        with self._img_lock:
            faces = dict(self.faces)
            painted = self.scenes.get(act_num) if act_num else None
            collected = set(self.collected)
            done = act_num in self.acts_painted if act_num else False
        # Fall back to the (always-available) manifest with no backdrop until painted.
        src = painted if painted is not None else \
            [{**m, "backdrop": None} for m in (self.scene_manifests.get(act_num, []) if act_num else [])]
        scenes = []
        for sc in (src or []):
            objs = [{"id": o["id"], "kind": o["kind"], "name": o["name"],
                     "x": o.get("x", .5), "y": o.get("y", .5),
                     "w": o.get("w", .22), "h": o.get("h", .22),
                     "collected": o["id"] in collected}
                    for o in sc.get("objects", [])]
            scenes.append({
                "id": sc["id"], "title": sc.get("title", ""),
                "backdrop": f"/assets/cache/{sc['backdrop']}" if sc.get("backdrop") else None,
                "objects": objs,
            })
        return {
            "art_enabled": self.art_enabled,
            "video_enabled": self.video_enabled,
            "images_done": done,
            "backdrop": scenes[0]["backdrop"] if scenes else None,   # intro waits on this
            "faces": {n: (f"/assets/cache/{fn}" if fn else None)
                      for n, fn in faces.items()},
            "scenes": scenes,
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

    # ---- scene objects: pick up clues / inspect decoys --------------
    def _find_obj(self, obj_id):
        """Locate an object by id within the current act's scene manifests."""
        cur = self._cur()
        if not cur:
            return None
        for sc in self.scene_manifests.get(cur["act"]["act"], []):
            for o in sc.get("objects", []):
                if o["id"] == obj_id:
                    return o
        return None

    def pickup(self, obj_id):
        """Collect a clue object: add it to the evidence pouch (for showing suspects and
        the accusation), mark it collected. Only collectibles present in the rolled plot
        are pickable — decoys are not."""
        if not self.ready:
            return {"error": "loading"}
        cur = self._cur()
        if not cur or cur["phase"]["type"] != "search":
            return {"error": "not_search"}
        o = self._find_obj(obj_id)
        if not o:
            return {"error": "bad_obj"}
        if o.get("kind") != "collectible":
            return {"error": "not_collectible"}
        if obj_id in self.collected:
            return {"ok": True, "already": True, "name": o["name"]}
        self.collected.add(obj_id)
        if o["name"] not in self._have:
            self._have.add(o["name"])
            self.evidence.append({"name": o["name"],
                                  "found_text": o.get("found_text", "")})
        fn = self.item_images.get(o["name"])
        return {"ok": True, "name": o["name"], "found_text": o.get("found_text", ""),
                "image": f"/assets/cache/{fn}" if fn else None}

    def inspect(self, obj_id):
        """Look at an object without picking it up. Decoys return flavour text; a
        collectible peeked at returns its found_text (it is still pickable)."""
        if not self.ready:
            return {"error": "loading"}
        o = self._find_obj(obj_id)
        if not o:
            return {"error": "bad_obj"}
        if o.get("kind") == "collectible":
            return {"ok": True, "kind": "collectible", "name": o["name"],
                    "text": o.get("found_text", ""), "collected": obj_id in self.collected}
        return {"ok": True, "kind": "decoy", "name": o["name"],
                "text": o.get("flavor", "")}

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
        # Voice the player's question now, so it plays while the reply generates.
        ask_audio = self._enqueue_tts(_voice_text(text), self.player_voice)
        reply_raw = self._npc_reply(name, text)
        # Strip the leading [emotion] tag + any stage directions; the feeling goes into the
        # voice, the chat shows only spoken words.
        emotion, spoken = _split_emotion(reply_raw)
        spoken = _clean_reply(spoken) or _clean_reply(reply_raw) or reply_raw
        self.chat[name].append({"who": "npc", "text": spoken})
        voice = ttsgen.voice_for(name, self.voices_assign, self.default_voice)
        vt = _voice_text(spoken)
        reply_audio = self._enqueue_tts(vt, voice, emotion=emotion)
        reply_video = self._enqueue_talk(vt, voice, emotion, name)   # lip-sync talking head
        return {"ok": True, "reply": spoken, "emotion": emotion,
                "ask_audio": ask_audio, "reply_audio": reply_audio, "reply_video": reply_video}

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
        reply_raw = self._npc_reply(name, msg)
        emotion, spoken = _split_emotion(reply_raw)
        spoken = _clean_reply(spoken) or _clean_reply(reply_raw) or reply_raw
        self.chat[name].append({"who": "npc", "text": spoken})
        voice = ttsgen.voice_for(name, self.voices_assign, self.default_voice)
        vt = _voice_text(spoken)
        reply_audio = self._enqueue_tts(vt, voice, emotion=emotion)
        reply_video = self._enqueue_talk(vt, voice, emotion, name)   # lip-sync talking head
        return {"ok": True, "reply": spoken, "emotion": emotion,
                "reply_audio": reply_audio, "reply_video": reply_video}

    def hint(self):
        """Last-resort help when time is short and the player is short of two clues:
        point (gently) at up to (2 - collected) uncollected clue OBJECTS in the current
        act's scenes. Also returns unsearched spots for the classic UI. Minimum reveal."""
        if not self.ready:
            return {"objects": [], "spots": []}
        cur = self._cur()
        if not cur or cur["phase"]["type"] != "search":
            return {"objects": [], "spots": []}
        act = cur["act"]; act_no = act["act"]
        collectibles = [(sc["id"], o) for sc in self.scene_manifests.get(act_no, [])
                        for o in sc.get("objects", []) if o.get("kind") == "collectible"]
        got = sum(1 for _, o in collectibles if o["id"] in self.collected)
        need = max(0, 2 - got)
        objects = [{"scene_id": sid, "obj_id": o["id"]}
                   for sid, o in collectibles if o["id"] not in self.collected][:need]
        # classic-UI fallback: unsearched spots that hold a present item
        spots = []
        if need > 0:
            for s in act["spots"]:
                if s["id"] in self.searched:
                    continue
                if any(item_present(i, self.gt) for i in s.get("items", [])):
                    spots.append(s["id"])
                if len(spots) >= need:
                    break
        return {"objects": objects, "spots": spots}

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


@app.after_request
def _no_cache_html(resp):
    """Never let the browser/CDN serve a stale game.html or game JS — otherwise an
    updated client (e.g. the voice playback fix) won't reach players who refresh."""
    ct = resp.headers.get("Content-Type", "")
    if "text/html" in ct or "javascript" in ct:
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
    return resp


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

# Configure the (optional) text-to-speech generator the same way.
try:
    _audcfg = dict(_CFG.get("audio") or {})
    for _k in ("cache_dir", "voices_dir"):
        if _audcfg.get(_k) and not os.path.isabs(_audcfg[_k]):
            _audcfg[_k] = os.path.join(HERE, _audcfg[_k])
    ttsgen.configure(_audcfg)
except Exception:
    ttsgen.configure({"enabled": False})

# Configure the (optional) lip-sync talking-head generator (Wav2Lip).
try:
    _vidcfg = dict(_CFG.get("video") or {})
    if _vidcfg.get("cache_dir") and not os.path.isabs(_vidcfg["cache_dir"]):
        _vidcfg["cache_dir"] = os.path.join(HERE, _vidcfg["cache_dir"])
    talkgen.configure(_vidcfg)
except Exception:
    talkgen.configure({"enabled": False})


def _game():
    sid = request.headers.get("X-Session")
    if not sid:
        sid = (request.get_json(silent=True) or {}).get("sid")
    g = GAMES.get(sid)
    if g:
        g.last_seen = time.time()        # touch so active sessions aren't evicted
    return sid, g


SESSION_TTL = 1800        # evict sessions idle > 30 min
SESSION_MAX = 16          # hard cap on concurrent sessions

def _evict_idle():
    """Drop idle/excess sessions; when none remain, free the GPU pipelines
    (image + voice) so VRAM is reclaimed. They lazily reload on next use."""
    now = time.time()
    for s in [s for s, g in list(GAMES.items()) if now - getattr(g, "last_seen", now) > SESSION_TTL]:
        GAMES.pop(s, None)
    if len(GAMES) > SESSION_MAX:                       # evict oldest beyond the cap
        for s, _ in sorted(GAMES.items(), key=lambda kv: getattr(kv[1], "last_seen", 0))[:len(GAMES) - SESSION_MAX]:
            GAMES.pop(s, None)
    if not GAMES:
        for _mod in (imagegen, ttsgen, talkgen):
            try: _mod.dispose()
            except Exception: pass


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
    _evict_idle()                       # reclaim memory/GPU from stale sessions
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


@app.post("/api/pickup")
def api_pickup():
    sid, g = _game()
    if not g:
        return jsonify({"error": "no_session"}), 404
    r = g.pickup((request.get_json(silent=True) or {}).get("obj_id", ""))
    return _respond(g, {"action": r})


@app.post("/api/inspect")
def api_inspect():
    sid, g = _game()
    if not g:
        return jsonify({"error": "no_session"}), 404
    r = g.inspect((request.get_json(silent=True) or {}).get("obj_id", ""))
    return _respond(g, {"action": r})


@app.post("/api/talk")
def api_talk():
    sid, g = _game()
    if not g:
        return jsonify({"error": "no_session"}), 404
    d = request.get_json(silent=True) or {}
    r = g.talk(d.get("name", ""), d.get("text", ""))
    return _respond(g, {"action": r})


@app.post("/api/say")
def api_say():
    """Voice an arbitrary line (default: the player's question voice) and return
    its audio URL immediately. Lets the browser play the question while the reply
    is still being generated. Returns {"audio": null} when voice is unavailable."""
    sid, g = _game()
    if not g:
        return jsonify({"error": "no_session"}), 404
    d = request.get_json(silent=True) or {}
    url = g._enqueue_tts(d.get("text", ""), d.get("voice") or g.player_voice)
    return jsonify({"audio": url})


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
