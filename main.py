"""
main.py - Shifting Truth, the three-act playable build.

Run:
    pip install pyyaml
    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py                       # uses config.yaml
    python main.py --config myconf.yaml --seed 7
    python main.py --lang zh             # play in Chinese
    python main.py --mode developer      # full debug logging

Language is the player's choice: --lang en|zh, or game.lang in
config.yaml, or an interactive prompt at startup if neither is set.
English stays the default; each language ships its own case file.

Agents (NPCs, judge) each take their brain from config.yaml:
public API, local ONNX runtime, or mock - independently.

Structure (phase order is defined per act in case.yaml):
    Act 1: SEARCH the scene (timed) -> TALK to suspects (timed)
    Act 2: SEARCH (timed) -> TALK (timed)   [+ gossip dealt between acts]
    Act 3: SEARCH -> TALK -> SEARCH (each timed)
    Then: you MUST name who, why, how. One shot. The judge decides.
"""

from __future__ import annotations
import argparse
import sys
import textwrap

import yaml

from providers import provider_from_config
from gamelog import GameLog
from i18n import t, normalize_lang, EN
from engine import (
    load_case, Director, GroundTruth, build_npc_system_prompt,
    referee_check, judge_accusation, item_present,
    judge_generate_deeds, deal_boundary_gossip,
    judge_select_culprit,
)

WRAP = 76


def say(text: str, indent: str = "") -> None:
    for para in str(text).split("\n"):
        print(textwrap.fill(para, WRAP, initial_indent=indent,
                            subsequent_indent=indent) if para else "")


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def resolve_lang(game_cfg: dict, override: str | None) -> str:
    """--lang > config game.lang > interactive startup prompt."""
    if override:
        return normalize_lang(override)
    if game_cfg.get("lang"):
        return normalize_lang(game_cfg["lang"])
    return normalize_lang(ask(EN["ui"]["lang_prompt"]))


def resolve_case_path(game_cfg: dict, lang: str) -> str:
    """Pick the case file for the chosen language."""
    cases = game_cfg.get("cases") or {}
    return (cases.get(lang)
            or game_cfg.get(f"case_{lang}")
            or game_cfg.get("case", "case.yaml"))


class Game:
    def __init__(self, config_path: str, seed_override: int | None,
                 mode_override: str | None = None,
                 lang_override: str | None = None):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        game_cfg = self.cfg["game"]
        self.costs = self.cfg.get("costs", {})
        self.npc_llm = provider_from_config(self.cfg["agents"]["npc"])
        self.judge_llm = provider_from_config(self.cfg["agents"]["judge"])

        # Language: player's choice (flag > config > startup prompt).
        self.lang = resolve_lang(game_cfg, lang_override)
        self.L = t(self.lang)
        self.ui = self.L["ui"]

        # Launch mode: production (conversation only) or developer (also
        # logs the judge's choice, ground truth, and NPC memory per stage).
        mode = mode_override or game_cfg.get("mode", "production")
        self.log = GameLog(mode, game_cfg.get("log_dir", "logs"))

        self.case = load_case(resolve_case_path(game_cfg, self.lang))
        seed = seed_override if seed_override is not None \
            else game_cfg.get("seed")
        self.director = Director(self.case, seed=seed, lang=self.lang)
        # The judge LLM picks the culprit + motive; deterministic code
        # still rolls method, timeline, clues and the accident special.
        self.gt: GroundTruth = self.director.roll(selector=self._select_culprit)
        self.log.dev_log(
            "ground_truth", lang=self.lang, is_murder=self.gt.is_murder,
            culprit=self.gt.culprit, flaws=self.gt.active_flaws,
            method=self.gt.method["id"] if self.gt.method else None,
            motive=self.gt.motive, secret_timeline=self.gt.secret_timeline,
            distributed_clues=self.gt.distributed_clues)

        self.names = [c["character"] for c in self.case["characters"]]
        self.deeds = judge_generate_deeds(
            self.case, self.gt, self.judge_llm, self.director.rng, self.lang)
        self.log.dev_log("deeds", deeds=self.deeds)
        self.base_prompt = {
            c["character"]: build_npc_system_prompt(
                self.case, c, self.gt,
                deeds=self.deeds.get(c["character"]), lang=self.lang)
            for c in self.case["characters"]
        }
        self.extras: dict[str, list[str]] = {n: [] for n in self.names}
        self.histories: dict[str, list[dict]] = {n: [] for n in self.names}
        self.questions_log: dict[str, list[str]] = {n: [] for n in self.names}
        self.evidence: list[dict] = []   # {name, found_text}
        self.searched: set[str] = set()
        self.cur_act: int = 0            # set by play() for log context
        # Reverse map: any accepted token -> canonical command verb.
        self._alias = {tok: canon
                       for canon, toks in self.L["commands"].items()
                       for tok in toks}
        self.log.conv("session_start", mode=self.log.mode, lang=self.lang,
                      title=self.case["scenario"]["title"], seed=seed)

    # ---- director plumbing -------------------------------------------
    def _select_culprit(self) -> tuple[str, list[str]]:
        """Selector handed to the director: the judge LLM picks who and why."""
        sel = judge_select_culprit(
            self.case, self.judge_llm, self.director.rng, self.lang)
        self.log.dev_log("culprit_selection", **sel)
        return sel["culprit"], sel["flaws"]

    def log_npc_memory(self, stage: str) -> None:
        """Developer mode: snapshot every NPC's memory at a stage boundary."""
        if not self.log.dev:
            return
        for n in self.names:
            self.log.dev_log(
                "npc_memory", stage=stage, name=n,
                system_prompt=self.system_for(n),
                hearsay=list(self.extras[n]),
                statements=[m for m in self.histories[n]
                            if m["role"] == "assistant"],
                history=list(self.histories[n]))

    # ---- command parsing ---------------------------------------------
    def canon(self, verb: str) -> str:
        """Map a typed verb (en or zh alias) to its canonical command."""
        return self._alias.get(verb.lower(), verb.lower())

    # ---- NPC plumbing ------------------------------------------------
    def system_for(self, name: str) -> str:
        sysp = self.base_prompt[name]
        if self.extras[name]:
            sysp += ("\n\n" + self.ui["new_memories"] + "\n" +
                     "\n".join(f"  - {x}" for x in self.extras[name]))
        return sysp

    def npc_reply(self, name: str, user_text: str) -> str:
        self.histories[name].append({"role": "user", "content": user_text})
        reply = self.npc_llm.chat(self.system_for(name),
                                  self.histories[name])
        hint = referee_check(reply, name, self.gt, self.lang)
        if hint:
            self.histories[name].append(
                {"role": "assistant", "content": reply})
            self.histories[name].append(
                {"role": "user",
                 "content": f"[OUT OF CHARACTER CORRECTION: {hint}]"})
            reply = self.npc_llm.chat(self.system_for(name),
                                      self.histories[name])
        self.histories[name].append({"role": "assistant", "content": reply})
        return reply

    def resolve_name(self, token: str) -> str | None:
        token = token.strip().lower()
        if token.isdigit() and 1 <= int(token) <= len(self.names):
            return self.names[int(token) - 1]
        for n in self.names:
            if token and token in n.lower():
                return n
        return None

    # ---- shared commands ---------------------------------------------
    def show_cast(self) -> None:
        for i, c in enumerate(self.case["characters"], 1):
            say(self.ui["cast_line"].format(i=i, name=c["character"],
                                            role=c["role"]))

    def show_evidence(self) -> None:
        if not self.evidence:
            say(self.ui["evidence_empty"])
            return
        say(self.ui["evidence_header"])
        for e in self.evidence:
            say(self.ui["evidence_item"].format(name=e["name"]), indent="  ")
            say(e["found_text"], indent="      ")

    def show_notes(self) -> None:
        any_notes = False
        for n in self.names:
            turns = [m for m in self.histories[n]
                     if m["role"] == "assistant"]
            if turns:
                any_notes = True
                say(self.ui["notes_header"].format(n=n, k=len(turns)))
                for t_ in turns:
                    say(t_["content"], indent="  ")
        if not any_notes:
            say(self.ui["notes_empty"])

    # ---- SEARCH phase --------------------------------------------------
    def run_search(self, act: dict, minutes: int) -> None:
        spots = act["spots"]
        cost = int(self.costs.get("search", 2))
        say(self.ui["search_header"].format(m=minutes, cost=cost))
        while minutes > 0:
            cmd = ask(self.ui["search_prompt"].format(m=minutes))
            if not cmd:
                continue
            verb, _, rest = cmd.partition(" ")
            verb = self.canon(verb)
            if verb == "next":
                return
            if verb == "look":
                for s in spots:
                    tag = self.ui["searched_tag"] if s["id"] in self.searched else ""
                    say(self.ui["spot_line"].format(name=s["name"], tag=tag))
                continue
            if verb == "evidence":
                self.show_evidence()
                continue
            if verb == "help":
                say(self.ui["search_help"])
                continue
            if verb == "search":
                target = None
                for s in spots:
                    if rest and (rest.lower() in s["id"] or
                                 rest.lower() in s["name"].lower()):
                        target = s
                        break
                if not target or not rest:
                    say(self.ui["search_where"])
                    continue
                if minutes < cost:
                    say(self.ui["no_time_search"])
                    return
                minutes -= cost
                if target["id"] in self.searched:
                    say(self.ui["already_searched"].format(name=target["name"]))
                    continue
                self.searched.add(target["id"])
                found = [i for i in target.get("items", [])
                         if item_present(i, self.gt)]
                if not found:
                    say(target.get("empty_text", self.ui["nothing"]),
                        indent="  ")
                else:
                    for item in found:
                        say(self.ui["found"].format(name=item["name"]),
                            indent="  ")
                        say(item["found_text"], indent="    ")
                        self.evidence.append(
                            {"name": item["name"],
                             "found_text": item["found_text"]})
                self.log.conv("search", act=self.cur_act, spot=target["name"],
                              found=[i["name"] for i in found])
                continue
            say(self.ui["unknown_cmd"])
        say(self.ui["time_up_search"])

    # ---- TALK phase ----------------------------------------------------
    def run_talk(self, minutes: int) -> None:
        qcost = int(self.costs.get("question", 1))
        scost = int(self.costs.get("show", 1))
        say(self.ui["talk_header"].format(m=minutes, q=qcost, s=scost))
        while minutes > 0:
            cmd = ask(self.ui["talk_prompt"].format(m=minutes))
            if not cmd:
                continue
            verb, _, rest = cmd.partition(" ")
            verb = self.canon(verb)
            if verb == "next":
                return
            if verb == "cast":
                self.show_cast()
                continue
            if verb == "evidence":
                self.show_evidence()
                continue
            if verb == "notes":
                self.show_notes()
                continue
            if verb == "help":
                say(self.ui["talk_help"])
                continue
            if verb == "talk":
                name = self.resolve_name(rest)
                if not name:
                    say(self.ui["talk_to_whom"])
                    continue
                say(self.ui["corner"].format(name=name))
                while minutes > 0:
                    q = ask(self.ui["you_to"].format(name=name, m=minutes))
                    if not q:
                        continue
                    qverb, _, qrest = q.partition(" ")
                    qcanon = self.canon(qverb)
                    if qcanon == "back" and not qrest.strip():
                        break
                    if qcanon == "show" and qrest.strip():
                        token = qrest.strip().lower()
                        item = next(
                            (e for e in self.evidence
                             if token in e["name"].lower()), None)
                        if not item:
                            say(self.ui["no_item"])
                            continue
                        minutes -= scost
                        msg = self.ui["evidence_present"].format(
                            name=item["name"], text=item["found_text"])
                        try:
                            reply = self.npc_reply(name, msg)
                            say(f"{name}: {reply}", indent="  ")
                        except Exception as e:
                            reply = self.ui["provider_error"].format(e=e)
                            say(reply)
                        self.log.conv("show_evidence", act=self.cur_act,
                                      name=name, item=item["name"],
                                      reply=reply, minutes_left=minutes)
                        continue
                    minutes -= qcost
                    self.questions_log[name].append(q)
                    try:
                        reply = self.npc_reply(name, q)
                        say(f"{name}: {reply}", indent="  ")
                    except Exception as e:
                        reply = self.ui["provider_error"].format(e=e)
                        say(reply)
                    self.log.conv("question", act=self.cur_act, name=name,
                                  question=q, reply=reply, minutes_left=minutes)
                if minutes <= 0:
                    break
                continue
            say(self.ui["unknown_cmd"])
        say(self.ui["time_up"])

    # ---- act boundary ----------------------------------------------------
    def boundary(self, act_no: int) -> None:
        say(self.ui["between_acts"])
        gossip = deal_boundary_gossip(
            self.case, self.gt, act_no, self.deeds,
            self.questions_log, self.judge_llm, self.director.rng, self.lang)
        prefix = self.ui["hearsay_prefix"]
        for n, lines in gossip.items():
            self.extras[n].extend(f"{prefix}{s}" for s in lines)
        self.log.dev_log("boundary_gossip", after_act=act_no, gossip=gossip)

    # ---- forced conclusion ----------------------------------------------
    def conclusion(self) -> None:
        print("\n" + "=" * WRAP)
        say(self.ui["concl_intro"])
        self.show_cast()
        accused = ""
        while not accused:
            accused = ask(self.ui["ask_who"].format(v=self.director.victim))
        reasoning = ask(self.ui["ask_why"])
        how = ask(self.ui["ask_how"])
        self.log.conv("accusation", accused=accused, why=reasoning, how=how)
        say(self.ui["room_quiet"])
        verdict = judge_accusation(self.case, self.gt, accused,
                                   reasoning, how, self.judge_llm, self.lang)
        print("=" * WRAP)
        say(verdict)
        print("=" * WRAP)
        self.log.conv("verdict", accused=accused, verdict=verdict)
        self.log_npc_memory("final")

    # ---- the night ---------------------------------------------------------
    def play(self) -> None:
        sc = self.case["scenario"]
        print("=" * WRAP)
        say(self.ui["title"].format(title=sc["title"]))
        print("=" * WRAP)
        say(sc["setting"])
        print()
        say(sc["the_accident"])
        acts = self.case["acts"]
        self.log_npc_memory("game_start")
        for idx, act in enumerate(acts, 1):
            self.cur_act = act["act"]
            print("\n" + "-" * WRAP)
            say(self.ui["act_header"].format(act=act["act"], title=act["title"]))
            print("-" * WRAP)
            say(act.get("scene_intro", ""))
            for phase in act["phases"]:
                print()
                if phase["type"] == "search":
                    self.run_search(act, int(phase["time"]))
                else:
                    self.run_talk(int(phase["time"]))
            self.log_npc_memory(f"act{act['act']}_end")
            if idx < len(acts):
                self.boundary(act["act"])
        self.conclusion()


def main() -> None:
    ap = argparse.ArgumentParser(description="Shifting Truth")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mode", choices=["developer", "production"],
                    default=None,
                    help="developer also logs the judge's choice, ground "
                         "truth, and NPC memory per stage (overrides config)")
    ap.add_argument("--lang", choices=["en", "zh"], default=None,
                    help="play language (overrides config; prompts if unset)")
    args = ap.parse_args()
    Game(args.config, args.seed, mode_override=args.mode,
         lang_override=args.lang).play()


if __name__ == "__main__":
    main()
