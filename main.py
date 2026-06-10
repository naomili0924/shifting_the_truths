"""
main.py - Shifting Truth, the three-act playable build.

Run:
    pip install pyyaml
    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py                       # uses config.yaml
    python main.py --config myconf.yaml --seed 7

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
from engine import (
    load_case, Director, GroundTruth, build_npc_system_prompt,
    referee_check, judge_accusation, item_present,
    judge_generate_deeds, deal_boundary_gossip,
    judge_select_culprit, record_ground_truth,
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


class Game:
    def __init__(self, config_path: str, seed_override: int | None):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.costs = self.cfg.get("costs", {})
        self.npc_llm = provider_from_config(self.cfg["agents"]["npc"])
        self.judge_llm = provider_from_config(self.cfg["agents"]["judge"])

        self.case = load_case(self.cfg["game"].get("case", "case.yaml"))
        seed = seed_override if seed_override is not None \
            else self.cfg["game"].get("seed")
        # Developer debug log for the judge's hidden choices (null to disable).
        self.debug_path = self.cfg["game"].get("debug_log")
        self.director = Director(self.case, seed=seed)
        # The judge LLM picks the culprit + motive; deterministic code
        # still rolls method, timeline, clues and the accident special.
        self.gt: GroundTruth = self.director.roll(selector=self._select_culprit)
        record_ground_truth(self.debug_path, self.gt)

        self.names = [c["character"] for c in self.case["characters"]]
        self.deeds = judge_generate_deeds(
            self.case, self.gt, self.judge_llm, self.director.rng)
        self.base_prompt = {
            c["character"]: build_npc_system_prompt(
                self.case, c, self.gt,
                deeds=self.deeds.get(c["character"]))
            for c in self.case["characters"]
        }
        self.extras: dict[str, list[str]] = {n: [] for n in self.names}
        self.histories: dict[str, list[dict]] = {n: [] for n in self.names}
        self.questions_log: dict[str, list[str]] = {n: [] for n in self.names}
        self.evidence: list[dict] = []   # {name, found_text}
        self.searched: set[str] = set()

    # ---- director plumbing -------------------------------------------
    def _select_culprit(self) -> tuple[str, list[str]]:
        """Selector handed to the director: the judge LLM picks who and why."""
        sel = judge_select_culprit(
            self.case, self.judge_llm, self.director.rng, self.debug_path)
        return sel["culprit"], sel["flaws"]

    # ---- NPC plumbing ------------------------------------------------
    def system_for(self, name: str) -> str:
        sysp = self.base_prompt[name]
        if self.extras[name]:
            sysp += ("\n\nNEW MEMORIES (things you did, found out, or "
                     "heard since the night began - treat as true "
                     "unless marked hearsay):\n" +
                     "\n".join(f"  - {x}" for x in self.extras[name]))
        return sysp

    def npc_reply(self, name: str, user_text: str) -> str:
        self.histories[name].append({"role": "user", "content": user_text})
        reply = self.npc_llm.chat(self.system_for(name),
                                  self.histories[name])
        hint = referee_check(reply, name, self.gt)
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
            say(f"{i}. {c['character']} - {c['role']}")

    def show_evidence(self) -> None:
        if not self.evidence:
            say("[Your evidence pouch is empty.]")
            return
        say("EVIDENCE COLLECTED:")
        for e in self.evidence:
            say(f"* {e['name']}", indent="  ")
            say(e["found_text"], indent="      ")

    def show_notes(self) -> None:
        any_notes = False
        for n in self.names:
            turns = [m for m in self.histories[n]
                     if m["role"] == "assistant"]
            if turns:
                any_notes = True
                say(f"--- {n} ({len(turns)} statements) ---")
                for t in turns:
                    say(t["content"], indent="  ")
        if not any_notes:
            say("[Nobody has told you anything yet.]")

    # ---- SEARCH phase --------------------------------------------------
    def run_search(self, act: dict, minutes: int) -> None:
        spots = act["spots"]
        cost = int(self.costs.get("search", 2))
        say(f"[SEARCH PHASE - {minutes} minutes. 'look' to survey, "
            f"'search <spot>' ({cost} min each), 'evidence', 'next' "
            "to stop early.]")
        while minutes > 0:
            cmd = ask(f"\n(search, {minutes} min) > ")
            if not cmd:
                continue
            verb, _, rest = cmd.partition(" ")
            verb = verb.lower()
            if verb in ("next", "skip"):
                return
            if verb == "look":
                for s in spots:
                    tag = " (searched)" if s["id"] in self.searched else ""
                    say(f"- {s['name']}{tag}")
                continue
            if verb == "evidence":
                self.show_evidence()
                continue
            if verb == "help":
                say("look | search <spot> | evidence | next")
                continue
            if verb == "search":
                target = None
                for s in spots:
                    if rest.lower() in s["id"] or \
                       rest.lower() in s["name"].lower():
                        target = s
                        break
                if not target or not rest:
                    say("Search where? 'look' lists the spots.")
                    continue
                if minutes < cost:
                    say("[No time left to search properly.]")
                    return
                minutes -= cost
                if target["id"] in self.searched:
                    say(f"[You already went over {target['name']}.]")
                    continue
                self.searched.add(target["id"])
                found = [i for i in target.get("items", [])
                         if item_present(i, self.gt)]
                if not found:
                    say(target.get("empty_text",
                                   "Nothing of interest."), indent="  ")
                else:
                    for item in found:
                        say(f"FOUND: {item['name']}", indent="  ")
                        say(item["found_text"], indent="    ")
                        self.evidence.append(
                            {"name": item["name"],
                             "found_text": item["found_text"]})
                continue
            say("Unknown command. Try 'help'.")
        say("[Time's up - Elena is calling everyone together.]")

    # ---- TALK phase ----------------------------------------------------
    def run_talk(self, minutes: int) -> None:
        qcost = int(self.costs.get("question", 1))
        scost = int(self.costs.get("show", 1))
        say(f"[TALK PHASE - {minutes} minutes. 'cast', 'talk <name>', "
            "'evidence', 'notes', 'next'. In conversation: ask "
            f"anything ({qcost} min), 'show <item>' ({scost} min), "
            "'back'.]")
        while minutes > 0:
            cmd = ask(f"\n(talk, {minutes} min) > ")
            if not cmd:
                continue
            verb, _, rest = cmd.partition(" ")
            verb = verb.lower()
            if verb in ("next", "skip"):
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
                say("cast | talk <name> | evidence | notes | next")
                continue
            if verb == "talk":
                name = self.resolve_name(rest)
                if not name:
                    say("Talk to whom? 'cast' lists them.")
                    continue
                say(f"[You corner {name}.]")
                while minutes > 0:
                    q = ask(f"You -> {name} ({minutes} min): ")
                    if not q:
                        continue
                    low = q.lower()
                    if low in ("back", "leave"):
                        break
                    if low.startswith("show "):
                        token = q[5:].strip().lower()
                        item = next(
                            (e for e in self.evidence
                             if token in e["name"].lower()), None)
                        if not item:
                            say("[You don't have that. 'evidence' "
                                "lists what you carry.]")
                            continue
                        minutes -= scost
                        msg = ("[The journalist places evidence in "
                               f"front of you: {item['name']}. "
                               f"{item['found_text']}]")
                        try:
                            reply = self.npc_reply(name, msg)
                            say(f"{name}: {reply}", indent="  ")
                        except Exception as e:
                            say(f"[provider error: {e}]")
                        continue
                    minutes -= qcost
                    self.questions_log[name].append(q)
                    try:
                        reply = self.npc_reply(name, q)
                        say(f"{name}: {reply}", indent="  ")
                    except Exception as e:
                        say(f"[provider error: {e}]")
                if minutes <= 0:
                    break
                continue
            say("Unknown command. Try 'help'.")
        say("[Time's up.]")

    # ---- act boundary ----------------------------------------------------
    def boundary(self, act_no: int) -> None:
        say("\n[Between acts, the suspects gather and murmur. "
            "Notes are compared. Stories adjust.]")
        gossip = deal_boundary_gossip(
            self.case, self.gt, act_no, self.deeds,
            self.questions_log, self.judge_llm, self.director.rng)
        for n, lines in gossip.items():
            self.extras[n].extend(f"(hearsay) {s}" for s in lines)

    # ---- forced conclusion ----------------------------------------------
    def conclusion(self) -> None:
        say("\n" + "=" * WRAP)
        say("Headlights in the courtyard. The police are walking up "
            "the drive. There is no more time: you must name the "
            "truth NOW. One accusation. No second chance.")
        self.show_cast()
        accused = ""
        while not accused:
            accused = ask("\nWHO is responsible for Diana's death? "
                          "(name, or 'accident'): ")
        reasoning = ask("WHY did they do it? Name the real motive: ")
        how = ask("HOW was it done? (optional, Enter to skip): ")
        say("\n[The room goes quiet as you speak...]")
        verdict = judge_accusation(self.case, self.gt, accused,
                                   reasoning, how, self.judge_llm)
        print("=" * WRAP)
        say(verdict)
        print("=" * WRAP)

    # ---- the night ---------------------------------------------------------
    def play(self) -> None:
        sc = self.case["scenario"]
        print("=" * WRAP)
        say(f"SHIFTING TRUTH - {sc['title']}")
        print("=" * WRAP)
        say(sc["setting"])
        print()
        say(sc["the_accident"])
        acts = self.case["acts"]
        for idx, act in enumerate(acts, 1):
            print("\n" + "-" * WRAP)
            say(f"ACT {act['act']}: {act['title']}")
            print("-" * WRAP)
            say(act.get("scene_intro", ""))
            for phase in act["phases"]:
                print()
                if phase["type"] == "search":
                    self.run_search(act, int(phase["time"]))
                else:
                    self.run_talk(int(phase["time"]))
            if idx < len(acts):
                self.boundary(act["act"])
        self.conclusion()


def main() -> None:
    ap = argparse.ArgumentParser(description="Shifting Truth")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    Game(args.config, args.seed).play()


if __name__ == "__main__":
    main()
