"""
gamelog.py — session logging for Shifting Truth.

Two launch modes (config game.mode or --mode):

  production  : logs the full CONVERSATION — every player question and
                suspect reply, evidence shown, searches and their finds,
                and the final accusation + verdict.

  developer   : everything production logs, PLUS the hidden layer —
                the judge's culprit/motive choice, the resolved ground
                truth, and a snapshot of every NPC's memory at each stage
                (system prompt, accumulated hearsay, conversation so far).

Logs are JSON Lines under <log_dir>/<session>/:
    conversation.jsonl   (both modes)
    developer.jsonl      (developer mode only; the spoiler layer)

Everything is written to disk only — nothing reaches the player's screen.
"""

from __future__ import annotations
import datetime
import json
import os

PRODUCTION = "production"
DEVELOPER = "developer"


def normalize_mode(mode: str | None) -> str:
    return DEVELOPER if str(mode or "").lower().startswith("dev") else PRODUCTION


class GameLog:
    def __init__(self, mode: str | None = PRODUCTION,
                 log_dir: str = "logs", session: str | None = None):
        self.mode = normalize_mode(mode)
        self.dev = self.mode == DEVELOPER
        # Timestamp + pid so concurrent or rapid-fire games never share a
        # session directory (the per-second stamp alone can collide).
        self.session = session or (
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            + f"-{os.getpid()}")
        self.dir = os.path.join(log_dir, self.session)
        try:
            os.makedirs(self.dir, exist_ok=True)
        except Exception:
            pass
        self.conv_path = os.path.join(self.dir, "conversation.jsonl")
        self.dev_path = (os.path.join(self.dir, "developer.jsonl")
                         if self.dev else None)

    # ---- low-level ----------------------------------------------------
    def _write(self, path: str | None, record: dict) -> None:
        if not path:
            return
        rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
               **record}
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ---- conversation (both modes) ------------------------------------
    def conv(self, event: str, **fields) -> None:
        self._write(self.conv_path, {"event": event, **fields})

    # ---- developer-only (the spoiler layer) ---------------------------
    def dev_log(self, event: str, **fields) -> None:
        if self.dev:
            self._write(self.dev_path, {"event": event, **fields})
