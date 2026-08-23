"""Claim-window resolution — shared by the LLM orchestrator, the batched
rollout and the DNN self-play driver.

Lives in its own dependency-light module so conventional-agent code paths
never import the LLM orchestration stack (langgraph): a cloud DNN box
bootstraps torch+numpy only. (Learned the hard way: two spot VMs
self-terminated on this import.)
"""

import re
from typing import List, Optional

from src.tasks.mahjong.table import PyMahjongTable, ACTION_RE
from src.core.chat_format import visible_text

_PRIORITY = {"kan": 1, "pon": 2, "chi": 3}


def _extract_action(raw_output: str) -> Optional[str]:
    """Extracts the action tag from OUTSIDE the think block only —
    actions merely mentioned while reasoning must not be executed."""
    m = ACTION_RE.search(visible_text(raw_output))
    return m.group(0) if m else None


def _resolve_claims(table: PyMahjongTable, candidates: List[dict]):
    """Applies one interrupt window's declarations to the table.

    `candidates` must be in seat order starting from the discarder — that
    order decides who collects the riichi sticks on a multiple ron.
    Returns (executed_candidates, game_over). Writes each candidate's
    engine reward into cand["reward"].
    """
    for cand in candidates:
        if cand["parsed"] is None:
            cand["reward"] = table.FORMAT_PENALTY

    actionable = [c for c in candidates
                  if c["parsed"] is not None and c["type"] not in (None, "skip")]

    # RCR 3.11: all declared rons are settled in one call; illegal ones
    # (furiten / no yaku) are penalized there and the melds play on.
    rons = [c for c in actionable if c["type"] == "ron"]
    if rons:
        _, rewards, done, info = table.step_ron([c["player_id"] for c in rons])
        for cand in rons:
            cand["reward"] = rewards[cand["player_id"]]
        winners = set(info.get("winners", []))
        if winners:
            return [c for c in rons if c["player_id"] in winners], done

    executed = []
    for cand in sorted([c for c in actionable if c["type"] != "ron"],
                       key=lambda c: _PRIORITY.get(c["type"], 9)):
        if executed:
            continue  # lost the priority race: action not applied, no penalty
        _, rewards, _, info = table.step_interrupt(cand["player_id"], cand["parsed"])
        cand["reward"] = rewards[cand["player_id"]]
        if info.get("interrupt", False):
            executed = [cand]
    return executed, False
