"""Python-engine self-play loop with observation hooks, shared by the
behaviour probes (scripts/probe_defense.py, scripts/probe_riichi.py).

One net sits on all four seats; claims are resolved exactly as
`selfplay.play_game` does (same engine calls in the same order, so a T=0
probe game reproduces the trainer's game step for step), but no
trajectory is kept — the caller observes each turn-phase decision through
two callbacks instead:

    before_turn(table, pid, actions) -> ctx      # state BEFORE the policy acts
    after_turn(ctx, table, pid, actions, action_str)   # the chosen engine action,
                                                       # still before table.step

Interrupt-window decisions (ron/pon/chi/kan replies) are played by the same
net but not reported: the probes only measure turn-phase choices.
"""

import random

from src.agents.dnn.selfplay import _choose
from src.tasks.mahjong.claims import _resolve_claims
from src.tasks.mahjong.table import ACTION_RE, PyMahjongTable


def play_probe_game(net, seed, temperature, device="cpu",
                    before_turn=None, after_turn=None,
                    randomize_round: bool = True, max_steps: int = 600):
    """Play one deal from `seed` and return the finished table (the caller
    reads result_summary / points / riichi from it)."""
    random.seed(seed)
    table = PyMahjongTable(randomize_round=randomize_round)
    table.text_obs = False
    guard = 0
    while not table.finished and guard < max_steps:
        guard += 1
        pid = table.turn
        actions = table.get_legal_actions(pid)
        if not actions:
            break
        ctx = before_turn(table, pid, actions) if before_turn else None
        _, action_str = _choose(net, table, pid, actions, temperature,
                                device, cmode="none")
        if after_turn:
            after_turn(ctx, table, pid, actions, action_str)
        _, _, done, info = table.step(pid, action_str)
        if done:
            break
        if not (info.get("discarded") or info.get("chankan")):
            continue
        candidates = []
        for offset in range(1, 4):
            other = (pid + offset) % 4
            options = table.get_interrupt_actions(other)
            if len(options) == 1:
                continue
            _, a_str = _choose(net, table, other, options, temperature,
                               device, cmode="none")
            mm = ACTION_RE.search(a_str)
            candidates.append({"player_id": other, "parsed": a_str,
                               "type": mm.group(1) if mm else None,
                               "reward": 0.0})
        executed, done = _resolve_claims(table, candidates)
        if done:
            break
        if not executed:
            if table.pending_kan:
                table.resolve_pending_kan()
            else:
                _, r_done = table.advance_turn()
                if r_done:
                    break
    return table
