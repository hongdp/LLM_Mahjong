"""Conditional-entropy probe (roadmap step 0, 2026-08-23): is the policy's
randomness CONDITIONED on how much the choice matters?

For closed-hand discard decisions we bucket by the tile-efficiency gap
between the best and second-best discard (post-shanten first, then ukeire
count; gap 0 = the top two discards are exactly equivalent) and report the
mean policy entropy per bucket. A human-expert-shaped policy shows a
monotonically DECREASING curve (ties -> high entropy, clear best -> ~0);
a uniform-entropy-bonus policy shows a flat line.
"""
import argparse
import collections
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_arena_dnn import load_dnn                           # noqa: E402
from src.agents.dnn.encoder import encode_state, legal_mask          # noqa: E402
from src.agents.dnn.selfplay import _resolve_claims                  # noqa: E402
from src.tasks.mahjong.shanten import TileEfficiency                 # noqa: E402
from src.tasks.mahjong.table import PyMahjongTable, ACTION_RE        # noqa: E402

_TE = TileEfficiency()
BUCKETS = ((0, "tie"), (1, "1-2"), (3, "3-5"), (6, "6+"))


def gap_bucket(hand):
    ranked = _TE.evaluate_discards_ranked(hand)
    scored = sorted(((sh, -len(uk)) for sh, uk in ranked.values()))
    if len(scored) < 2:
        return None
    (s1, u1), (s2, u2) = scored[0], scored[1]
    if s1 != s2:
        return "6+"                       # second best loses a whole shanten
    gap = (-u1) - (-u2) if False else (u2 - u1)
    gap = abs(u1 - u2)
    for lo, name in reversed(BUCKETS):
        if gap >= lo:
            return name
    return "tie"


def probe(net, games, seed0):
    variant = getattr(net, "encoder_variant", "v1")
    agg = collections.defaultdict(lambda: [0.0, 0])
    for g in range(games):
        random.seed(seed0 + g)
        table = PyMahjongTable(randomize_round=True)
        table.text_obs = False
        guard = 0
        while not table.finished and guard < 600:
            guard += 1
            pid = table.turn
            actions = table.get_legal_actions(pid)
            if not actions:
                break
            planes, scalars = encode_state(table, pid, variant=variant)
            mask, lookup = legal_mask(actions)
            with torch.no_grad():
                logits, _ = net.forward_with_value(planes[None], scalars[None], mask[None])
                logp = torch.log_softmax(logits[0], 0)
                p = logp.exp()
            ent = float(-(p[p > 0] * logp[p > 0]).sum())
            hand = table.hands[pid]
            if not table.melds[pid] and not table.riichi[pid] and len(hand) == 14 \
                    and table._shanten(hand, 0) <= 2:
                b = gap_bucket(hand)
                if b:
                    agg[b][0] += ent
                    agg[b][1] += 1
            chosen = lookup[int(torch.multinomial(p, 1))]
            _, _, done, info = table.step(pid, chosen)
            if done:
                break
            if not (info.get("discarded") or info.get("chankan")):
                continue
            cands = []
            for off in range(1, 4):
                other = (pid + off) % 4
                opts = table.get_interrupt_actions(other)
                if len(opts) == 1:
                    continue
                p2, s2 = encode_state(table, other, variant=variant)
                m2, l2 = legal_mask(opts)
                with torch.no_grad():
                    lg, _ = net.forward_with_value(p2[None], s2[None], m2[None])
                ch = l2[int(torch.multinomial(torch.softmax(lg[0], 0), 1))]
                mm = ACTION_RE.search(ch)
                cands.append({"player_id": other, "parsed": ch,
                              "type": mm.group(1) if mm else None, "reward": 0.0})
            executed, done = _resolve_claims(table, cands)
            if done:
                break
            if not executed:
                if table.pending_kan:
                    table.resolve_pending_kan()
                else:
                    _, rd = table.advance_turn()
                    if rd:
                        break
    return {name: {"mean_entropy": v[0] / v[1], "n": v[1]}
            for name, v in agg.items() if v[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=43000000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(8)
    res = {}
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        res[name] = probe(load_dnn(path, "cpu"), a.games, a.seed0)
        row = "  ".join(f"{b}: H={res[name].get(b, {}).get('mean_entropy', float('nan')):.2f}"
                        f"(n={res[name].get(b, {}).get('n', 0)})" for _, b in BUCKETS)
        print(f"[{name}] {row}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
