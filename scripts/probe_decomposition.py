"""Hand-decomposition probe (exp27): does the policy's discard keep the
hand's best shape on tile-efficiency grounds?

Measurement only (never a training signal): for closed-hand own-turn
decisions with shanten <= 2, the tile-efficiency oracle
(TileEfficiency.evaluate_discards_ranked: lowest post-discard shanten,
then widest ukeire, visible-tile-aware ukeire ignored) defines the set of
"shape-optimal" discards. We report the policy's greedy agreement and its
probability mass on that set, stratified by shanten and by shape
complexity = the longest single-suit block (tiles of the most-populated
suit), which is where multiple readings (2333s-style overlaps) live.

Usage: python scripts/probe_decomposition.py --ckpt NAME=path.pt ... --games 300
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


def optimal_discards(hand):
    ranked = _TE.evaluate_discards_ranked(hand)
    best = min((sh, -len(uk)) for sh, uk in ranked.values())
    return {t for t, (sh, uk) in ranked.items() if (sh, -len(uk)) == best}, best[0]


def complexity(hand):
    return max(sum(1 for t in hand if t[-1] == s) for s in "mps")


def probe(net, games, seed0, temperature=1.0):
    variant = getattr(net, "encoder_variant", "v1")
    agg = collections.defaultdict(lambda: collections.Counter())
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
                probs = torch.softmax(logits[0], 0)
            greedy = lookup[int(probs.argmax())]
            chosen = lookup[int(torch.multinomial(probs, 1))] if temperature > 0 else greedy
            hand = table.hands[pid]
            if (not table.melds[pid] and not table.riichi[pid] and len(hand) == 14
                    and table._shanten(hand, 0) <= 2):
                opt, sh = optimal_discards(hand)
                key = (sh, "long" if complexity(hand) >= 9 else ("mid" if complexity(hand) >= 6 else "short"))
                c = agg[key]
                c["n"] += 1
                gm = ACTION_RE.search(greedy)
                gt = gm.group(2)
                if gm.group(1) in ("discard", "riichi") and gt:
                    gt = "5" + gt[1] if gt[0] == "0" else gt
                    c["greedy_opt"] += int(gt in opt)
                mass = 0.0
                for i, a in lookup.items():
                    m = ACTION_RE.search(a)
                    if m.group(1) in ("discard", "riichi") and m.group(2):
                        t = m.group(2); t = "5" + t[1] if t[0] == "0" else t
                        if t in opt:
                            mass += float(probs[i])
                c["mass_opt"] += mass
                c["n_opt"] += len(opt)
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
    out = {}
    for (sh, cx), c in sorted(agg.items()):
        out[f"shanten{sh}_{cx}"] = {"n": c["n"], "greedy_agree": c["greedy_opt"] / c["n"],
                                   "mass_on_opt": c["mass_opt"] / c["n"],
                                   "mean_n_opt": c["n_opt"] / c["n"]}
    tot = sum(c["n"] for c in agg.values())
    out["all"] = {"n": tot,
                  "greedy_agree": sum(c["greedy_opt"] for c in agg.values()) / max(tot, 1),
                  "mass_on_opt": sum(c["mass_opt"] for c in agg.values()) / max(tot, 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="NAME=path pairs")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=98000000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(8)
    res = {}
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        net = load_dnn(path, "cpu")
        res[name] = probe(net, a.games, a.seed0)
        print(name, json.dumps(res[name]["all"]), flush=True)
        for k, v in res[name].items():
            if k != "all":
                print(f"   {k:16s} n={v['n']:5d} greedy_agree={v['greedy_agree']:.3f} mass={v['mass_on_opt']:.3f} |opt|={v['mean_n_opt']:.2f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
