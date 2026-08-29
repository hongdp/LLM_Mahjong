"""Hanchan-scale 2v2 arena (exp53): placement/uma results on full matches.

Sides: a checkpoint path (planes-based policy) or "mortal:<state.pth>"
(black-box via its own libriichi, fed the per-deal event stream — its
home format, GRP placement head finally in-distribution).

Seating: side A on (0,2) for even game indices, (1,3) for odd. Per
hanchan the score is sign(uma_A - uma_B); the summary reports win share,
mean uma per side, placement histograms, and the pairwise Elo delta
400*log10(s/(1-s)). Results are hanchan-scale — NEVER write them into
the single-deal league history.

Usage:
  python scripts/run_hanchan_arena.py --a <ckpt|mortal:path> --b <...> \
      --games 100 --seed0 47000001 --out experiments/exp53_.../x.json
"""

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks.mahjong.hanchan import play_hanchan                     # noqa: E402
from src.agents.dnn.mjai_export import play_game_mjai                  # noqa: E402
from scripts.run_arena_dnn import load_dnn                             # noqa: E402
from scripts.arena_mortal_mjai import (MORTAL_DIR, MortalSeat,         # noqa: E402
                                       build_mortal_engine, net_policy)


def make_side(spec: str, device: str):
    """Returns (kind, factory) where factory(seats) -> policies dict for
    those seats + optional per-deal sink hook."""
    if spec.startswith("mortal:"):
        engine, _ = build_mortal_engine(spec.split(":", 1)[1], device)
        sys.path.insert(0, MORTAL_DIR)
        from libriichi.mjai import Bot
        return "mortal", (engine, Bot)
    net = load_dnn(spec, device)
    pol = net_policy(net, device, temperature=1.0)
    return "net", pol


def play_one(a_kind, a_impl, b_kind, b_impl, a_seats, seed, stats):
    seats_b = tuple(s for s in range(4) if s not in a_seats)
    policies = {}
    mortal_seats = {}
    for kind, impl, seats in ((a_kind, a_impl, a_seats),
                              (b_kind, b_impl, seats_b)):
        if kind == "net":
            for s in seats:
                policies[s] = impl
        else:
            engine, Bot = impl
            for s in seats:
                ms = MortalSeat(Bot, engine, s, stats)
                mortal_seats[s] = ms
                policies[s] = ms.policy

    if mortal_seats:
        # feed the per-deal event stream to the bots via a wrapped driver:
        # monkey-wire play_game_mjai's sink through play_hanchan by giving
        # the hanchan module a sink-capable policies dict — simplest is to
        # replay through play_hanchan with a sink closure via functools.
        from src.tasks.mahjong import hanchan as H
        orig = H.play_game_mjai

        def wrapped(table, pols, observer, sink):
            def fan(ev):
                for ms in mortal_seats.values():
                    ms.feed(ev)
            return orig(table, pols, observer=None, sink=fan)
        H.play_game_mjai = wrapped
        try:
            res = play_hanchan(policies, seed)
        finally:
            H.play_game_mjai = orig
    else:
        res = play_hanchan(policies, seed)
    uma_a = sum(res.uma_points[s] for s in a_seats)
    uma_b = sum(res.uma_points[s] for s in seats_b)
    return res, uma_a, uma_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)

    a_kind, a_impl = make_side(args.a, args.device)
    b_kind, b_impl = make_side(args.b, args.device)
    stats = {"ok": 0, "fallback": 0}
    rows, s_sum = [], 0.0
    plc = {"a": [0] * 4, "b": [0] * 4}
    t0 = time.time()
    for g in range(args.games):
        a_seats = (0, 2) if g % 2 == 0 else (1, 3)
        res, ua, ub = play_one(a_kind, a_impl, b_kind, b_impl,
                               a_seats, args.seed0 + g, stats)
        s = 1.0 if ua > ub else 0.0 if ua < ub else 0.5
        s_sum += s
        for st in range(4):
            (plc["a"] if st in a_seats else plc["b"])[res.placements[st] - 1] += 1
        rows.append({"game": g, "a_seats": list(a_seats), "uma_a": ua,
                     "uma_b": ub, "deals": len(res.deals),
                     "placements": res.placements, "busted": res.busted})
        if (g + 1) % 10 == 0:
            print(f"[{g+1}/{args.games}] share={s_sum/(g+1):.3f} "
                  f"({(g+1)/(time.time()-t0)*60:.1f} hanchan/min, "
                  f"fallback {stats['fallback']})", flush=True)
    n = args.games
    p = min(max(s_sum / n, 1e-6), 1 - 1e-6)
    delta = 400 * math.log10(p / (1 - p))
    se_p = math.sqrt(p * (1 - p) / n)
    out = {"a": args.a, "b": args.b, "games": n, "share_a": s_sum / n,
           "share_se": se_p, "elo_delta_a_minus_b": round(delta, 1),
           "mean_uma_a": sum(r["uma_a"] for r in rows) / n,
           "mean_uma_b": sum(r["uma_b"] for r in rows) / n,
           "placements": plc, "bridge_fallbacks": stats["fallback"],
           "seed0": args.seed0, "rows": rows}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"RESULT share_a={out['share_a']:.3f}±{se_p:.3f} "
          f"elo_delta={out['elo_delta_a_minus_b']:+.1f} "
          f"uma_a={out['mean_uma_a']:.0f} uma_b={out['mean_uma_b']:.0f}", flush=True)


if __name__ == "__main__":
    main()
