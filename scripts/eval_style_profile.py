"""Style profile of a self-play policy vs human reference statistics.

Self-play win rate cannot measure absolute strength (four copies of one net
split decisive games by symmetry at ANY skill level). What CAN be compared
against published human统计 is the STYLE vector: per-seat agari rate,
deal-in (houjuu) rate, riichi rate, call rate, and the table's draw rate.
The deal-in rate is the sharpest mirror — a reckless policy can fake a
human-like draw rate but not a human-like houjuu rate.

Caveats printed with the output: our games are single hands (no hanchan
placement strategy), EMA RCR-flavored rules, and the call rate counts ankan
as "called" (proxy). Human reference numbers must be cited in the report,
not hardcoded here as truth.

Usage:
  python scripts/eval_style_profile.py --ckpt A=path1.pt B=path2.pt \
      --games 4000 --workers 15
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.dnn.selfplay import play_game                 # noqa: E402
from scripts.run_arena_dnn import load_dnn                    # noqa: E402

WIN_RE = re.compile(r"玩家(\d)\s*(?:自摸|荣和|抢杠)")
HOUJUU_RE = re.compile(r"放铳:玩家(\d)")
TSUMO_RE = re.compile(r"玩家(\d)\s*自摸")


def _worker(args):
    path, n_games, seed0, temperature = args
    torch.set_num_threads(1)
    net = load_dnn(path, "cpu")
    agg = {"games": 0, "draws": 0, "wins": 0, "tsumo": 0, "deal_ins": 0,
           "riichi": 0, "called": 0, "seats": 0}
    for g in range(n_games):
        game = play_game(net, temperature=temperature, device="cpu",
                         deal_seed=seed0 + g)
        r = game.result or ""
        agg["games"] += 1
        agg["seats"] += 4
        winners = {int(m) for m in WIN_RE.findall(r)}
        if not winners:
            agg["draws"] += 1
        agg["wins"] += len(winners)
        agg["tsumo"] += len(TSUMO_RE.findall(r))
        agg["deal_ins"] += len(set(HOUJUU_RE.findall(r)))
        agg["riichi"] += sum(1 for x in (game.riichi or []) if x)
        agg["called"] += sum(1 for n in (game.n_melds or []) if n > 0)
    return agg


def profile(path, games, workers, temperature, seed0):
    per = [games // workers] * workers
    for i in range(games - sum(per)):
        per[i] += 1
    jobs = [(path, per[w], seed0 + sum(per[:w]), temperature)
            for w in range(workers) if per[w]]
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        parts = pool.map(_worker, jobs)
    tot = {k: sum(p[k] for p in parts) for k in parts[0]}
    n, s = tot["games"], tot["seats"]
    return {
        "games": n,
        "draw_rate": tot["draws"] / n,
        "agari_rate": tot["wins"] / s,          # per-seat per-hand
        "tsumo_share": tot["tsumo"] / max(tot["wins"], 1),
        "houjuu_rate": tot["deal_ins"] / s,     # per-seat per-hand
        "riichi_rate": tot["riichi"] / s,
        "call_rate": tot["called"] / s,         # ankan counted (proxy)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="label=path pairs")
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed0", type=int, default=90260816)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    for spec in args.ckpt:
        label, path = spec.split("=", 1)
        t0 = time.time()
        results[label] = profile(path, args.games, args.workers,
                                 args.temperature, args.seed0)
        r = results[label]
        print(f"[{label}] {r['games']} games ({time.time()-t0:.0f}s)  "
              f"流局 {r['draw_rate']:.1%}  和牌 {r['agari_rate']:.1%}  "
              f"放铳 {r['houjuu_rate']:.1%}  立直 {r['riichi_rate']:.1%}  "
              f"副露 {r['call_rate']:.1%}  自摸占比 {r['tsumo_share']:.1%}",
              flush=True)
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1, ensure_ascii=False)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
