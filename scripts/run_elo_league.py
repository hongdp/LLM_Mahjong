"""Elo anchor-league rating for DNN mahjong policies.

Design doc: docs/design_elo_league.md. Two modes:

  calibrate  — round-robin among the anchor pool, joint MLE (bc_cnn pinned
               at 1000), writes experiments/elo_league/anchors.json
  rate       — score one candidate checkpoint against frozen anchors,
               appends to experiments/elo_league/history.jsonl

Every raw match is saved under experiments/elo_league/matches/ so any fit
can be redone later from primary data.
"""

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_arena_dnn import load_dnn            # noqa: E402
from src.tasks.mahjong.arena import run_match          # noqa: E402

LEAGUE_DIR = "experiments/elo_league"
LN10_400 = math.log(10) / 400.0

ANCHOR_POOL = {
    "bc_cnn":      "experiments/arch_sweep/models/cnn_m.pt",
    "rf600":       "experiments/dnn_scratch_massive_20260815/games_600000.pt",
    "ppo44_240":   "experiments/dnn_ppo_massive_20260815/games_240000.pt",
    "ppo44_600":   "experiments/dnn_ppo_massive_20260815/games_600000.pt",
    "reuse11_600": "experiments/dnn_ppo_reuse1_20260815/games_600000.pt",
    "e700":        "experiments/dnn_exp12_E_20260816/games_700000.pt",
    "vit240":      "experiments/_cloud_ckpts/dnn_vit_rl_r4/games_240000.pt",
}
PINNED = ("bc_cnn", 1000.0)   # scale origin, fixed forever


def deal_scores(rows):
    """Per-deal outcome for side A: 1 win / 0.5 tie / 0 loss by sign(diff)."""
    return [1.0 if r["diff"] > 0 else 0.0 if r["diff"] < 0 else 0.5
            for r in rows]


def expected(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def play_pair(name_a, path_a, name_b, path_b, deals, seed0, parallel, device):
    """One duplicate-deal match; returns per-deal scores for side A."""
    t0 = time.time()
    policies = {"A": load_dnn(path_a, device), "B": load_dnn(path_b, device)}
    seeds = [seed0 + i for i in range(deals)]
    rows = run_match(None, None, seeds, parallel=parallel,
                     dnn_policies=policies, dnn_device=device)
    scores = deal_scores(rows)
    diffs = [r["diff"] for r in rows]
    mean = sum(diffs) / len(diffs)
    out = {"a": name_a, "b": name_b, "path_a": path_a, "path_b": path_b,
           "deals": deals, "seed0": seed0, "score_a": sum(scores),
           "mean_diff": mean, "elapsed_s": round(time.time() - t0, 1),
           "rows": rows}
    os.makedirs(f"{LEAGUE_DIR}/matches", exist_ok=True)
    fn = f"{LEAGUE_DIR}/matches/{name_a}_vs_{name_b}_{seed0}.json"
    json.dump(out, open(fn, "w"))
    print(f"[match] {name_a} vs {name_b}: score {sum(scores):.1f}/{deals} "
          f"mean_diff {mean:+.0f}  ({out['elapsed_s']}s)", flush=True)
    return scores


def fit_ratings(games, ratings, free, iters=500, damp=1.0):
    """Joint MLE, damped diagonal-Newton. games: (name_a, name_b, s)."""
    for _ in range(iters):
        grad = {n: 0.0 for n in free}
        info = {n: 1e-9 for n in free}
        for a, b, s in games:
            e = expected(ratings[a], ratings[b])
            g = LN10_400 * (s - e)
            w = (LN10_400 ** 2) * e * (1 - e)
            if a in grad:
                grad[a] += g
                info[a] += w
            if b in grad:
                grad[b] -= g
                info[b] += w
        max_step = 0.0
        for n in free:
            step = damp * grad[n] / info[n]
            ratings[n] += step
            max_step = max(max_step, abs(step))
        if max_step < 0.01:
            break
    return ratings


def rating_se(games, ratings, name):
    info = sum((LN10_400 ** 2) * expected(ratings[a], ratings[b])
               * (1 - expected(ratings[a], ratings[b]))
               for a, b, _ in games if name in (a, b))
    return float("inf") if info == 0 else 1.0 / math.sqrt(info)


def residuals(games, ratings, name):
    """Per-opponent (actual − expected) mean score for `name`."""
    per = {}
    for a, b, s in games:
        if a == name:
            per.setdefault(b, []).append(s - expected(ratings[a], ratings[b]))
        elif b == name:
            per.setdefault(a, []).append((1 - s) - expected(ratings[b], ratings[a]))
    return {o: round(sum(v) / len(v), 3) for o, v in per.items()}


def cmd_calibrate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = list(ANCHOR_POOL)
    games = []
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            scores = play_pair(na, ANCHOR_POOL[na], nb, ANCHOR_POOL[nb],
                               args.deals, args.seed0, args.parallel, device)
            games += [(na, nb, s) for s in scores]
    ratings = {n: 1000.0 for n in names}
    free = [n for n in names if n != PINNED[0]]
    ratings[PINNED[0]] = PINNED[1]
    fit_ratings(games, ratings, free)
    table = {n: {"rating": round(ratings[n], 1),
                 "se": round(rating_se(games, ratings, n), 1),
                 "path": ANCHOR_POOL[n],
                 "residuals": residuals(games, ratings, n)}
             for n in names}
    os.makedirs(LEAGUE_DIR, exist_ok=True)
    json.dump({"pinned": PINNED, "deals_per_pair": args.deals,
               "seed0": args.seed0, "date": args.date, "anchors": table},
              open(f"{LEAGUE_DIR}/anchors.json", "w"), indent=1)
    for n in sorted(names, key=lambda x: -ratings[x]):
        print(f"{n:>12}  {ratings[n]:7.1f} ± {table[n]['se']:.1f}")
    print(f"saved {LEAGUE_DIR}/anchors.json")


def rate_checkpoint(ckpt, label, deals, seed0, parallel, device,
                    use=None, init_guess=1000.0):
    """Rate one checkpoint against frozen anchors; append to history.jsonl.

    use: anchor-name subset; None = all. init_guess seeds the fit and (in
    the ladder watcher) drives nearest-anchor selection upstream.
    """
    league = json.load(open(f"{LEAGUE_DIR}/anchors.json"))
    anchors = league["anchors"]
    use = use or list(anchors)
    games = []
    for n in use:
        scores = play_pair("cand", ckpt, n, anchors[n]["path"],
                           deals, seed0, parallel, device)
        games += [("cand", n, s) for s in scores]
    ratings = {n: anchors[n]["rating"] for n in use}
    ratings["cand"] = init_guess
    fit_ratings(games, ratings, ["cand"])
    rec = {"ckpt": ckpt, "label": label,
           "elo": round(ratings["cand"], 1),
           "se": round(rating_se(games, ratings, "cand"), 1),
           "anchors": use, "deals_per_anchor": deals, "seed0": seed0,
           "date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "residuals": residuals(games, ratings, "cand")}
    os.makedirs(LEAGUE_DIR, exist_ok=True)
    with open(f"{LEAGUE_DIR}/history.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"ELO {label}: {rec['elo']} ± {rec['se']}  residuals {rec['residuals']}",
          flush=True)
    return rec


def cmd_rate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use = args.anchors.split(",") if args.anchors else None
    rate_checkpoint(args.ckpt, args.label or args.ckpt, args.deals,
                    args.seed0, args.parallel, device, use=use)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ca = sub.add_parser("calibrate")
    ca.add_argument("--deals", type=int, default=200)
    ca.add_argument("--seed0", type=int, default=20260816)
    ca.add_argument("--parallel", type=int, default=20)
    ca.set_defaults(fn=cmd_calibrate)
    ra = sub.add_parser("rate")
    ra.add_argument("--ckpt", required=True)
    ra.add_argument("--label", default=None)
    ra.add_argument("--anchors", default=None,
                    help="comma-separated anchor subset (default: all)")
    ra.add_argument("--deals", type=int, default=100)
    ra.add_argument("--seed0", type=int, required=True,
                    help="fresh seed0 per evaluation; recorded in history")
    ra.add_argument("--parallel", type=int, default=20)
    ra.set_defaults(fn=cmd_rate)
    args = ap.parse_args()
    args.date = time.strftime("%Y-%m-%d %H:%M:%S")
    args.fn(args)


if __name__ == "__main__":
    main()
