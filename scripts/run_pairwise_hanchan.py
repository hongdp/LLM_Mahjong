"""All-pairs hanchan ladder + Bradley-Terry fit + transitivity residuals (exp123).

Drives `scripts/run_hanchan_arena.py` for every unordered pair of players,
several pairs at a time, over the SAME seed segments (common walls across
pairs; >= 2 non-overlapping segments per SKILLS), then fits Bradley-Terry
ratings by MLE and reports, per pair, the observed share, the BT-predicted
share and the z residual. Non-transitivity = a pair whose observed share is
far from what the ratings predict; an older checkpoint beating a newer one
is reported separately.

Players are given as label=path. Results are hanchan-scale (uma sign) —
never written into the single-deal league history.

Usage:
  python scripts/run_pairwise_hanchan.py --players L48=... C384=... \
      --segments 123000001,456000001 --games-per-segment 150 --jobs 6 \
      --threads 4 --temperature 0 --out-dir experiments/exp123_.../pairs \
      --summary experiments/exp123_.../ladder.json
  python scripts/run_pairwise_hanchan.py --analyze-only --players ... --out-dir ... --summary ...
"""

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ARENA = os.path.join(HERE, "run_hanchan_arena.py")


def pair_path(out_dir, a, b, seed0):
    return os.path.join(out_dir, f"{a}__vs__{b}__s{seed0}.json")


def run_pair(a, pa, b, pb, seed0, games, out_dir, threads, temperature, device):
    out = pair_path(out_dir, a, b, seed0)
    if os.path.exists(out):
        return out, "cached"
    cmd = [sys.executable, ARENA, "--a", pa, "--b", pb, "--games", str(games),
           "--seed0", str(seed0), "--device", device, "--threads", str(threads),
           "--temperature", str(temperature), "--out", out]
    log = out[:-5] + ".log"
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    return out, f"rc={rc}"


def bt_fit(players, wins, n_games, iters=2000, lr=0.5):
    """Bradley-Terry MLE on a logistic scale (rating in Elo units, sum fixed
    at 0). wins[a][b] = number of hanchan a beat b (ties count 0.5 each side)."""
    r = {p: 0.0 for p in players}
    for _ in range(iters):
        grad = {p: 0.0 for p in players}
        for a in players:
            for b in players:
                if a == b or n_games[a][b] == 0:
                    continue
                pa = 1.0 / (1.0 + 10 ** ((r[b] - r[a]) / 400.0))
                grad[a] += wins[a][b] - n_games[a][b] * pa
        for p in players:
            r[p] += lr * grad[p]
        m = sum(r.values()) / len(players)
        for p in players:
            r[p] -= m
    return r


def analyze(players, order, out_dir, segments, summary_path):
    wins = {a: {b: 0.0 for a in players for b in players} for a in players}
    n_games = {a: {b: 0 for a in players for b in players} for a in players}
    rows = {}
    for a, b in itertools.combinations(players, 2):
        s_sum, n, uma_a, uma_b = 0.0, 0, 0.0, 0.0
        for seed0 in segments:
            p = pair_path(out_dir, a, b, seed0)
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            n += d["games"]
            s_sum += d["share_a"] * d["games"]
            uma_a += d["mean_uma_a"] * d["games"]
            uma_b += d["mean_uma_b"] * d["games"]
        if n == 0:
            continue
        wins[a][b] += s_sum
        wins[b][a] += n - s_sum
        n_games[a][b] += n
        n_games[b][a] += n
        rows[(a, b)] = {"n": n, "share_a": s_sum / n,
                        "mean_uma_a": uma_a / n, "mean_uma_b": uma_b / n}
    ratings = bt_fit(players, wins, n_games)
    pairs = []
    for (a, b), row in rows.items():
        pred = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400.0))
        se = math.sqrt(max(pred * (1 - pred), 1e-9) / row["n"])
        z = (row["share_a"] - pred) / se
        se_obs = math.sqrt(max(row["share_a"] * (1 - row["share_a"]), 1e-9) / row["n"])
        z_vs_half = (row["share_a"] - 0.5) / se_obs
        # order[] is the lineage order (older first); a is always the earlier label
        older_beats_newer = (order.index(a) < order.index(b)) and z_vs_half >= 2.0
        pairs.append({"a": a, "b": b, **row, "bt_pred_a": pred, "z_resid": z,
                      "z_vs_half": z_vs_half, "older_beats_newer_2sigma": older_beats_newer})
    pairs.sort(key=lambda r: -abs(r["z_resid"]))
    summary = {"players": players, "ratings_elo": ratings, "pairs": pairs,
               "max_abs_z_resid": max((abs(p["z_resid"]) for p in pairs), default=0.0),
               "n_pairs_abs_z_ge_2p5": sum(abs(p["z_resid"]) >= 2.5 for p in pairs),
               "n_older_beats_newer_2sigma": sum(p["older_beats_newer_2sigma"] for p in pairs)}
    json.dump(summary, open(summary_path, "w"), indent=1)
    print("Bradley-Terry ratings (Elo units, mean 0):")
    for p in order:
        if p in ratings:
            print(f"  {p:8s} {ratings[p]:+7.1f}")
    print(f"{'pair':22s} {'n':>4s} {'share_a':>7s} {'bt_pred':>7s} {'z_res':>6s} {'z_half':>6s}")
    for r in pairs:
        print(f"{r['a']+' vs '+r['b']:22s} {r['n']:4d} {r['share_a']:7.3f} {r['bt_pred_a']:7.3f} "
              f"{r['z_resid']:+6.2f} {r['z_vs_half']:+6.2f}"
              + ("  <-- older beats newer" if r["older_beats_newer_2sigma"] else ""))
    print(f"max |z_resid| = {summary['max_abs_z_resid']:.2f}; pairs with |z|>=2.5: "
          f"{summary['n_pairs_abs_z_ge_2p5']}; older-beats-newer (2 sigma): "
          f"{summary['n_older_beats_newer_2sigma']}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", nargs="+", required=True, help="label=path, in lineage order (oldest first)")
    ap.add_argument("--segments", default="123000001,456000001", help="comma list of seed0 per wall segment")
    ap.add_argument("--games-per-segment", type=int, default=150)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    order = [p.split("=", 1)[0] for p in args.players]
    paths = {p.split("=", 1)[0]: p.split("=", 1)[1] for p in args.players}
    segments = [int(s) for s in args.segments.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.analyze_only:
        jobs = [(a, paths[a], b, paths[b], s) for a, b in itertools.combinations(order, 2)
                for s in segments]
        print(f"{len(jobs)} arena jobs ({len(order)} players, {len(segments)} segments, "
              f"{args.games_per_segment} hanchan each), {args.jobs} at a time", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(run_pair, a, pa, b, pb, s, args.games_per_segment, args.out_dir,
                              args.threads, args.temperature, args.device) for a, pa, b, pb, s in jobs]
            for i, f in enumerate(futs):
                out, status = f.result()
                print(f"[{i+1}/{len(jobs)}] {os.path.basename(out)} {status}", flush=True)
    analyze(order, order, args.out_dir, segments, args.summary)


if __name__ == "__main__":
    main()
