"""Rate Mortal on the HANCHAN ladder as an external reference (exp56).

Mortal is a black box behind the mjai bridge, so it cannot ride the
vectorized worker: it plays batch-1, single-threaded, ~6.7 hanchan/min
(measured), which is 36x slower than our own vectorized path and another
4.5x on top of that for the bridge (JSON<->Rust per event, two bot
instances, a 934-plane observation). The fix that costs nothing is
horizontal: the process is one core of 24, so shard the match seeds over
N processes and merge.

Two further economies:
  * duplicate pairing (both orientations on the SAME match seed), the
    same unit the ladder scores, so the numbers are directly comparable;
  * rate against INFORMATIVE anchors only — a 200-0 pair carries almost
    no information about the rating, it just burns an hour.

Anchors play at whatever temperature their pool was calibrated at; the
pool file is the authority (exp56: never assume T=1 again).

  python scripts/rate_mortal_hanchan.py --pairs 200 --seed0 49800000 \
      --anchors bc49,bc51_v3r2,exp46I,exp46Cb,exp27A_1M --shards 6
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_arena_dnn import load_dnn                            # noqa: E402
from scripts.run_elo_league import (anchors_path, engine_fingerprint,  # noqa: E402
                                    fit_ratings, league_dir, rating_se,
                                    residuals, HANCHAN_FILES)
from scripts.arena_mortal_mjai import (MORTAL_DIR, MortalSeat,         # noqa: E402
                                       build_mortal_engine, net_policy)
from src.tasks.mahjong.hanchan import play_hanchan                     # noqa: E402

DEFAULT_ANCHORS = "bc49,bc51_v3r2,exp46I,exp46Cb,exp27A_1M"


def play_one(engine, Bot, anchor_pol, seed, a_seats, stats):
    """One hanchan with Mortal on `a_seats`. Returns Mortal's summed uma."""
    from src.tasks.mahjong import hanchan as H
    seats = {s: MortalSeat(Bot, engine, s, stats) for s in a_seats}
    policies = {s: (seats[s].policy if s in a_seats else anchor_pol)
                for s in range(4)}
    orig = H.play_game_mjai

    def wrapped(table, pols, observer, sink):
        def fan(ev):
            for ms in seats.values():
                ms.feed(ev)
        return orig(table, pols, observer=None, sink=fan)

    H.play_game_mjai = wrapped
    try:
        res = play_hanchan(policies, seed)
    finally:
        H.play_game_mjai = orig
    return sum(res.uma_points[s] for s in a_seats), res


def run_shard(a):
    """Worker: `pairs` duplicate match-pairs against one anchor."""
    engine, _ = build_mortal_engine(a.state, a.device)
    sys.path.insert(0, MORTAL_DIR)
    from libriichi.mjai import Bot
    pool = json.load(open(anchors_path(True, a.tag)))
    entry = pool["anchors"][a.anchor]
    temp = (float(a.force_temp) if a.force_temp is not None
            else float(pool.get("temperature", 1.0)))
    anchor_pol = net_policy(load_dnn(entry["path"], a.device), a.device,
                            temperature=temp)
    stats = {"ok": 0, "fallback": 0}
    scores, diffs, rows = [], [], []
    for i in range(a.pairs):
        seed = a.seed0 + i
        uma = 0.0
        for orient in (0, 1):
            a_seats = (0, 2) if orient == 0 else (1, 3)
            u, res = play_one(engine, Bot, anchor_pol, seed, a_seats, stats)
            uma += u
            # primary record, same shape the vectorized arena archives: any
            # other fit (per-match sign, pt margin, placements) is a refit,
            # never a replay
            rows.append({"seed": seed, "wall": seed, "a_seats": list(a_seats),
                         "a_pts": u, "b_pts": -u,
                         "placements": list(res.placements),
                         "n_deals": len(res.deals), "busted": bool(res.busted),
                         "uma": list(res.uma_points)})
        # uma is zero-sum over the table, so Mortal's summed uma IS the margin
        scores.append(1.0 if uma > 0 else 0.0 if uma < 0 else 0.5)
        diffs.append(2.0 * uma)
    json.dump({"anchor": a.anchor, "seed0": a.seed0, "pairs": a.pairs,
               "anchor_temperature": temp, "scores": scores, "diffs": diffs,
               "games": rows,
               "fallback": stats["fallback"], "ok": stats["ok"]},
              open(a.out, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="data/mortal_ext/mortal_298k.pth")
    ap.add_argument("--pairs", type=int, default=200,
                    help="duplicate match-pairs per anchor (2 hanchan each)")
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--anchors", default=DEFAULT_ANCHORS,
                    help="informative subset; saturated pairs are a waste")
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--tag", default=None, help="pool tag (e.g. T0)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--force_temp", type=float, default=None,
                    help="override the opponent temperature (takes the run "
                         "OFF the pool's calibrated scale — use for a direct "
                         "deployment-vs-deployment probe, not for rating)")
    # worker mode
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--anchor", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.worker:
        return run_shard(a)

    names = a.anchors.split(",")
    pool_file = anchors_path(True, a.tag)
    pool = json.load(open(pool_file))
    want = pool.get("engine", {}).get("hanchan_fingerprint")
    have = engine_fingerprint(files=HANCHAN_FILES)
    if want and want != have:
        raise SystemExit(f"HANCHAN EPOCH MISMATCH: pool {want} vs current {have}")
    anchors = pool["anchors"]
    outdir = a.outdir or f"{league_dir(True)}/mortal_shards"
    os.makedirs(outdir, exist_ok=True)

    games, per_anchor, fb = [], {}, 0
    for name in names:
        t0 = time.time()
        per = [a.pairs // a.shards] * a.shards
        for i in range(a.pairs - sum(per)):
            per[i] += 1
        procs, outs, lo = [], [], 0
        for k, cnt in enumerate(per):
            if cnt == 0:
                continue
            out = f"{outdir}/{name}_{a.seed0}_{k}.json"
            outs.append(out)
            cmd = [sys.executable, os.path.abspath(__file__), "--worker",
                   "--anchor", name, "--pairs", str(cnt),
                   "--seed0", str(a.seed0 + lo), "--out", out,
                   "--state", a.state, "--device", a.device]
            if a.tag:
                cmd += ["--tag", a.tag]
            if a.force_temp is not None:
                cmd += ["--force_temp", str(a.force_temp)]
            lo += cnt
            procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.STDOUT))
        for p in procs:
            if p.wait() != 0:
                raise SystemExit(f"shard failed for anchor {name}")
        sc, df, gm = [], [], []
        for out in outs:
            blob = json.load(open(out))
            sc += blob["scores"]
            df += blob["diffs"]
            gm += blob.get("games") or []
            fb += blob["fallback"]
        os.makedirs(f"{league_dir(True)}/matches", exist_ok=True)
        json.dump({"a": "mortal298k", "b": name, "path_a": a.state,
                   "path_b": anchors[name]["path"], "deals": len(sc),
                   "seed0": a.seed0, "score_a": sum(sc), "unit": "hanchan",
                   "temp_a": 0.0, "temp_b": float(pool.get("temperature", 1.0)),
                   "mean_diff": sum(df) / len(df), "rows": [],
                   "games": gm},
                  open(f"{league_dir(True)}/matches/"
                       f"mortal298k_vs_{name}_{a.seed0}.json", "w"))
        games += [("cand", name, s) for s in sc]
        per_anchor[name] = {"pairs": len(sc), "share": round(sum(sc) / len(sc), 4),
                            "mean_diff": round(sum(df) / len(df), 1)}
        print(f"[match] mortal vs {name}: score {sum(sc):.1f}/{len(sc)} hanchan "
              f"mean_diff {per_anchor[name]['mean_diff']:+.0f} "
              f"({time.time()-t0:.0f}s, {a.shards} shards, fallback {fb})",
              flush=True)

    ratings = {n: anchors[n]["rating"] for n in names}
    ratings["cand"] = 1500.0
    fit_ratings(games, ratings, ["cand"])
    rec = {"ckpt": a.state, "label": a.label or f"H6{'T0' if a.tag else ''}_mortal298k",
           "elo": round(ratings["cand"], 1),
           "se": round(rating_se(games, ratings, "cand"), 1),
           "anchors": names, "deals_per_anchor": a.pairs, "seed0": a.seed0,
           "date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "engine": engine_fingerprint(),
           "hanchan_engine": have, "engine_mismatch": False,
           "temperature": "mortal-native-greedy",
           "anchor_temperature": float(pool.get("temperature", 1.0)),
           "pool": pool_file, "unit": "hanchan",
           "per_anchor": per_anchor, "bridge_fallbacks": fb,
           "residuals": residuals(games, ratings, "cand")}
    if a.force_temp is None:
        with open(f"{league_dir(True)}/history.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    else:
        rec["probe_forced_temperature"] = a.force_temp
        os.makedirs("experiments/probes", exist_ok=True)
        json.dump(rec, open("experiments/probes/"
                            f"exp56_mortal_vs_{'_'.join(names)}_T"
                            f"{a.force_temp:g}.json", "w"), indent=1)
    print(f"ELO {rec['label']}: {rec['elo']} ± {rec['se']}  "
          f"fallbacks {fb}  residuals {rec['residuals']}", flush=True)


if __name__ == "__main__":
    main()
