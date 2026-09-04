"""Elo anchor-league rating for DNN mahjong policies.

Design doc: experiments/designs/design_elo_league.md. Two modes:

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
# exp56: the hanchan-unit ladder lives in its own subtree. The two scales
# must never share anchors.json/history.jsonl — a hanchan Elo is ~1.8x a
# single-deal Elo, so mixing them silently corrupts both.
HANCHAN_DIR = f"{LEAGUE_DIR}/hanchan"
LN10_400 = math.log(10) / 400.0


def league_dir(hanchan: bool) -> str:
    return HANCHAN_DIR if hanchan else LEAGUE_DIR


def anchors_path(hanchan: bool, tag=None) -> str:
    """A pool is defined by its calibration condition, so a pool calibrated
    under a different one (e.g. all-greedy) gets its own file rather than
    overwriting the T=1 pool."""
    return f"{league_dir(hanchan)}/anchors{'_' + tag if tag else ''}.json"

ANCHOR_POOL = {
    "bc_cnn":      "experiments/arch_sweep/models/cnn_m.pt",
    "rf600":       "experiments/dnn_scratch_massive_20260815/games_600000.pt",
    "ppo44_240":   "experiments/dnn_ppo_massive_20260815/games_240000.pt",
    "ppo44_600":   "experiments/dnn_ppo_massive_20260815/games_600000.pt",
    "reuse11_600": "experiments/dnn_ppo_reuse1_20260815/games_600000.pt",
    "e700":        "experiments/dnn_exp12_E_20260816/games_700000.pt",
    "vit240":      "experiments/_cloud_ckpts/dnn_vit_rl_r4/games_240000.pt",
    "bcrl14_600":  "experiments/_cloud_ckpts/dnn_exp14_bcvit_rl_20260816/games_final.pt",
    # epoch-4 promotion (2026-08-23): batch-1 champion, first epoch-3-native anchor
    "exp27A_1M":   "experiments/_cloud_ckpts/dnn_exp27_A_cnn_m_r_20260823/games_final.pt",
    # epoch-6 promotions (2026-08-30): the human-prior lineage's frozen
    # milestones. Mixed action spaces are hosted natively now, so 46-slot
    # models can anchor alongside the legacy 374-slot pool — this turns
    # every modern rating from an extrapolation above the old top anchor
    # (1111) into an interpolation inside the pool.
    "bc49":        "experiments/_anchors_epoch6/bc49.pt",
    "bc51_v3r2":   "experiments/_anchors_epoch6/bc51_v3r2.pt",
    "exp46Cb":     "experiments/_anchors_epoch6/exp46Cb_gen10.pt",
    "exp46I":      "experiments/_anchors_epoch6/exp46I_gen10.pt",
}
PINNED = ("bc_cnn", 1000.0)   # scale origin, fixed forever

# Epoch rule (design_elo_league.md §长期维护 3): any change to the engine /
# scoring / rules invalidates every match played so far. The engine is
# fingerprinted by content (not git commit) so uncommitted edits count too.
ENGINE_FILES = ("table.py", "shanten.py", "claims.py", "wrapper.py", "arena.py")
# The hanchan ladder rides a second rule surface the engine files do not
# cover: the driver-side match state machine (renchan/honba/kyotaku/uma/
# nagashi). exp56 found exp53's readings were silently voided by two
# post-hoc rule fixes to it, so the hanchan scale gets its own epoch guard.
HANCHAN_FILES = ("hanchan.py",)


def engine_fingerprint(rev=None, files=ENGINE_FILES):
    """sha256 over the engine sources (working tree, or a git rev)."""
    import hashlib
    import subprocess
    h = hashlib.sha256()
    for f in files:
        rel = f"src/tasks/mahjong/{f}"
        if rev:
            blob = subprocess.run(["git", "show", f"{rev}:{rel}"],
                                  capture_output=True).stdout
        else:
            blob = open(rel, "rb").read() if os.path.exists(rel) else b""
        h.update(f.encode() + b"\0" + blob + b"\0")
    return h.hexdigest()[:16]


def engine_stamp(hanchan: bool = False):
    import subprocess
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    stamp = {"fingerprint": engine_fingerprint(), "git": git,
             "files": list(ENGINE_FILES)}
    if hanchan:
        stamp["hanchan_fingerprint"] = engine_fingerprint(files=HANCHAN_FILES)
        stamp["hanchan_files"] = list(HANCHAN_FILES)
    return stamp


def check_engine_epoch(league, allow):
    """Refuse to rate against anchors calibrated under a different engine."""
    want_h = league.get("engine", {}).get("hanchan_fingerprint")
    if want_h is not None and want_h != engine_fingerprint(files=HANCHAN_FILES):
        msg = (f"HANCHAN EPOCH MISMATCH: anchors calibrated under hanchan "
               f"driver {want_h}, current is "
               f"{engine_fingerprint(files=HANCHAN_FILES)} — match-level rules "
               f"(renchan/uma/nagashi) changed; recalibrate the hanchan pool.")
        if not allow:
            raise SystemExit(msg)
        print("WARN " + msg, flush=True)
    want = league.get("engine", {}).get("fingerprint")
    have = engine_fingerprint()
    if want is None:
        print("WARN anchors.json carries no engine stamp — cannot verify epoch",
              flush=True)
        return False
    if want == have:
        return False
    msg = (f"ENGINE EPOCH MISMATCH: anchors calibrated under engine {want} "
           f"(git {league['engine'].get('git')}), current engine is {have}. "
           f"Historical matches are void under the epoch rule — recalibrate "
           f"(`calibrate`) before rating.")
    if not allow:
        raise SystemExit(msg)
    print("WARN " + msg + " (--allow_engine_mismatch: rating anyway, flagged)",
          flush=True)
    return True


def deal_scores(rows):
    """Per-deal outcome for side A: 1 win / 0.5 tie / 0 loss by sign(diff)."""
    return [1.0 if r["diff"] > 0 else 0.0 if r["diff"] < 0 else 0.5
            for r in rows]


def expected(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def play_pair_vector(path_a, path_b, deals, seed0, parallel, device,
                     temp_a: float = 1.0, temp_b: float = 1.0,
                     hanchan: bool = False):
    """Vectorized duplicate match (perf 2026-08-30): reuses the trainer's
    batched-GPU rollout (collect_parallel arena mode) instead of the
    batch-1 per-move arena path — measured ~50x on the rating workload.
    Orientation rides the seed's low bit; wall seed = seed >> 1.
    Returns (per-deal scores for A, per-deal point diffs, per-GAME rows).
    The rows are the primary record — per-seat uma, placements, match
    length — so any other fit (per-match sign, uma margin, placement-only)
    can be redone later without replaying a single hand (exp56).

    hanchan=True (exp56) swaps the per-deal generator for the full-match
    one (exp55-D's four-seat rollout, here in 2v2 arena seating) and
    scores by uma instead of raw points: `deals` then counts MATCHES."""
    from src.agents.dnn.parallel_rollout import collect_parallel
    import torch as _t
    blob = _t.load(path_a, map_location="cpu")
    net = load_dnn(path_a, "cpu")
    cfg = dict(channels=blob.get("channels", 64), blocks=blob.get("blocks", 3),
               arch=blob.get("arch"),
               temperature=1.0, gamma=1.0, games_per_worker=16,
               rollout_temps=None, shaping=False, seed=seed0,
               hanchan=hanchan, hanchan_w_path=None,
               critic_feats="none", gpu_infer=True, gpu_infer_opponents=True,
               infer_max_batch=128, infer_wait_ms=0.0, infer_device=device,
               bf16_infer=False, arena=True, arena_temp_a=temp_a,
               arena_temp_b=temp_b,
               no_episodes=True, league_frac=1.0,
               league=[{"name": "B", "path": path_b}],
               encoder_variant=getattr(net, "encoder_variant", "v1"),
               action_space=getattr(net, "action_space", "native"),
               symmetrize=blob.get("symmetrize"))
    seeds = []
    for d in range(deals):
        w = seed0 + d
        seeds += [w * 2, w * 2 + 1]
    collect_parallel(net, len(seeds), cfg, parallel, seeds)
    games = collect_parallel.last_games
    per_wall = {}
    rows = []
    for g in games:
        w = g["seed"] >> 1
        if hanchan:
            # uma = final points - 25000 + placement bonus; sums to 0 over
            # the table, so A's total IS the match margin on the uma scale
            uma = (g.get("hanchan") or {}).get("uma_points")
            if not uma:
                raise SystemExit("hanchan arena game carried no uma_points — "
                                 "the hanchan generator did not run")
            pts = uma
        else:
            pts = g["points"]
        A = sorted(g["learner_seats"])
        a_pts = sum(pts[p] for p in A)
        b_pts = sum(pts[p] for p in range(4) if p not in A)
        rec = {"seed": g["seed"], "wall": g["seed"] >> 1, "a_seats": A,
               "a_pts": a_pts, "b_pts": b_pts}
        if hanchan:
            rec["placements"] = list(g["hanchan"]["placements"])
            rec["n_deals"] = g["hanchan"]["n_deals"]
            rec["busted"] = bool(g["hanchan"]["busted"])
            rec["uma"] = list(uma)
        rows.append(rec)
        pa, pb = per_wall.get(w, (0, 0))
        per_wall[w] = (pa + a_pts, pb + b_pts)
    scores, diffs = [], []
    for w in sorted(per_wall):
        pa, pb = per_wall[w]
        scores.append(1.0 if pa > pb else 0.0 if pa < pb else 0.5)
        diffs.append(float(pa - pb))
    return scores, diffs, rows


def play_pair(name_a, path_a, name_b, path_b, deals, seed0, parallel, device,
              temp_a: float = 1.0, legacy: bool = False,
              hanchan: bool = False, temp_b: float = 1.0):
    """One duplicate-deal match; returns per-deal scores for side A.
    temp_a: the candidate's sampling temperature (anchors always play at
    T=1, their calibration condition); 0 = greedy rating (exp28).
    hanchan=True scores full matches by uma (`deals` counts matches)."""
    t0 = time.time()
    # mixed action spaces are hosted natively since 2026-08-30 (server pads
    # the batch to the pool's max width), so the fast path is universal
    fast_ok = not legacy and str(device).startswith("cuda")
    if hanchan and not fast_ok:
        raise SystemExit("hanchan rating needs the vectorized CUDA path "
                         "(the batch-1 fallback is scripts/run_hanchan_arena.py)")
    if fast_ok:
        scores, diffs, games = play_pair_vector(path_a, path_b, deals, seed0,
                                                parallel, device, temp_a,
                                                temp_b, hanchan=hanchan)
        rows = [{"seed": seed0 + i, "diff": diffs[i]} for i in range(len(diffs))]
        mean = sum(diffs) / len(diffs)
    else:
        policies = {"A": load_dnn(path_a, device), "B": load_dnn(path_b, device)}
        seeds = [seed0 + i for i in range(deals)]
        rows = run_match(None, None, seeds, parallel=parallel,
                         dnn_policies=policies, dnn_device=device,
                         dnn_temperature={"A": temp_a, "B": 1.0})
        scores = deal_scores(rows)
        diffs = [r["diff"] for r in rows]
        mean = sum(diffs) / len(diffs)
        games = None
    out = {"a": name_a, "b": name_b, "path_a": path_a, "path_b": path_b,
           "deals": deals, "seed0": seed0, "score_a": sum(scores),
           "unit": "hanchan" if hanchan else "deal",
           "temp_a": temp_a, "temp_b": temp_b,
           "mean_diff": mean, "elapsed_s": round(time.time() - t0, 1),
           "rows": rows, "games": games}
    d = league_dir(hanchan)
    os.makedirs(f"{d}/matches", exist_ok=True)
    fn = f"{d}/matches/{name_a}_vs_{name_b}_{seed0}.json"
    json.dump(out, open(fn, "w"))
    print(f"[match] {name_a} vs {name_b}: score {sum(scores):.1f}/{deals} "
          f"{'hanchan' if hanchan else 'deals'} mean_diff {mean:+.0f}  "
          f"({out['elapsed_s']}s)", flush=True)
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
    hanchan = bool(getattr(args, "hanchan", False))
    temp = float(getattr(args, "temperature", 1.0) or 0.0)
    names = list(ANCHOR_POOL)
    games = []
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            scores = play_pair(na, ANCHOR_POOL[na], nb, ANCHOR_POOL[nb],
                               args.deals, args.seed0, args.parallel, device,
                               hanchan=hanchan, temp_a=temp, temp_b=temp)
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
    d = league_dir(hanchan)
    os.makedirs(d, exist_ok=True)
    fn = anchors_path(hanchan, getattr(args, "tag", None))
    # the calibration temperature IS part of the scale definition: rate()
    # reads it back so a candidate can never silently meet anchors in a
    # condition other than the one they were calibrated under
    json.dump({"pinned": PINNED, "deals_per_pair": args.deals,
               "unit": "hanchan" if hanchan else "deal",
               "temperature": temp, "seed0": args.seed0,
               "date": args.date or time.strftime("%Y-%m-%d %H:%M:%S"),
               "engine": engine_stamp(hanchan), "anchors": table},
              open(fn, "w"), indent=1)
    for n in sorted(names, key=lambda x: -ratings[x]):
        print(f"{n:>12}  {ratings[n]:7.1f} ± {table[n]['se']:.1f}")
    print(f"saved {fn}")


def rate_checkpoint(ckpt, label, deals, seed0, parallel, device,
                    use=None, init_guess=1000.0, allow_engine_mismatch=False,
                    temperature: float = 1.0, hanchan: bool = False,
                    anchor_temperature=None, tag=None):
    """Rate one checkpoint against frozen anchors; append to history.jsonl.

    use: anchor-name subset; None = all. init_guess seeds the fit and (in
    the ladder watcher) drives nearest-anchor selection upstream.
    """
    d = league_dir(hanchan)
    fn = anchors_path(hanchan, tag)
    league = json.load(open(fn))
    # anchors play in their calibration condition unless explicitly forced
    if anchor_temperature is None:
        anchor_temperature = float(league.get("temperature", 1.0))
    mismatch = check_engine_epoch(league, allow_engine_mismatch)
    anchors = league["anchors"]
    use = use or list(anchors)
    games = []
    for n in use:
        scores = play_pair("cand", ckpt, n, anchors[n]["path"],
                           deals, seed0, parallel, device, temp_a=temperature,
                           hanchan=hanchan, temp_b=anchor_temperature)
        games += [("cand", n, s) for s in scores]
    ratings = {n: anchors[n]["rating"] for n in use}
    ratings["cand"] = init_guess
    fit_ratings(games, ratings, ["cand"])
    # 0/100% scores make the MLE unbounded; clamp to anchor range ±600
    # (≈97% expected score) and flag it so curves show a floor, not garbage.
    lo = min(r for n, r in ratings.items() if n != "cand") - 600
    hi = max(r for n, r in ratings.items() if n != "cand") + 600
    at_bound = not (lo <= ratings["cand"] <= hi)
    ratings["cand"] = max(lo, min(hi, ratings["cand"]))
    rec = {"ckpt": ckpt, "label": label,
           "elo": round(ratings["cand"], 1), "at_bound": at_bound,
           "se": round(rating_se(games, ratings, "cand"), 1),
           "anchors": use, "deals_per_anchor": deals, "seed0": seed0,
           "date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "engine": engine_fingerprint(),
           "hanchan_engine": (engine_fingerprint(files=HANCHAN_FILES)
                              if hanchan else None),
           "engine_mismatch": mismatch,
           "temperature": temperature, "anchor_temperature": anchor_temperature,
           "pool": fn, "unit": "hanchan" if hanchan else "deal",
           "residuals": residuals(games, ratings, "cand")}
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/history.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"ELO {label}: {rec['elo']} ± {rec['se']}  residuals {rec['residuals']}",
          flush=True)
    return rec


def cmd_rate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use = args.anchors.split(",") if args.anchors else None
    rate_checkpoint(args.ckpt, args.label or args.ckpt, args.deals,
                    args.seed0, args.parallel, device, use=use,
                    allow_engine_mismatch=args.allow_engine_mismatch,
                    temperature=args.temperature,
                    hanchan=bool(getattr(args, "hanchan", False)),
                    anchor_temperature=args.anchor_temperature,
                    tag=args.tag)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ca = sub.add_parser("calibrate")
    ca.add_argument("--deals", type=int, default=200)
    ca.add_argument("--seed0", type=int, default=20260816)
    ca.add_argument("--parallel", type=int, default=20)
    ca.add_argument("--date", default=None,
                    help="label written into anchors.json (default: now)")
    ca.add_argument("--temperature", type=float, default=1.0,
                    help="calibration temperature for EVERY pool member "
                         "(0 = an all-greedy, deployment-form pool). Stored "
                         "in anchors.json and reused by rate")
    ca.add_argument("--tag", default=None,
                    help="write anchors_<tag>.json instead of anchors.json")
    ca.add_argument("--hanchan", action="store_true",
                    help="exp56: rate on the hanchan/uma scale (--deals then "
                         "counts full matches); writes the parallel ladder "
                         "under experiments/elo_league/hanchan/")
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
    ra.add_argument("--temperature", type=float, default=1.0,
                    help="candidate sampling temperature (anchors stay at T=1); 0 = greedy")
    ra.add_argument("--anchor_temperature", type=float, default=None,
                    help="override the anchor temperature. Default: whatever "
                         "the pool was calibrated at (read from anchors.json) "
                         "— overriding it takes the candidate OFF the pool's "
                         "calibrated scale")
    ra.add_argument("--tag", default=None,
                    help="rate against the pool anchors_<tag>.json "
                         "(e.g. T0 for the all-greedy pool)")
    ra.add_argument("--hanchan", action="store_true",
                    help="exp56: hanchan/uma scale (--deals counts matches)")
    ra.add_argument("--allow_engine_mismatch", action="store_true",
                    help="rate even though the engine changed since calibration "
                         "(record is flagged engine_mismatch=true)")
    ra.set_defaults(fn=cmd_rate)
    st = sub.add_parser("stamp", help="print the current engine fingerprint")
    st.set_defaults(fn=lambda a: print(json.dumps(engine_stamp())))
    args = ap.parse_args()
    args.date = time.strftime("%Y-%m-%d %H:%M:%S")
    args.fn(args)


if __name__ == "__main__":
    main()
