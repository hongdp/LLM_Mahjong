"""Vectorised GPU arena on top of the trainer's batched rollout
(`parallel_rollout.collect_parallel`, arena / table modes).

Two drivers share the rollout configuration built here:

- `play_pair_vector` — 2v2 duplicate match (single deal or full hanchan),
  the T=0 verdict protocol. Moved verbatim from scripts/run_elo_league.py
  (2026-11-01 structure clean-up); that script keeps a thin wrapper with
  the same name and signature because resident evaluators import it there.
- `src.eval.arena.play_table` — four-entity rotated table (rating v2).

`base_cfg` is the part both drivers agree on; each adds its own seating
keys on top, so the dicts handed to collect_parallel are byte-for-byte what
the two drivers built independently before the merge.
"""

import torch

from src.agents.dnn.checkpoint import load_dnn


def base_cfg(net, blob, device, seed0):
    """collect_parallel config common to every vectorised evaluator: batch
    GPU inference for the learner and every league opponent, no episodes
    returned, no shaping, the learner net's own encoder / action space."""
    return dict(channels=blob.get("channels", 64), blocks=blob.get("blocks", 3),
                arch=blob.get("arch"),
                temperature=1.0, gamma=1.0, games_per_worker=16,
                rollout_temps=None, shaping=False, seed=seed0,
                hanchan_w_path=None,
                critic_feats="none", gpu_infer=True, gpu_infer_opponents=True,
                infer_max_batch=128, infer_wait_ms=0.0, infer_device=device,
                bf16_infer=False,
                no_episodes=True, league_frac=1.0,
                encoder_variant=getattr(net, "encoder_variant", "v1"),
                action_space=getattr(net, "action_space", "native"))


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
    blob = torch.load(path_a, map_location="cpu")
    net = load_dnn(path_a, "cpu")
    cfg = base_cfg(net, blob, device, seed0)
    cfg.update(hanchan=hanchan,
               arena=True, arena_temp_a=temp_a, arena_temp_b=temp_b,
               league=[{"name": "B", "path": path_b}],
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
