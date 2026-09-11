"""Self-play rollout on the Rust engine (riichi_rs.VecEnv) — a drop-in for
parallel_rollout.collect_parallel in the mirror single-deal setting.

The whole game loop (rules, encoding, batching) runs in Rust; Python only
runs the policy forward on the batched observations and samples actions.
Episode payloads match _package_game except that planes ship as uint8
quantised by riichi_rs.PLANE_Q (every encoder plane value lies on the k/20
grid); the trainer widens them on the device with `u8.float() / PLANE_Q`,
which reproduces the encoder's float32 values bit-exactly. Cuts the
memory-bound observe/h2d/drain phases 4x (v3r planes 7.6 KB -> 1.9 KB/row).
"""
from typing import List, Optional

import numpy as np
import torch

from src.agents.dnn.style_stats import add_game, new_agg


def widen_planes(t: torch.Tensor) -> torch.Tensor:
    """Episode planes -> float32 on their device: uint8 (collect_rust, PLANE_Q grid)
    are dequantised by division (bit-exact with the encoder); float16/32 pass through."""
    if t.dtype == torch.uint8:
        import riichi_rs
        return t.float().div_(float(riichi_rs.PLANE_Q))
    return t.float()


def collect_rust(net, n_games: int, cfg: dict, workers: int, seeds: Optional[List[int]] = None,
                 device: str = "cuda"):
    """Returns (episodes, results) like collect_parallel. `workers` is unused
    (one process); cfg keys used: temperature, gamma, shaping, shaping_scale,
    games_per_worker (concurrent games K = games_per_worker * workers)."""
    import riichi_rs
    if seeds is None:
        seeds = [6_000_000 + i for i in range(n_games)]
    k = max(1, int(cfg.get("games_per_worker", 32)) * max(1, workers))
    variant = cfg.get("encoder_variant") or getattr(net, "encoder_variant", "v1r")
    if variant not in ("v1r", "v3r"):
        raise SystemExit(f"collect_rust: encoder variant {variant!r} not ported (v1r/v3r only)")
    # hanchan (exp83+): four learner copies over a full match (play_hanchan_gen /
    # --hanchan_pure), credit=none only; `n_games` then counts MATCHES.
    hanchan = bool(cfg.get("hanchan"))
    if hanchan:
        if not cfg.get("hanchan_pure"):
            raise SystemExit("collect_rust: only --hanchan_pure (mirror) is ported; four-seat league tables need the Python engine")
        if cfg.get("hanchan_credit", "none") != "none":
            raise SystemExit("collect_rust: hanchan_credit must be 'none' on the Rust engine (rank/W credits are Python-only)")
        if cfg.get("shaping"):
            raise SystemExit("collect_rust: PBRS shaping is per-deal and not supported across a hanchan")
        if cfg.get("league") and float(cfg.get("league_frac", 0.0) or 0.0) > 0:
            raise SystemExit("collect_rust: league pool + hanchan not supported")
    env = riichi_rs.VecEnv([int(s) for s in seeds], k, float(cfg.get("gamma", 0.995)),
                           bool(cfg.get("shaping", False)), float(cfg.get("shaping_scale", 1.0)),
                           True, variant, hanchan=hanchan, max_deals=int(cfg.get("hanchan_max_deals", 24)))
    temperature = float(cfg.get("temperature", 1.0))
    q = float(riichi_rs.PLANE_Q)
    dev = torch.device(device)
    games = []
    # league (exp82, 2026-09-10): frozen pool nets fill the non-learner seats of a
    # seed-deterministic fraction of deals (same league_plan as collect_parallel, so
    # dup replicas of a deal share the seat assignment and the group baseline stays
    # like-with-like). Only learner-seat episodes are returned.
    pool = list(cfg.get("league") or []) if float(cfg.get("league_frac", 0.0) or 0.0) > 0 else []
    pool_nets, plan_arr, seed_index, plans = [], None, {}, {}
    if pool:
        from src.agents.dnn.parallel_rollout import _load_policy_ckpt, league_plan
        for e in pool:
            pn = _load_policy_ckpt(e["path"]).to(dev)
            pv = getattr(pn, "encoder_variant", "v1r")
            if pv != variant:
                raise SystemExit(f"collect_rust league: pool {e.get('name')} encoder {pv!r} != learner {variant!r} (one VecEnv encoding)")
            pool_nets.append(pn)
        seed_index = {s: i for i, s in enumerate(seeds)}
        plan_arr = np.full((len(seeds), 4), -1, dtype=np.int64)       # seat -> pool idx, -1 = learner
        for s in seeds:
            ls, opp = league_plan(s, cfg)
            plans[s] = (ls, opp)
            for seat, j in opp.items():
                plan_arr[seed_index[s], seat] = j
        opp_temp = cfg.get("league_opp_temp")
        opp_temp = temperature if opp_temp is None else float(opp_temp)
    # perf note 2026-09-09: splitting K over two VecEnvs to overlap the GPU forward
    # with the other env's CPU step was measured 20-30% SLOWER (halved batch per
    # forward, doubled per-round overhead; the forward is only ~30% of a round),
    # and per-call pinned uploads cost more than they save. Single env it stays.
    with torch.no_grad():
        while not env.done():
            planes, scalars, mask, seats, gids = env.observe()
            n = planes.shape[0]
            if n > 0:
                if not mask.any(axis=1).all():
                    raise ValueError("collect_rust: VecEnv produced a row with no legal actions")
                P = torch.from_numpy(planes).to(dev).view(n, -1, 34).float().div_(q)
                S = torch.from_numpy(scalars).to(dev)
                M = torch.from_numpy(mask).to(dev)
                if not pool:
                    idx, lp = net.act(P, S, M, temperature=temperature, check=False)   # legality checked on host
                    acts_np, lps_np = idx.cpu().numpy().astype(np.int64), lp.float().cpu().numpy()
                else:
                    slot_seed = np.asarray(env.slot_seeds(), dtype=np.int64)
                    sidx = np.fromiter((seed_index[int(s)] if s >= 0 else 0 for s in slot_seed), dtype=np.int64, count=len(slot_seed))
                    grp = plan_arr[sidx[np.asarray(gids, dtype=np.int64)], np.asarray(seats, dtype=np.int64)]
                    acts_np = np.zeros(n, dtype=np.int64); lps_np = np.zeros(n, dtype=np.float32)
                    for j in np.unique(grp):
                        sel = np.nonzero(grp == j)[0]
                        tsel = torch.from_numpy(sel).to(dev)
                        who = net if j < 0 else pool_nets[int(j)]
                        idx, lp = who.act(P[tsel], S[tsel], M[tsel], temperature=temperature if j < 0 else opp_temp, check=False)
                        acts_np[sel] = idx.cpu().numpy(); lps_np[sel] = lp.float().cpu().numpy()
                env.step(acts_np.tolist(), lps_np.tolist())
            else:
                env.step([], [])
            games.extend(env.drain_finished())
    episodes, results = [], []
    agg = new_agg()
    lg = {}
    pool_names = [e.get("name", str(j)) for j, e in enumerate(pool)]
    for g in games:
        ls, opp = plans.get(int(g["seed"]), (list(range(4)), {})) if pool else (list(range(4)), {})
        if opp:
            g["episodes"] = [e for e in g["episodes"] if int(e["key"][1]) in ls]
            g["learner_seats"] = list(ls)
            g["league"] = dict(opp)
            pts = g.get("points") or []
            for L in ls:
                for seat, j in opp.items():
                    name = pool_names[j]
                    w, nn, d = lg.get(name, (0.0, 0, 0.0))
                    sc = 1.0 if pts[L] > pts[seat] else 0.0 if pts[L] < pts[seat] else 0.5
                    lg[name] = (w + sc, nn + 1, d + (pts[L] - pts[seat]))
        for e in g["episodes"]:
            e["planes_log"] = None      # planes stay uint8 (PLANE_Q); widened in the trainer
        episodes.extend(g["episodes"])
        results.append(g["result"])
        add_game(agg, g["result"], g.get("riichi"), g.get("n_melds"), g.get("n_discards"),
                 seats=ls, points=g.get("points"), start_points=g.get("start_points"))
    collect_rust.last_style = agg
    collect_rust.last_games = games
    collect_rust.last_league = {k: {"learner_share": round(w / nn, 4), "n": nn, "mean_diff": round(d / nn, 1)}
                                for k, (w, nn, d) in lg.items()}
    hz = {}
    for g in games:
        h = g.get("hanchan")
        if h:
            # exp70/83 pure mirror: placement spread (mean |uma|) is the only meaningful statistic
            u, nn = hz.get("pure_abs_uma", (0.0, 0))
            hz["pure_abs_uma"] = (u + sum(abs(x) for x in h["uma_points"]) / 4.0, nn + 1)
    collect_rust.last_hanchan = {k: {"mean_uma": round(u / nn, 1), "n": nn} for k, (u, nn) in hz.items()}
    collect_rust.last_cf_n = 0
    collect_rust.last_cf_skipped = 0
    return episodes, results
