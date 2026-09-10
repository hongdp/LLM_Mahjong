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
    env = riichi_rs.VecEnv([int(s) for s in seeds], k, float(cfg.get("gamma", 0.995)),
                           bool(cfg.get("shaping", False)), float(cfg.get("shaping_scale", 1.0)),
                           True, variant)
    temperature = float(cfg.get("temperature", 1.0))
    q = float(riichi_rs.PLANE_Q)
    dev = torch.device(device)
    games = []
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
                idx, lp = net.act(P, S, M, temperature=temperature, check=False)   # legality checked on host
                env.step(idx.cpu().numpy().astype(np.int64).tolist(), lp.float().cpu().numpy().tolist())
            else:
                env.step([], [])
            games.extend(env.drain_finished())
    episodes, results = [], []
    agg = new_agg()
    for g in games:
        for e in g["episodes"]:
            e["planes_log"] = None      # planes stay uint8 (PLANE_Q); widened in the trainer
        episodes.extend(g["episodes"])
        results.append(g["result"])
        add_game(agg, g["result"], g.get("riichi"), g.get("n_melds"), g.get("n_discards"),
                 seats=range(4), points=g.get("points"), start_points=g.get("start_points"))
    collect_rust.last_style = agg
    collect_rust.last_games = games
    collect_rust.last_league = {}
    collect_rust.last_hanchan = {}
    collect_rust.last_cf_n = 0
    collect_rust.last_cf_skipped = 0
    return episodes, results
