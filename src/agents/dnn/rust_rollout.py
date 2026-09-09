"""Self-play rollout on the Rust engine (riichi_rs.VecEnv) — a drop-in for
parallel_rollout.collect_parallel in the mirror single-deal setting.

The whole game loop (rules, encoding, batching) runs in Rust; Python only
runs the policy forward on the batched observations and samples actions.
Episode payloads match _package_game so the PPO trainer is unchanged.
"""
from typing import List, Optional

import numpy as np
import torch

from src.agents.dnn.style_stats import add_game, new_agg


def collect_rust(net, n_games: int, cfg: dict, workers: int, seeds: Optional[List[int]] = None,
                 device: str = "cuda"):
    """Returns (episodes, results) like collect_parallel. `workers` is unused
    (one process); cfg keys used: temperature, gamma, shaping, shaping_scale,
    games_per_worker (concurrent games K = games_per_worker * workers)."""
    import riichi_rs
    if seeds is None:
        seeds = [6_000_000 + i for i in range(n_games)]
    k = max(1, int(cfg.get("games_per_worker", 32)) * max(1, workers))
    env = riichi_rs.VecEnv([int(s) for s in seeds], k, float(cfg.get("gamma", 0.995)),
                           bool(cfg.get("shaping", False)), float(cfg.get("shaping_scale", 1.0)),
                           True)
    temperature = float(cfg.get("temperature", 1.0))
    dev = torch.device(device)
    games = []
    with torch.no_grad():
        while not env.done():
            planes, scalars, mask, seats, gids = env.observe()
            n = planes.shape[0]
            if n > 0:
                P = torch.from_numpy(planes).to(dev).view(n, -1, 34)
                S = torch.from_numpy(scalars).to(dev)
                M = torch.from_numpy(mask).to(dev)
                idx, lp = net.act(P, S, M, temperature=temperature)
                env.step(idx.cpu().numpy().astype(np.int64).tolist(), lp.float().cpu().numpy().tolist())
            else:
                env.step([], [])
            games.extend(env.drain_finished())
    episodes, results = [], []
    agg = new_agg()
    for g in games:
        for e in g["episodes"]:
            e["planes"] = e["planes"].astype(np.float16)     # _package_game ships fp16
            e["planes_log"] = None
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
