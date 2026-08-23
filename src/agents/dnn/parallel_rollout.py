"""Multiprocess self-play collection for the DNN agent.

The DNN rollout is pure Python + small CPU tensors, so it is single-core
bound and trivially parallel across games. This module forks W workers,
each replaying its own slice of deal seeds against a snapshot of the
current weights, then merges the trajectories.

Why it matters: the whole reason to keep a conventional baseline around
is that it can afford experiments the LLM never could — e.g. measuring
how strength scales with the NUMBER of RL games. That needs throughput,
not a bigger model.

Each worker pins torch to 1 thread: 16 processes each spawning 24 BLAS
threads is catastrophically slower than 16 single-threaded ones.
"""

import os
import random
from typing import List, Optional

import multiprocessing as mp

import numpy as np
import torch

from src.agents.dnn.net import MahjongPolicyNet
from src.agents.dnn.selfplay import apply_shaping, play_game, returns_to_go
from src.agents.dnn.yaku_features import completion_labels


def _load_policy_ckpt(path):
    """Frozen opponent loader (league): rebuilds the net from the checkpoint's
    own arch tag and loads policy keys only (critic variants tolerated)."""
    from src.agents.dnn.net import load_compatible
    blob = torch.load(path, map_location="cpu")
    if blob.get("arch"):
        from src.agents.dnn.arch_zoo import ZOO
        net = ZOO[blob["arch"]][0]()
    else:
        net = MahjongPolicyNet(channels=blob.get("channels", 64),
                               blocks=blob.get("blocks", 3))
    load_compatible(net, blob["state_dict"])
    return net.eval()


def league_plan(seed, cfg):
    """Deterministic per-deal composition (exp22): all dup replicas of a deal
    share the SAME seat assignment so the (seed, seat) group-baseline key
    still compares like with like. Returns (learner_seats, {seat: pool_idx})."""
    pool = cfg.get("league") or []
    frac = float(cfg.get("league_frac", 0.0))
    if not pool or frac <= 0 or seed is None:
        return list(range(4)), {}
    rng = random.Random(int(seed) * 7919 + 17)
    if rng.random() >= frac:
        return list(range(4)), {}
    n_learner = rng.choice((1, 2))
    learner = sorted(rng.sample(range(4), n_learner))
    opp = {pid: rng.randrange(len(pool)) for pid in range(4) if pid not in learner}
    return learner, opp


def _worker(rank, n_games, seeds, state_np, cfg):
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"] * 1000 + rank)
    import random as _rnd
    _rnd.seed(cfg["seed"] * 7919 + rank)
    if cfg.get("gpu_infer"):
        # batched GPU inference (perf 2026-08-22): the server was started by
        # the parent before the fork; we only need a slot-bound shim
        from src.agents.dnn.infer_server import RemotePolicy
        net = RemotePolicy(rank, cfg.get("encoder_variant", "v1"))
    else:
        if cfg.get("arch"):
            from src.agents.dnn.arch_zoo import ZOO
            net = ZOO[cfg["arch"]][0]()
        else:
            net = MahjongPolicyNet(channels=cfg["channels"], blocks=cfg["blocks"])
        from src.agents.dnn.net import load_compatible
        load_compatible(net, {k: torch.from_numpy(v) for k, v in state_np.items()})
        net.eval()
    cmode = cfg.get("critic_feats", "none")
    pool_nets = {}
    if cfg.get("league"):
        if cfg.get("gpu_infer") and cfg.get("gpu_infer_opponents"):
            # opponents hosted on the GPU server too (model ids 1..K share
            # this worker's slot; a worker is single-threaded so sequential
            # use of one slot by several shims is safe)
            from src.agents.dnn.infer_server import RemotePolicy
            for j, entry in enumerate(cfg["league"]):
                pool_nets[j] = RemotePolicy(rank, entry.get("encoder_variant", "v1"),
                                            model_id=j + 1)
        else:
            # frozen opponents on CPU inside the worker (fine for cnn_m-class
            # nets; a 192x40 opponent at 17 ms/call would dominate)
            for j, entry in enumerate(cfg["league"]):
                pool_nets[j] = _load_policy_ckpt(entry["path"])
    payload = []
    for i in range(n_games):
        seed = seeds[i] if seeds else None
        learner_seats, opp = league_plan(seed, cfg)
        seat_nets = {pid: pool_nets[j] for pid, j in opp.items()} if opp else None
        g = play_game(net, temperature=cfg["temperature"], device="cpu",
                      deal_seed=seed, shaping=cfg["shaping"],
                      critic_feats=cmode, seat_nets=seat_nets)
        labels = completion_labels(g.result or "") if cmode == "hazard" else None
        eps = []
        for pid in range(4):
            if pid not in learner_seats:
                continue                       # opponents' trajectories are not ours
            steps = g.trajectories[pid]
            if not steps:
                continue
            if cfg["shaping"]:
                apply_shaping(steps, cfg["gamma"])
            rets = returns_to_go(steps, cfg["gamma"])
            # ship compact tensors, not the whole step objects
            # numpy on the wire: torch tensors travel via shared-memory
            # file descriptors, which is fragile across fork + conda run
            # (observed: SocketClient FileNotFoundError). Arrays pickle.
            ep = {
                "planes": torch.stack([s.planes for s in steps]).numpy(),
                "scalars": torch.stack([s.scalars for s in steps]).numpy(),
                "mask": torch.stack([s.mask for s in steps]).numpy(),
                "actions": np.array([s.action_idx for s in steps], dtype=np.int64),
                "old_logprobs": np.array([s.logprob for s in steps], dtype=np.float32),
                "returns": np.array(rets, dtype=np.float32),
                "rewards": np.array([s.reward for s in steps], dtype=np.float32),
                "key": (seed, pid),
            }
            if cmode != "none":
                ep["cfeats"] = torch.stack([s.cfeats for s in steps]).numpy()
            if labels is not None:
                # one settled-fact label vector per (game, seat); the trainer
                # broadcasts it over the episode's steps for the BCE channel
                ep["hlabels"] = np.array(labels[pid], dtype=np.float32)
            eps.append(ep)
        payload.append({"episodes": eps, "result": g.result or "",
                        "league": bool(opp)})
    return payload


def collect_parallel(net, n_games: int, cfg: dict, workers: int,
                     seeds: Optional[List[int]] = None):
    """Returns (episodes, results). Episodes carry per-step tensors and a
    (deal_seed, seat) key so the caller can apply the group baseline."""
    state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    per = [n_games // workers] * workers
    for i in range(n_games - sum(per)):
        per[i] += 1

    state_np = {k: v.numpy() for k, v in state_dict.items()}
    server = None
    if cfg.get("gpu_infer"):
        from src.agents.dnn.infer_server import InferenceServer
        from src.agents.dnn.encoder import (N_PLANES, N_PLANES_V3, N_SCALARS,
                                            N_SCALARS_V3)
        variant = getattr(net, "encoder_variant", "v1")
        cfg = dict(cfg, encoder_variant=variant)
        if cfg.get("league") and cfg.get("gpu_infer_opponents"):
            # tag each pool entry with its encoder variant (from its checkpoint)
            tagged = []
            for entry in cfg["league"]:
                blob = torch.load(entry["path"], map_location="cpu")
                arch = blob.get("arch") or ""
                tagged.append(dict(entry, encoder_variant="v3" if arch.endswith("_v3") else "v1"))
            cfg["league"] = tagged
        n_pl, n_sc = ((N_PLANES_V3, N_SCALARS_V3) if variant == "v3"
                      else (N_PLANES, N_SCALARS))
        server = InferenceServer(state_np, cfg, n_slots=workers, n_planes=n_pl,
                                 n_scalars=n_sc, device=cfg.get("infer_device", "cuda"),
                                 max_batch=cfg.get("infer_max_batch", 256),
                                 wait_ms=cfg.get("infer_wait_ms", 4.0))
        state_np = {}                      # workers don't need weights
    ctx = mp.get_context("fork")
    args, lo = [], 0
    for r in range(workers):
        if per[r] == 0:
            continue
        chunk = seeds[lo:lo + per[r]] if seeds else None
        lo += per[r]
        args.append((r, per[r], chunk, state_np, cfg))
    try:
        with ctx.Pool(len(args)) as pool:
            collected = []
            for payload in pool.starmap(_worker, args):
                collected.extend(payload)
    finally:
        if server is not None:
            server.stop()

    episodes, results = [], []
    for game in collected:
        episodes.extend(game["episodes"])
        results.append(game["result"])
    return episodes, results


def apply_group_baseline(episodes, gamma: float) -> None:
    """Leave-one-out group baseline over (deal_seed, seat) replicas, applied
    in place to each episode's `returns`. Same correction shape as the LLM
    path: spread over per-step rewards, never a flat episode constant."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, ep in enumerate(episodes):
        if ep["key"][0] is not None:
            groups[ep["key"]].append(i)
    for key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        g0 = [float(episodes[i]["returns"][0]) for i in idxs]
        total = sum(g0)
        for j, i in enumerate(idxs):
            loo = (total - g0[j]) / (len(idxs) - 1)
            rets = episodes[i]["returns"]
            n = len(rets)
            d = loo / n
            m = (n - np.arange(n, dtype=np.float32))
            tail = m if gamma == 1.0 else (1 - gamma ** m) / (1 - gamma)
            episodes[i]["returns"] = rets - d * tail
