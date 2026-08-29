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
import time
import random
from typing import List, Optional

import multiprocessing as mp

import numpy as np
import torch
from src.agents.dnn.encoder import encode_state, legal_mask

from src.agents.dnn.net import MahjongPolicyNet
from src.agents.dnn.selfplay import DnnStep, critic_features, apply_shaping, play_game, returns_to_go
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
    K = int(cfg.get("games_per_worker", 1) or 1)
    if cfg.get("gpu_infer") and K > 1:
        return _worker_vectorized(rank, n_games, seeds, cfg, net, pool_nets, cmode, K)
    for i in range(n_games):
        seed = seeds[i] if seeds else None
        learner_seats, opp = league_plan(seed, cfg)
        seat_nets = {pid: pool_nets[j] for pid, j in opp.items()} if opp else None
        temps = cfg.get("rollout_temps")
        if temps:
            # mixed temperatures (exp28): each seat draws its T from the list,
            # deterministic in the deal seed + replica so dup_k replicas differ
            rng = random.Random((seed or 0) * 7919 + i)
            temperature = {p: rng.choice(temps) for p in range(4)}
        else:
            temperature = cfg["temperature"]
        g = play_game(net, temperature=temperature, device="cpu",
                      deal_seed=seed, shaping=cfg["shaping"],
                      critic_feats=cmode, seat_nets=seat_nets)
        payload.append(_package_game(g, learner_seats, seed, cfg, cmode, bool(opp)))
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
        from src.agents.dnn.encoder import variant_shape, variant_of_arch
        variant = getattr(net, "encoder_variant", "v1")
        cfg = dict(cfg, encoder_variant=variant)
        if cfg.get("league") and cfg.get("gpu_infer_opponents"):
            # tag each pool entry with its encoder variant (from its checkpoint)
            tagged = []
            for entry in cfg["league"]:
                blob = torch.load(entry["path"], map_location="cpu")
                arch = blob.get("arch") or ""
                tagged.append(dict(entry, encoder_variant=variant_of_arch(arch)))
            cfg["league"] = tagged
        n_pl, n_sc = variant_shape(variant)
        K = int(cfg.get("games_per_worker", 1) or 1)
        server = InferenceServer(state_np, cfg, n_slots=workers, n_planes=n_pl,
                                 n_scalars=n_sc, device=cfg.get("infer_device", "cuda"),
                                 max_batch=cfg.get("infer_max_batch", 256),
                                 wait_ms=cfg.get("infer_wait_ms", 4.0),
                                 rows_per_worker=3 * K if K > 1 else 1)
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
    # style facts of this iteration's games (learner seats only), for TB
    from src.agents.dnn.style_stats import new_agg, add_game
    agg = new_agg()
    for game in collected:
        add_game(agg, game["result"], game.get("riichi"), game.get("n_melds"),
                 game.get("n_discards"), seats=game.get("learner_seats") or range(4),
                 points=game.get("points"), start_points=game.get("start_points"))
    collect_parallel.last_style = agg
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


def _package_game(g, learner_seats, seed, cfg, cmode, league):
    """Compact numpy episodes for one finished game (both worker paths)."""
    labels = completion_labels(g.result or "") if cmode == "hazard" else None
    eps = []
    for pid in range(4):
        if pid not in learner_seats:
            continue                       # opponents' trajectories are not ours
        steps = g.trajectories[pid]
        if not steps:
            continue
        if any(getattr(st, "extra_steps", None) for st in steps):
            expanded = []
            for st in steps:
                expanded.extend(getattr(st, "extra_steps", None) or [])
                st.extra_steps = None
                expanded.append(st)
            steps = expanded
        if cfg["shaping"]:
            apply_shaping(steps, cfg["gamma"])
        rets = returns_to_go(steps, cfg["gamma"])
        # ship compact tensors, not the whole step objects
        # numpy on the wire: torch tensors travel via shared-memory
        # file descriptors, which is fragile across fork + conda run
        # (observed: SocketClient FileNotFoundError). Arrays pickle.
        # perf/memory 2026-08-25: observations dominate the episode payload
        # (measured 11.1 MB per game for the 934-plane Mortal obs vs 0.3 MB for
        # our 21-plane one -> ~23 GB per 2048-game iteration, which OOM-killed
        # exp41 arm B). Ship them as float16: every plane is a probability,
        # indicator or rescaled ratio in [0, 1], and the update forward runs
        # under bf16 autocast anyway, so fp16 storage is lossless in effect and
        # halves both the pickle traffic and the trainer's resident tensor.
        logs = [st.sparse_planes for st in steps]
        use_log = all(l is not None for l in logs)
        ep = {
            "planes": (None if use_log else
                       torch.stack([s.planes for s in steps]).numpy().astype(np.float16)),
            "planes_log": logs if use_log else None,
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
    return {"episodes": eps, "result": g.result or "", "league": league,
            "riichi": list(g.riichi or []), "n_melds": list(g.n_melds or []),
            "n_discards": g.n_discards, "learner_seats": sorted(learner_seats),
            "points": list(g.points or []), "start_points": list(g.start_points or [])}


def _worker_vectorized(rank, n_games, seeds, cfg, net, pool_nets, cmode, K):
    """perf 2026-08-23: K games interleaved per process; every round the
    pending decisions of all K games go to the GPU server in ONE batched
    RPC (rows = K * 3 >= max requests per round). Hides the RPC latency
    that left single-game workers idle ~80% of the time and gives the
    server larger batches. Game logic: play_game_gen == play_game."""
    from src.agents.dnn.selfplay import play_game_gen, make_step
    from src.agents.dnn.action_space import get_space, REGISTRY
    _space = REGISTRY.get(cfg.get("action_space") or "native", REGISTRY["native"])
    variant = cfg.get("encoder_variant", "v1")
    pool_variant = {j + 1: e.get("encoder_variant", "v1") for j, e in enumerate(cfg.get("league") or [])}
    temps_list = cfg.get("rollout_temps")
    payload = []
    queue = list(range(n_games))
    active = {}            # game idx -> dict(gen, pending, seed, learner, opp, temps)
    _DIAG = {"rpc": 0.0, "rounds": 0, "rows": 0, "t0": time.perf_counter()}

    def start(i):
        seed = seeds[i] if seeds else None
        learner_seats, opp = league_plan(seed, cfg)
        if temps_list:
            rng = random.Random((seed or 0) * 7919 + i)
            temps = {p: rng.choice(temps_list) for p in range(4)}
        else:
            temps = {p: cfg["temperature"] for p in range(4)}
        gen = play_game_gen(deal_seed=seed, shaping=cfg["shaping"])
        try:
            table, reqs = next(gen)
        except StopIteration as e:
            payload.append(_package_game(e.value, learner_seats, seed, cfg, cmode, bool(opp)))
            return
        active[i] = {"gen": gen, "table": table, "reqs": reqs, "seed": seed,
                     "learner": learner_seats, "opp": opp, "temps": temps}

    while queue and len(active) < K:
        start(queue.pop(0))
    while active:
        # gather every pending request across the active games
        rows = []          # (game idx, req idx, pid, actions, model_id, variant)
        for gi, st in active.items():
            for ri, (pid, actions) in enumerate(st["reqs"]):
                mid = (st["opp"] or {}).get(pid)
                model_id = 0 if mid is None else mid + 1
                rows.append((gi, ri, pid, actions, model_id, pool_variant.get(model_id, variant)))
        planes, scalars, masks, lookups, temps, mids = [], [], [], [], [], []
        sparse_logs = []
        for gi, ri, pid, actions, model_id, var in rows:
            st = active[gi]
            sp_log = None
            if var in ("mortal_v3", "mortal_v3_pure"):
                # encode ONCE into the compact log, then densify locally for
                # the RPC (the shared-memory protocol still wants a dense row).
                # The log is what gets stored and shipped afterwards.
                from src.agents.dnn.mortal_obs import (encode_mortal_obs_sparse,
                                                       densify)
                sp_log = encode_mortal_obs_sparse(
                    st["table"], pid, derived=(var == "mortal_v3"))
                pl = densify([sp_log])[0]
                _, sc = encode_state(st["table"], pid, variant="v1")
            else:
                pl, sc = encode_state(st["table"], pid, variant=var)
            sparse_logs.append(sp_log)
            mask, lookup = _space.mask(actions)
            if os.environ.get("INFER_DEBUG") and not bool(mask.any()):
                with open("/tmp/vec_debug.txt", "a") as _f:
                    _f.write(f"EMPTY vec mask pid={pid} actions={actions!r}\n")
            planes.append(pl); scalars.append(sc); masks.append(mask); lookups.append(lookup)
            temps.append(float(st["temps"][pid])); mids.append(model_id)
        maxp = max(p.shape[0] for p in planes)
        P = torch.zeros(len(rows), maxp, planes[0].shape[1])
        for k, pl in enumerate(planes):
            P[k, :pl.shape[0]] = pl
        maxs = max(sc.shape[0] for sc in scalars)
        S = torch.zeros(len(rows), maxs)
        for k, sc in enumerate(scalars):
            S[k, :sc.shape[0]] = sc
        M = torch.stack(masks)
        _t0 = time.perf_counter()
        idx, lp = net.act_batch(P, S, M, torch.tensor(temps), torch.tensor(mids, dtype=torch.int32))
        _DIAG["rpc"] += time.perf_counter() - _t0; _DIAG["rounds"] += 1; _DIAG["rows"] += len(rows)
        if rank == 0 and _DIAG["rounds"] % 200 == 0 and os.environ.get("INFER_DIAG"):
            print(f"[worker0] rounds {_DIAG['rounds']} rows/round {_DIAG['rows']/_DIAG['rounds']:.1f} "
                  f"rpc {_DIAG['rpc']/_DIAG['rounds']*1000:.2f} ms/round wall {(time.perf_counter()-_DIAG['t0'])/_DIAG['rounds']*1000:.2f} ms/round", flush=True)
        # Multi-step action spaces (Mortal's declare-then-choose riichi/kan)
        # need a SECOND batched round for the rows that asked for a follow-up.
        # Native never sets `pending`, so this whole block is skipped and the
        # single-round behaviour is bit-for-bit what it always was.
        first_steps = [
            DnnStep(planes=planes[k], scalars=scalars[k], mask=masks[k],
                    action_idx=int(idx[k]), logprob=float(lp[k]),
                    cfeats=critic_features(active[rows[k][0]]["table"], rows[k][2], cmode),
                    sparse_planes=sparse_logs[k])
            for k in range(len(rows))
        ]
        pending = []
        for k, (gi, ri, pid, actions, model_id, var) in enumerate(rows):
            mode = _space.follow_up(int(idx[k]), actions)
            if mode is not None:
                pending.append((k, mode))
        follow = {}
        if pending:
            f_masks, f_lookups = [], []
            for k, mode in pending:
                m2, lk2 = _space.mask(rows[k][3], mode=mode)
                if os.environ.get("INFER_DEBUG") and not bool(m2.any()):
                    with open("/tmp/vec_debug.txt", "a") as _f:
                        _f.write(f"EMPTY vec FOLLOWUP mode={mode} actions={rows[k][3]!r}\n")
                f_masks.append(m2); f_lookups.append(lk2)
            P2 = torch.stack([planes[k] for k, _ in pending])
            S2 = torch.stack([scalars[k] for k, _ in pending])
            if P2.shape[1] < maxp:
                P2 = torch.nn.functional.pad(P2, (0, 0, 0, maxp - P2.shape[1]))
            idx2, lp2 = net.act_batch(
                P2, S2, torch.stack(f_masks),
                torch.tensor([temps[k] for k, _ in pending]),
                torch.tensor([mids[k] for k, _ in pending], dtype=torch.int32))
            for j, (k, _mode) in enumerate(pending):
                gi, ri, pid, actions, model_id, var = rows[k]
                a2 = int(idx2[j])
                follow[k] = (
                    DnnStep(planes=planes[k], scalars=scalars[k], mask=f_masks[j],
                            action_idx=a2, logprob=float(lp2[j]),
                            cfeats=critic_features(active[gi]["table"], pid, cmode),
                            sparse_planes=sparse_logs[k]),
                    f_lookups[j][a2])

        # distribute replies per game and advance each generator
        replies = {gi: [None] * len(active[gi]["reqs"]) for gi in active}
        for k, (gi, ri, pid, actions, model_id, var) in enumerate(rows):
            if k in follow:
                step2, action_str = follow[k]
                # the generator contract carries ONE step per request, so the
                # declaration step rides along and is unpacked by _package_game
                step2.extra_steps = [first_steps[k]]
                replies[gi][ri] = (step2, action_str)
            else:
                replies[gi][ri] = (first_steps[k], lookups[k][int(idx[k])])
        for gi in list(active):
            st = active[gi]
            try:
                st["table"], st["reqs"] = st["gen"].send(replies[gi])
            except StopIteration as e:
                payload.append(_package_game(e.value, st["learner"], st["seed"], cfg, cmode, bool(st["opp"])))
                del active[gi]
                if queue:
                    start(queue.pop(0))
    return payload
