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

import copy
import os
import re
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
    if blob.get("symmetrize"):
        from src.agents.dnn.symmetry import maybe_symmetrize
        net = maybe_symmetrize(net, blob["symmetrize"])
    return net.eval()


def _pad_stack(items):
    """Stack rows whose leading dim may differ (mixed encoder variants /
    action spaces since 2026-08-30): zero-pad every row up to the batch
    max. Masks pad with False, so padded slots are never legal."""
    mx = max(x.shape[0] for x in items)
    out = torch.zeros((len(items), mx) + tuple(items[0].shape[1:]),
                      dtype=items[0].dtype)
    for k, x in enumerate(items):
        out[k, :x.shape[0]] = x
    return out


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
    n_learner = int(cfg.get("league_learner_seats") or 0) or rng.choice((1, 2))
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
        if cfg.get("symmetrize"):
            from src.agents.dnn.symmetry import maybe_symmetrize
            net = maybe_symmetrize(net, cfg["symmetrize"])
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
        if opp and cfg.get("league_opp_temp") is not None:
            if not isinstance(temperature, dict):
                temperature = {p: temperature for p in range(4)}
            for pid in opp:
                temperature[pid] = cfg["league_opp_temp"]
        g = play_game(net, temperature=temperature, device="cpu",
                      deal_seed=seed, shaping=cfg["shaping"],
                      critic_feats=cmode, seat_nets=seat_nets)
        payload.append(_package_game(g, learner_seats, seed, cfg, cmode, opp))
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
                var = blob.get("encoder_variant") or variant_of_arch(blob.get("arch") or "")
                from src.agents.dnn.action_space import space_of_arch
                tagged.append(dict(entry, encoder_variant=var,
                                   action_space=space_of_arch(blob.get("arch") or "")))
            cfg["league"] = tagged
        n_pl, n_sc = variant_shape(variant)
        K = int(cfg.get("games_per_worker", 1) or 1)
        if cfg.get("cf_p") and not cfg.get("hanchan"):
            # exp72: branch rollouts are extra interleaved games; give them
            # their own slot budget (default: as many as the parents)
            cfg = dict(cfg, cf_branch_slots=int(cfg.get("cf_branch_slots") or K))
        B = int(cfg.get("cf_branch_slots") or 0)
        server = InferenceServer(state_np, cfg, n_slots=workers, n_planes=n_pl,
                                 n_scalars=n_sc, device=cfg.get("infer_device", "cuda"),
                                 max_batch=cfg.get("infer_max_batch", 256),
                                 wait_ms=cfg.get("infer_wait_ms", 4.0),
                                 rows_per_worker=3 * (K + B) if K > 1 else 1)
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
    n_cf = 0
    for game in collected:
        cfm = game.get("cf") or {}
        if cfm:
            for ep in game["episodes"]:
                recs = cfm.get(ep["key"][1])
                if not recs:
                    continue
                rets = ep["returns"]
                adv, fold = {}, {}
                for step, rec in recs.items():
                    alts = list(rec["alts"].values())
                    if not alts or step >= len(rets):
                        continue
                    # executed action's realised return minus the mean of the
                    # same-wall alternative continuations (baseline chosen
                    # independently of the executed action -> unbiased)
                    adv[int(step)] = float(rets[step] - sum(alts) / len(alts))
                    g = rec.get("genbutsu")
                    if g is not None and g in rec["alts"]:
                        fold[int(step)] = float(rec["alts"][g] - rets[step])
                ep["cf_adv"], ep["cf_fold_gain"] = adv, fold
                n_cf += len(adv)
        episodes.extend(game["episodes"])
        results.append(game["result"])
    collect_parallel.last_cf_n = n_cf
    collect_parallel.last_cf_skipped = sum(g.get("cf_skipped", 0) for g in collected)
    # style facts of this iteration's games (learner seats only), for TB
    from src.agents.dnn.style_stats import new_agg, add_game
    agg = new_agg()
    for game in collected:
        add_game(agg, game["result"], game.get("riichi"), game.get("n_melds"),
                 game.get("n_discards"), seats=game.get("learner_seats") or range(4),
                 points=game.get("points"), start_points=game.get("start_points"))
    collect_parallel.last_style = agg
    collect_parallel.last_games = collected
    # rollout ratings (exp46-C rev3): the learner is every pool member's
    # common opponent, so pairwise point-share vs the learner IS an
    # Elo-consistent strength order over the pool — for free, no ladder.
    pool_names = [e.get("name", str(j)) for j, e in enumerate(cfg.get("league") or [])]
    lg = {}
    for game in collected:
        opp = game.get("league") or {}
        if not isinstance(opp, dict) or not opp:
            continue
        pts = game.get("points") or []
        if len(pts) != 4:
            continue
        for L in game.get("learner_seats") or []:
            for seat, j in opp.items():
                name = pool_names[j] if j < len(pool_names) else str(j)
                w, n, d = lg.get(name, (0.0, 0, 0.0))
                s = 1.0 if pts[L] > pts[seat] else 0.0 if pts[L] < pts[seat] else 0.5
                lg[name] = (w + s, n + 1, d + (pts[L] - pts[seat]))
    # share is sign-based (deal-luck robust); mean_diff keeps the magnitude
    # view so big-hand styles aren't underrated (user 2026-08-29)
    collect_parallel.last_league = {
        k: {"learner_share": round(w / n, 4), "n": n,
            "mean_diff": round(d / n, 1)} for k, (w, n, d) in lg.items()}
    # exp55-D: per-role hanchan uma (the greedy twin's mean uma is the live
    # deployment-form rating; learner uma shows the exploration tax)
    hz = {}
    for game in collected:
        h, roles = game.get("hanchan"), game.get("roles")
        if not h or not roles or "learner" not in roles:
            continue          # v2 table rows carry rotation roles, not seats
        if roles.get("pure"):
            # exp70: all four seats are the learner; report placement spread
            # (mean |uma|) as the only meaningful mirror statistic
            u, n = hz.get("pure_abs_uma", (0.0, 0))
            hz["pure_abs_uma"] = (u + sum(abs(x) for x in h["uma_points"]) / 4.0, n + 1)
            continue
        for role in ("learner", "twin", "bc", "top"):
            seat = roles[role]
            u, n = hz.get(role, (0.0, 0))
            hz[role] = (u + h["uma_points"][seat], n + 1)
    collect_parallel.last_hanchan = {
        k: {"mean_uma": round(u / n, 1), "n": n} for k, (u, n) in hz.items()}
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
    if cfg.get("no_episodes"):
        return {"episodes": [], "result": g.result or "", "league": league,
                "hanchan": getattr(g, "hanchan", None),
                "riichi": list(g.riichi or []), "n_melds": list(g.n_melds or []),
                "n_discards": g.n_discards,
                "learner_seats": sorted(learner_seats),
                "points": list(g.points or []),
                "start_points": list(g.start_points or []), "seed": seed}
    """Compact numpy episodes for one finished game (both worker paths).
    `league` is the {seat: pool_idx} opponent map ({} for mirror games) —
    shipped so the trainer can score learner-vs-pool outcomes for free
    (exp46-C rev3 rollout ratings)."""
    labels = completion_labels(g.result or "") if cmode == "hazard" else None
    eps = []
    # exp59 v1.5: off-policy value learning can consume EVERY seat's
    # trajectory (the engine records all four with per-seat rewards); the
    # caller must guarantee pool members share the learner's encoder/action
    # space, since episodes are concatenated tensor-wise downstream
    keep_seats = range(4) if cfg.get("all_seats_episodes") else learner_seats
    for pid in range(4):
        if pid not in keep_seats:
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
            "hanchan": getattr(g, "hanchan", None),
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
    # mixed action spaces (2026-08-30): masks, index lookups and follow-up
    # protocols are per MODEL, not per run — a 374-slot anchor seated against
    # a 46-slot learner must be masked in its own space
    pool_space = {j + 1: REGISTRY.get(e.get("action_space") or "native",
                                      REGISTRY["native"])
                  for j, e in enumerate(cfg.get("league") or [])}
    temps_list = cfg.get("rollout_temps")
    payload = []
    queue = list(range(n_games))
    active = {}            # game idx -> dict(gen, pending, seed, learner, opp, temps)
    _DIAG = {"rpc": 0.0, "rounds": 0, "rows": 0, "t0": time.perf_counter()}

    hanchan_credit = None
    if cfg.get("hanchan") and cfg.get("hanchan_credit") == "rank":
        # exp70 pure line: analytic rank-uma potential, no learned W
        from src.tasks.mahjong.hanchan import RankUmaCredit
        hanchan_credit = RankUmaCredit()
    elif cfg.get("hanchan") and cfg.get("hanchan_credit", "w") == "w" and cfg.get("hanchan_w_path"):
        # eval-mode hanchan (exp56 arena) carries no W and needs no reward
        from src.tasks.mahjong.hanchan import PlacementCredit
        hanchan_credit = PlacementCredit(cfg["hanchan_w_path"])

    # exp72 decision-level counterfactual rollouts: at a sampled learner
    # decision, clone the table (true wall, true hidden hands) and play
    # alternative actions to the end of the deal with the same policy; the
    # paired difference vs the executed action's realised return is a
    # low-noise advantage for that decision (design:
    # experiments/designs/design_offpolicy_data_and_counterfactual.md §3).
    # Single-deal games only (hanchan continuation needs the match context).
    cf_p = float(cfg.get("cf_p") or 0.0) if not cfg.get("hanchan") else 0.0
    cf_k = int(cfg.get("cf_k") or 2)
    cf_only_exposed = bool(cfg.get("cf_only_exposed", True))
    cf_records = {}          # parent game idx -> {seat: {step: {"alts": {slot: R}, "genbutsu": slot|None}}}
    cf_counter = [0]
    # branches get their own slot budget (the inference server sized its
    # per-worker rows for K parents + cf_branch_slots branches); when the
    # budget is full the decision is simply not evaluated (counted)
    cf_slots = int(cfg.get("cf_branch_slots") or 0) if cf_p > 0 else 0
    n_branch = [0]
    cf_skipped = [0]
    _TILE_RE = re.compile(r'tile="([^"]+)"')
    from src.tasks.mahjong.table import norm_tile as _norm

    def branch_return(g, pid):
        steps = g.trajectories[pid]
        if not steps:
            return 0.0
        if cfg["shaping"]:
            apply_shaping(steps, cfg["gamma"])
        rets = returns_to_go(steps, cfg["gamma"])
        return float(rets[0]) if len(rets) else 0.0

    def start_branch(parent_gi, pid, step_idx, slot, action_str, genbutsu):
        pst = active[parent_gi]
        clone = copy.deepcopy(pst["table"])
        gen = play_game_gen(table=clone, shaping=cfg["shaping"])
        try:
            _table, reqs = next(gen)
        except StopIteration:
            return
        if not reqs or reqs[0][0] != pid:
            return
        rec = cf_records.setdefault(parent_gi, {}).setdefault(pid, {}).setdefault(
            step_idx, {"alts": {}, "genbutsu": None})
        if genbutsu:
            rec["genbutsu"] = slot
        # the forced first action rides on a placeholder step: branches never
        # ship episodes, only the seat's realised return
        st0 = DnnStep(planes=torch.zeros(1), scalars=torch.zeros(1),
                      mask=torch.zeros(1, dtype=torch.bool), action_idx=slot, logprob=0.0)
        cf_counter[0] += 1
        key = f"b{cf_counter[0]}"
        try:
            table, reqs = gen.send([(st0, action_str)])
        except StopIteration as e:
            rec["alts"][slot] = branch_return(e.value, pid)
            return
        active[key] = {"gen": gen, "table": table, "reqs": reqs, "seed": None,
                       "learner": [pid], "opp": pst["opp"], "temps": pst["temps"],
                       "roles": None, "dev_rng": None, "deviated": True,
                       "branch": (parent_gi, pid, step_idx, slot), "nstep": [0, 0, 0, 0]}
        n_branch[0] += 1

    def maybe_branch(k, gi, pid, actions, chosen_slot, lookup):
        st = active[gi]
        if st.get("branch") or pid not in st["learner"] or len(lookup) < 2:
            return
        if not any('type="discard"' in a for a in actions):
            return                              # turn-phase discard decisions only
        tbl = st["table"]
        riichi_opp = [q for q in range(4) if q != pid and tbl.riichi[q]]
        if cf_only_exposed and not riichi_opp:
            return
        if st["cf_rng"].random() >= cf_p:
            return
        if n_branch[0] + cf_k > cf_slots:
            cf_skipped[0] += 1
            return
        chosen = lookup[chosen_slot]
        m = _TILE_RE.search(chosen)
        chosen_tile = _norm(m.group(1)) if m else None
        genbutsu = set()
        for q in riichi_opp:
            genbutsu.update(_norm(t.rstrip("*")) for t in tbl.discards[q])
        cands = [(s, a) for s, a in lookup.items() if s != chosen_slot and 'type="discard"' in a]
        def _tile(a):
            mm = _TILE_RE.search(a)
            return _norm(mm.group(1)) if mm else None
        gb = [(s, a) for s, a in cands if _tile(a) in genbutsu and _tile(a) != chosen_tile]
        picks = []
        if gb:
            s, a = st["cf_rng"].choice(gb)
            picks.append((s, a, True))
        rest = [(s, a) for s, a in cands if s not in {p[0] for p in picks}]
        st["cf_rng"].shuffle(rest)
        for s, a in rest[:max(0, cf_k - len(picks))]:
            picks.append((s, a, False))
        step_idx = st["nstep"][pid]
        for s, a, g in picks:
            start_branch(gi, pid, step_idx, s, a, g)

    def hanchan_plan(seed):
        """exp55-D four-seat table (user spec): learner T=1, greedy twin of
        the CURRENT weights T=0 (live deployment-form measurement), bc
        anchor T=0, one sampled top pool member T=0. Seat roles rotate
        with the match seed; dup replicas share it, so the group-baseline
        key still compares like with like."""
        rng = random.Random(int(seed) * 6011 + 3)
        order = rng.sample(range(4), 4)
        learner, twin, bc_seat, top_seat = order
        pool = cfg.get("league") or []
        # entry 0 is the bc anchor by driver convention; top sampled from rest
        top_idx = 0 if len(pool) <= 1 else 1 + rng.randrange(len(pool) - 1)
        opp = {bc_seat: 0, top_seat: top_idx}
        temps = {learner: cfg["temperature"], twin: 0.0,
                 bc_seat: 0.0, top_seat: 0.0}
        roles = {"learner": learner, "twin": twin, "bc": bc_seat,
                 "top": top_seat, "top_idx": top_idx}
        return [learner], opp, temps, roles

    def table_plan(seed):
        """Rating-system v2 (design D5): FOUR distinct entities at one
        table. A 2v2 duplicate yields one pairwise comparison per match;
        a four-entity table yields a full ranking (six pairwise contrasts)
        for the same wall. The seed's low two bits carry the rotation, so
        the same wall is played four times with entity i sitting at seat
        (i + rot) % 4 — every entity occupies every seat exactly once and
        the seat/dealer luck cancels the way duplicate pairing cancels
        wall luck. Entity 0 is the worker's own net; 1..3 are pool
        entries, each at its own temperature (an entity IS a checkpoint
        plus a condition)."""
        rot = seed & 3
        temps_by_entity = cfg["table_temps"]
        learner_seats, opp, temps = [], {}, {}
        for st in range(4):
            e = (st - rot) % 4
            if e == 0:
                learner_seats.append(st)
            else:
                opp[st] = e - 1
            temps[st] = float(temps_by_entity[e])
        return learner_seats, opp, temps, {"rot": rot,
                                           "entity_of_seat": [(st - rot) % 4
                                                              for st in range(4)]}

    def arena_plan(seed):
        """Duplicate-match eval (perf 2026-08-30): orientation lives in the
        seed's low bit (wall seed = seed >> 1). A occupies (0,2) or (1,3);
        B (league entry 0) fills the rest at T=1, A at cfg arena_temp_a."""
        orient = seed & 1
        a_seats = [0, 2] if orient == 0 else [1, 3]
        opp = {p: 0 for p in range(4) if p not in a_seats}
        temps = {p: (float(cfg.get("arena_temp_a", 1.0)) if p in a_seats
                     else float(cfg.get("arena_temp_b", 1.0)))
                 for p in range(4)}
        return a_seats, opp, temps, None

    def start(i):
        seed = seeds[i] if seeds else None
        roles = None
        if cfg.get("table"):
            learner_seats, opp, temps, roles = table_plan(seed)
        elif cfg.get("arena"):
            learner_seats, opp, temps, roles = arena_plan(seed)
        elif cfg.get("hanchan") and cfg.get("hanchan_pure"):
            # exp70 pure line: four copies of the learner at the training
            # temperature (mirror self-play over a full hanchan); the dup
            # replicas of the same match seed give the group baseline
            learner_seats, opp = list(range(4)), {}
            temps = {p: cfg["temperature"] for p in range(4)}
            roles = {"pure": True, "learner": 0}
        elif cfg.get("hanchan"):
            learner_seats, opp, temps, roles = hanchan_plan(seed)
        else:
            learner_seats, opp = league_plan(seed, cfg)
            if temps_list:
                rng = random.Random((seed or 0) * 7919 + i)
                temps = {p: rng.choice(temps_list) for p in range(4)}
            else:
                temps = {p: cfg["temperature"] for p in range(4)}
            if opp and cfg.get("league_opp_temp") is not None:
                # deterministic opponents (exp46-C'a): with frozen pool seats
                # at T~0, same-wall replicas diverge only through the
                # learner's own sampling — the dominant line-luck source
                # (75.9% of return variance measured 2026-08-29) drops out
                for pid in opp:
                    temps[pid] = cfg["league_opp_temp"]
        # arena hides the duplicate orientation in the seed's low bit, so
        # BOTH generators must take seed >> 1 as the wall/match seed — else
        # the two orientations play different walls and the duplicate
        # pairing silently degrades into two independent samples (exp56)
        # rotation rides the low bits: 2 bits for a four-entity table,
        # 1 for a duplicate pair, none for training rollouts
        wall_seed = ((seed >> 2) if cfg.get("table")
                     else (seed >> 1) if cfg.get("arena") else seed)
        if cfg.get("hanchan"):
            from src.tasks.mahjong.hanchan import play_hanchan_gen
            gen = play_hanchan_gen(wall_seed, shaping=cfg["shaping"],
                                   credit=hanchan_credit)
        else:
            gen = play_game_gen(deal_seed=wall_seed, shaping=cfg["shaping"])
        try:
            table, reqs = next(gen)
        except StopIteration as e:
            pkg = _package_game(e.value, learner_seats, seed, cfg, cmode, opp)
            pkg["roles"] = roles
            payload.append(pkg)
            return
        # exp68 oracle policy guiding: with probability (1 - hide_p) this game's
        # observations carry the hidden-state planes (v1ro encoder). Default
        # hide_p = 1.0 -> never visible, so every non-training path is blind.
        hide_p = float(cfg.get("oracle_hide_p", 1.0))
        if hide_p < 1.0:
            table.oracle_visible = random.Random((seed or 0) * 31 + 7).random() >= hide_p
        active[i] = {"gen": gen, "table": table, "reqs": reqs, "seed": seed,
                     "learner": learner_seats, "opp": opp, "temps": temps,
                     "roles": roles,
                     # single-deviation exploration (exp59 v1.4): at most one
                     # learner decision per game is sampled at single_dev_temp,
                     # every other one plays at the seat temperature (0 = greedy)
                     "dev_rng": random.Random((seed or 0) * 100003 + 17),
                     "deviated": False,
                     "branch": None, "nstep": [0, 0, 0, 0],
                     "cf_rng": random.Random((seed or 0) * 7919 + 101)}

    while queue and len(active) - n_branch[0] < K:
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
            mask, lookup = pool_space.get(model_id, _space).mask(actions)
            if os.environ.get("INFER_DEBUG") and not bool(mask.any()):
                with open("/tmp/vec_debug.txt", "a") as _f:
                    _f.write(f"EMPTY vec mask pid={pid} actions={actions!r}\n")
            planes.append(pl); scalars.append(sc); masks.append(mask); lookups.append(lookup)
            t = float(st["temps"][pid])
            dev_p = cfg.get("single_dev_p") or 0.0
            if (dev_p > 0 and not st["deviated"] and pid in st["learner"]
                    and len(lookup) > 1 and st["dev_rng"].random() < dev_p):
                # one coherent hand with exactly one plausible alternative
                # move: the counterfactual data a one-step improvement needs
                t = float(cfg.get("single_dev_temp", 1.0))
                st["deviated"] = True
            temps.append(t); mids.append(model_id)
        P = _pad_stack(planes)
        maxp = P.shape[1]
        S = _pad_stack(scalars)
        M = _pad_stack(masks)
        _t0 = time.perf_counter()
        idx, lp = net.act_batch(P, S, M, torch.tensor(temps), torch.tensor(mids, dtype=torch.int32))
        _DIAG["rpc"] += time.perf_counter() - _t0; _DIAG["rounds"] += 1; _DIAG["rows"] += len(rows)
        if cf_p > 0:
            for k, (gi, ri, pid, actions, model_id, var) in enumerate(rows):
                maybe_branch(k, gi, pid, actions, int(idx[k]), lookups[k])
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
            mode = pool_space.get(rows[k][4], _space).follow_up(int(idx[k]), actions)
            if mode is not None:
                pending.append((k, mode))
        follow = {}
        if pending:
            f_masks, f_lookups = [], []
            for k, mode in pending:
                m2, lk2 = pool_space.get(rows[k][4], _space).mask(rows[k][3], mode=mode)
                if os.environ.get("INFER_DEBUG") and not bool(m2.any()):
                    with open("/tmp/vec_debug.txt", "a") as _f:
                        _f.write(f"EMPTY vec FOLLOWUP mode={mode} actions={rows[k][3]!r}\n")
                f_masks.append(m2); f_lookups.append(lk2)
            P2 = _pad_stack([planes[k] for k, _ in pending])
            S2 = _pad_stack([scalars[k] for k, _ in pending])
            if P2.shape[1] < maxp:
                P2 = torch.nn.functional.pad(P2, (0, 0, 0, maxp - P2.shape[1]))
            idx2, lp2 = net.act_batch(
                P2, S2, _pad_stack(f_masks),
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
        # only the games that asked this round get a reply; branches started
        # mid-round (maybe_branch) already consumed their forced first action
        replies = {}
        for gi, _ri, _pid, _a, _m, _v in rows:
            replies.setdefault(gi, [None] * len(active[gi]["reqs"]))
        for k, (gi, ri, pid, actions, model_id, var) in enumerate(rows):
            if k in follow:
                step2, action_str = follow[k]
                # the generator contract carries ONE step per request, so the
                # declaration step rides along and is unpacked by _package_game
                step2.extra_steps = [first_steps[k]]
                replies[gi][ri] = (step2, action_str)
            else:
                replies[gi][ri] = (first_steps[k], lookups[k][int(idx[k])])
            active[gi]["nstep"][pid] += 2 if k in follow else 1
        for gi in list(replies):
            st = active[gi]
            try:
                st["table"], st["reqs"] = st["gen"].send(replies[gi])
            except StopIteration as e:
                if st.get("branch"):
                    pgi, bpid, step_idx, slot = st["branch"]
                    cf_records[pgi][bpid][step_idx]["alts"][slot] = branch_return(e.value, bpid)
                    del active[gi]
                    n_branch[0] -= 1
                    continue
                pkg = _package_game(e.value, st["learner"], st["seed"], cfg, cmode, st["opp"])
                pkg["roles"] = st.get("roles")
                pkg["_gi"] = gi
                payload.append(pkg)
                del active[gi]
                if queue and len(active) - n_branch[0] < K:
                    start(queue.pop(0))
    for pkg in payload:
        gi = pkg.pop("_gi", None)
        pkg["cf"] = cf_records.get(gi) if cf_p > 0 else None
    if payload and cf_p > 0:
        payload[0]["cf_skipped"] = cf_skipped[0]
    return payload
