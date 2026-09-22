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


def style_draw(seed: int, prior) -> list:
    """(p_pure, d_max, a_max): with prob p_pure the whole table plays the pure objective;
    otherwise each seat independently is pure (1/2) or draws tau_d ~ U(0, d_max), tau_a ~ U(0, a_max).
    Returns [d0, a0, d1, a1, d2, a2, d3, a3]."""
    p_pure, d_max, a_max = float(prior[0]), float(prior[1]), float(prior[2])
    rng = np.random.default_rng(seed * 2654435761 % (2 ** 32) + 89)
    if rng.random() < p_pure:
        return [0.0] * 8
    out = []
    for _ in range(4):
        if rng.random() < 0.5:
            out += [0.0, 0.0]
        else:
            out += [float(rng.random() * d_max), float(rng.random() * a_max)]
    return out


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
    if variant not in ("v1r", "v3r", "v3s"):
        raise SystemExit(f"collect_rust: encoder variant {variant!r} not ported (v1r/v3r/v3s only)")
    # exp89 style-conditioned population: per seed, each seat draws a reward style
    # (tau_d deal-in penalty, tau_a win bonus) from the prior; seeds are shared by the
    # dup_k replicas so the (seed, seat) group baseline still compares like with like.
    seat_styles = None
    prior = cfg.get("style_prior")
    if prior:
        if variant != "v3s":
            raise SystemExit("collect_rust: --style_prior needs the v3s encoder (arch cnn_m_v3s)")
        seat_styles = [style_draw(int(s), prior) for s in seeds]
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
                           True, variant, hanchan=hanchan, max_deals=int(cfg.get("hanchan_max_deals", 24)),
                           houjuu_extra=float(cfg.get("houjuu_extra", 0.0) or 0.0),
                           seat_styles=seat_styles,
                           houjuu_by_shanten=cfg.get("houjuu_by_shanten"))
    temperature = float(cfg.get("temperature", 1.0))
    q = float(riichi_rs.PLANE_Q)
    dev = torch.device(device)
    games = []
    # exp97 parameter-space noise: one factorised head perturbation PER TABLE SLOT, held for the
    # whole deal (resampled when the slot's seed changes), shared by the four mirror seats.
    # Actions are sampled from the noisy head; the recorded logprob is the CLEAN head's, so
    # PPO treats the perturbation as the policy's own exploration (NoisyNet reading, no IS).
    noise_sigma = float(cfg.get("param_noise_sigma", 0.0) or 0.0)
    noise = None
    if noise_sigma > 0:
        if cfg.get("league") and float(cfg.get("league_frac", 0.0) or 0.0) > 0:
            raise SystemExit("collect_rust: param_noise_sigma with a league pool is not supported")
        dims = net.noise_dims()
        mode = cfg.get("param_noise_mode", "full")
        noise = {"eps": net.sample_noise(k, dims, dev), "prev_seed": np.full(k, -2, dtype=np.int64),
                 "kl_sum": 0.0, "chg_sum": 0.0, "rows": 0, "resampled": 0, "mode": mode,
                 "res": [] if cfg.get("noise_diag") else None}      # reservoir of exposed far-hand states
        if mode == "full":
            # FULL-network perturbations (smoke 09-21: at matched KL 0.04 the coherent defensive lean of a
            # whole-net perturbation is 3.5 pp vs 0.4 pp for head-only rank-1 noise). E perturbed copies per
            # rollout, table slot i acts under copy i % E for every deal it hosts; the dup_k replicas of a
            # seed start in consecutive slots, so they see E distinct perturbations on the same wall.
            E = int(cfg.get("param_noise_copies", 16))
            noise["copies"] = perturbed_copies(net, E, noise_sigma)
            noise["resampled"] = E
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
                if noise is not None:
                    slot_seed = np.asarray(env.slot_seeds(), dtype=np.int64)
                    fresh = np.nonzero((slot_seed != noise["prev_seed"]) & (slot_seed >= 0))[0]
                    if len(fresh) and noise["mode"] != "full":
                        new = net.sample_noise(len(fresh), dims, dev)
                        ft = torch.from_numpy(fresh).to(dev)
                        for (ei, eo), (ni, no) in zip(noise["eps"], new):
                            ei[ft] = ni; eo[ft] = no
                        noise["prev_seed"] = slot_seed.copy(); noise["resampled"] += len(fresh)
                    g = torch.from_numpy(np.asarray(gids, dtype=np.int64)).to(dev)
                    if noise["mode"] == "full":
                        clean = net(P, S, M)
                        noisy = torch.empty_like(clean)
                        cid = (np.asarray(gids, dtype=np.int64) % len(noise["copies"]))
                        for j in np.unique(cid):
                            sel = torch.from_numpy(np.nonzero(cid == j)[0]).to(dev)
                            noisy[sel] = noise["copies"][int(j)](P[sel], S[sel], M[sel])
                    else:
                        eps_rows = [(ei[g], eo[g]) for ei, eo in noise["eps"]]
                        clean, noisy = net.forward_clean_and_noisy(P, S, M, eps_rows, noise_sigma)
                    lpc = torch.log_softmax(clean, 1)
                    if temperature <= 0:
                        idx = noisy.argmax(1)
                    else:
                        idx = torch.multinomial(torch.softmax(noisy / temperature, 1), 1).squeeze(1)
                    lp = lpc.gather(1, idx[:, None]).squeeze(1)
                    lpn = torch.log_softmax(noisy, 1)
                    pc = lpc.exp()
                    kl = (pc * (lpc - lpn).masked_fill(~M, 0.0)).sum(1)          # KL(clean || noisy) per row
                    noise["kl_sum"] += float(kl.sum()); noise["rows"] += n
                    noise["chg_sum"] += float((clean.argmax(1) != noisy.argmax(1)).sum())
                    if noise["res"] is not None and sum(r[0].shape[0] for r in noise["res"]) < 2048:
                        genb, info = env.safe_info()
                        sel = np.nonzero((info[:, 0] > 0) & (info[:, 1] >= 2) & mask[:, :34].any(1) & genb.any(1))[0]
                        if len(sel):
                            noise["res"].append((P[sel].cpu(), S[sel].cpu(), M[sel].cpu(), torch.from_numpy(genb[sel])))
                    acts_np, lps_np = idx.cpu().numpy().astype(np.int64), lp.float().cpu().numpy()
                elif not pool:
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
    collect_rust.last_noise = None
    if noise is not None:
        collect_rust.last_noise = {"sigma": noise_sigma, "kl": noise["kl_sum"] / max(noise["rows"], 1),
                                   "greedy_change": noise["chg_sum"] / max(noise["rows"], 1),
                                   "resampled": noise["resampled"], "rows": noise["rows"]}
        if cfg.get("noise_diag"):
            collect_rust.last_noise.update(noise_diag(games, variant))
            if noise["res"]:
                collect_rust.last_noise.update(lean_spread(net, noise["res"], dims, noise_sigma, dev,
                                                           copies=noise.get("copies")))
    return episodes, results


@torch.no_grad()
def perturbed_copies(net, n, sigma):
    """n deep copies of `net` with every weight tensor (dim > 1, value head excluded) perturbed by
    sigma * std(tensor) * N(0, 1) — relative isotropic parameter noise (Plappert et al. 2017)."""
    import copy as _copy
    out = []
    for _ in range(n):
        m = _copy.deepcopy(net).eval()
        for name, p in m.named_parameters():
            if p.dim() > 1 and not name.startswith("value"):
                p.data.add_(float(sigma) * p.data.std() * torch.randn_like(p))
        out.append(m)
    return out


@torch.no_grad()
def lean_spread(net, res, dims, sigma, dev, n_eps=32, copies=None):
    """State-controlled read of deal-coherent diversity: on a fixed reservoir of exposed far-hand
    states (>=1 opponent riichi, own shanten >= 2, a genbutsu discard available), the probability
    mass the NOISY head puts on genbutsu discards, averaged over states, for n_eps whole-table
    perturbations. lean_spread = std across perturbations (0 without noise) — the size of the
    coherent defensive lean a single perturbation induces; lean_clean = the clean head's value."""
    P = torch.cat([r[0] for r in res]).to(dev); S = torch.cat([r[1] for r in res]).to(dev)
    M = torch.cat([r[2] for r in res]).to(dev); G = torch.cat([r[3] for r in res]).to(dev)
    n = P.shape[0]
    h = net.trunk(P, S)
    def genb_mass(logits):
        p = torch.softmax(logits.masked_fill(~M, float("-inf")), 1)[:, :34]
        return float((p * G).sum(1).mean())
    clean = genb_mass(net.head(h))
    vals = []
    if copies is not None:
        vals = [genb_mass(c(P, S, M)) for c in copies]
    else:
        for _ in range(n_eps):
            e = net.sample_noise(1, dims, dev)
            eps_rows = [(ei.expand(n, -1), eo.expand(n, -1)) for ei, eo in e]
            vals.append(genb_mass(net.head_noisy(h, eps_rows, sigma)))
    vals = np.array(vals)
    return {"lean_clean": clean, "lean_noisy_mean": float(vals.mean()), "lean_spread": float(vals.std()), "lean_states": n}


def _hand_names(counts):
    return [f"{i % 9 + 1}{'mps'[i // 9]}" if i < 27 else f"{i - 26}z" for i in range(34) for _ in range(int(counts[i]))]


def noise_diag(games, variant):
    """exp97 mechanism read-out from the finished games of one rollout (all seats learner, dup_k
    replicas of each seed): per (seed, seat, replica) episode, the genbutsu rate of its exposed
    far-hand discards (>=1 opponent riichi flag, own shanten >= 2; genbutsu approximated by the
    riichi opponents' OWN rivers) and its point delta minus the (seed, seat) group mean. Reports
    corr(genbutsu rate, residual) — positive means the duplicate-wall group baseline can see the
    money in a deal-coherent defensive lean — and the between/within-table variance ratio."""
    import riichi_rs
    q = float(riichi_rs.PLANE_Q)
    rows = []                                   # (seed, seat, delta, genb_rate, n_exposed)
    for g in games:
        pts, sp = g.get("points"), g.get("start_points")
        if pts is None or sp is None:
            continue
        for e in g["episodes"]:
            seat = int(e["key"][1])
            pl = e["planes"]; sc = e["scalars"]; acts = np.asarray(e["actions"])
            if len(acts) == 0:
                continue
            disc = acts < 34
            riichi = sc[:, 5:8] > 0.5                          # opponents by offset 1..3
            expo = disc & riichi.any(1)
            n_e, n_g = 0, 0
            for t in np.nonzero(expo)[0]:
                counts = (pl[t, 0:4] > 0).sum(0)
                if riichi_rs.shanten(_hand_names(counts), int(round(float(sc[t, 11]) * 4))) < 2:
                    continue
                rivers = pl[t, 8:12] > 0
                safe = np.logical_and.reduce([rivers[1 + o] for o in range(3) if riichi[t, o]])
                n_e += 1; n_g += int(safe[int(acts[t])])
            rows.append((int(g["seed"]), seat, float(pts[seat] - sp[seat]), n_g / n_e if n_e else np.nan, n_e))
    if not rows:
        return {}
    a = np.array(rows, dtype=np.float64)
    key = a[:, 0] * 4 + a[:, 1]
    resid = a[:, 2].copy()
    for kk in np.unique(key):
        m = key == kk
        resid[m] -= a[m, 2].mean()
    ok = a[:, 4] >= 2
    out = {"diag_n": int(ok.sum()), "diag_exposed_far_rate": float(np.nanmean(a[ok, 3])) if ok.any() else 0.0}
    if ok.sum() > 8:
        out["diag_corr"] = float(np.corrcoef(a[ok, 3], resid[ok])[0, 1])
        # between-table (across replicas of the same seed/seat) vs within-noise variance of the fold lean
        kb = key[ok]; x = a[ok, 3]
        grp = np.array([x[kb == kk].var() for kk in np.unique(kb) if (kb == kk).sum() >= 2])
        out["diag_between_var"] = float(x.var()); out["diag_within_group_var"] = float(grp.mean()) if len(grp) else 0.0
    return out
