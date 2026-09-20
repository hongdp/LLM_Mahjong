"""exp96: prior-anchored Double DQN on the Rust engine ("Q-stitch").

Why (experiments/exp96_q_stitch_prereg/EXPERIMENT.md): exp94 measured that a safe
discard from a far hand is worth ~0 under the pure line's OWN continuation (A^pi,
what PPO sees) but +258 pts/decision under a good continuation. The Q-learning
target r + max Q(s', .) values an action under the BEST continuation in the data,
so it can stitch "push when tenpai" and "keep discarding safe tiles when far"
from different behaviour policies. Two fixes for the diagnosed exp59/71 failures:

* anchored operator — acting and bootstrapping both use
  argmax_a [log(pi0 + eps) + A(s,a)/tau]  (arch_zoo.AnchoredQPolicy), so Q noise
  among near-tie tile-efficiency actions cannot erode the prior's ordering;
* mixed behaviour population — every seat independently plays the learner, a frozen
  pool net, or the fold wrapper F (learner restricted to genbutsu discards while an
  opponent is in riichi and the own hand is not tenpai); all four seats' transitions
  enter the replay.

Usage (smoke):
  python scripts/train_dnn_qstitch.py --init <Q1x.pt> --pool M=<m.pt>,P4=<p4.pt> \
    --mix L=0.6,F=0.1,Q1x=0.1,M=0.1,P4=0.1 --total_games 20000 --exp_dir <dir>
"""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

SKIP0, RON0, TSUMO0, RED0 = 7 * 34, 5 * 34, 6 * 34, 8 * 34     # action slot = type_id * 34 + tile


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="aq_cnn_m_v3r")
    ap.add_argument("--init", required=True, help="policy ckpt: becomes the frozen prior AND the Q net's start")
    ap.add_argument("--resume", default=None, help="aq_* ckpt written by this script (replay is refilled)")
    ap.add_argument("--pool", default="", help="frozen behaviour nets 'name=path,...' (same encoder variant)")
    ap.add_argument("--mix", default="L=1.0",
                    help="per-seat behaviour probabilities 'L=..,F=..,<pool name>=..'; L = learner, "
                         "F = fold wrapper around the learner")
    ap.add_argument("--tau", type=float, default=0.05, help="scaled-return units per nat of prior log-odds")
    ap.add_argument("--eps", type=float, default=0.003, help="prior floor: any legal action costs <= log(1/eps) nats")
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling temperature of learner/F seats")
    ap.add_argument("--pool_temp", type=float, default=1.0)
    ap.add_argument("--total_games", type=int, default=20000)
    ap.add_argument("--games_per_iter", type=int, default=4096)
    ap.add_argument("--concurrent", type=int, default=1024, help="VecEnv concurrent tables")
    ap.add_argument("--nstep", type=int, default=3)
    ap.add_argument("--mc_until", type=int, default=300000, help="pure MC targets (trunk frozen) until this many games")
    ap.add_argument("--reward_scale", type=float, default=0.5, help="x thousand-point deal delta")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--trunk_lr_mult", type=float, default=0.3)
    ap.add_argument("--lr_final_frac", type=float, default=0.1, help="linear lr decay to this fraction at --total_games")
    ap.add_argument("--a_l2", type=float, default=0.25,
                    help="shrinkage: coef x mean over legal actions of A^2. The TD loss only touches the TAKEN action; "
                         "without this, A of rarely-visited actions is free generalisation noise that overruns the "
                         "anchor (smoke 09-20: 33%% override after 180 updates)")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--replay_cap", type=int, default=4_000_000)
    ap.add_argument("--replay_ratio", type=float, default=1.0, help="samples consumed per new env step")
    ap.add_argument("--min_replay", type=int, default=200_000)
    ap.add_argument("--target_every", type=int, default=1000)
    ap.add_argument("--ckpt_every", type=int, default=25, help="iterations between latest.pt saves")
    ap.add_argument("--milestone_games", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=96_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--exp_dir", required=True)
    return ap.parse_args()


def parse_kv(spec, cast=float):
    return {k: cast(v) for k, v in (x.split("=", 1) for x in spec.split(",") if x)}


class RingReplay:
    """Ring buffer of transitions; n-step targets (gamma = 1: one deal is a short finite
    horizon) are computed vectorised per ingested block. A per-slot block number
    invalidates rows whose bootstrap slot has since been overwritten."""

    def __init__(self, cap, n_planes, n_scal, n_act):
        self.cap = cap
        self.planes = np.zeros((cap, n_planes, 34), dtype=np.uint8)
        self.scal = np.zeros((cap, n_scal), dtype=np.float32)
        self.mask = np.zeros((cap, n_act), dtype=bool)
        self.act = np.zeros(cap, dtype=np.int64)
        self.nret = np.zeros(cap, dtype=np.float32)
        self.nidx = np.full(cap, -1, dtype=np.int64)
        self.mcret = np.zeros(cap, dtype=np.float32)
        self.block = np.zeros(cap, dtype=np.int64)
        self.head, self.size, self.block_no = 0, 0, 0

    def add_block(self, episodes, reward_scale, nstep):
        eps = [e for e in episodes if len(e["actions"]) > 0]
        if not eps:
            return 0
        lens = np.array([len(e["actions"]) for e in eps], dtype=np.int64)
        n = int(lens.sum())
        if n > self.cap:
            raise ValueError("block larger than the replay")
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        ep = np.repeat(np.arange(len(eps)), lens)
        pos = np.arange(n)
        end = starts[ep] + lens[ep]                                  # exclusive end of the row's episode
        rew = np.concatenate([np.asarray(e["rewards"], dtype=np.float64) for e in eps]) * reward_scale
        c = np.concatenate([[0.0], np.cumsum(rew)])
        self.block_no += 1
        slots = (self.head + pos) % self.cap
        self.planes[slots] = np.concatenate([e["planes"] for e in eps])
        self.scal[slots] = np.concatenate([e["scalars"] for e in eps])
        self.mask[slots] = np.concatenate([e["mask"] for e in eps])
        self.act[slots] = np.concatenate([np.asarray(e["actions"], dtype=np.int64) for e in eps])
        self.nret[slots] = (c[np.minimum(pos + nstep, end)] - c[pos]).astype(np.float32)
        self.mcret[slots] = (c[end] - c[pos]).astype(np.float32)
        self.nidx[slots] = np.where(pos + nstep < end, (self.head + pos + nstep) % self.cap, -1)
        self.block[slots] = self.block_no
        self.head = (self.head + n) % self.cap
        self.size = min(self.size + n, self.cap)
        return n

    def sample(self, n, rng):
        idx = rng.integers(0, self.size, size=n)
        boot = self.nidx[idx]
        ok = (boot < 0) | (self.block[np.clip(boot, 0, self.cap - 1)] == self.block[idx])
        return idx[ok]


def draw_roles(seed, probs):
    """Seed-deterministic behaviour role per seat: indices into the mix's role list."""
    rng = np.random.default_rng((int(seed) * 2654435761 + 96) % (2 ** 63))
    return rng.choice(len(probs), size=4, p=probs)


def fold_mask(mask, genb, info):
    """F wrapper: rows with an opponent in riichi and an own hand that is not tenpai (shanten >= 1)
    may only discard genbutsu (if any is legal) and never call (skip if offered). Win offers and
    rows without a safe option are left untouched."""
    out = mask.copy()
    exposed = (info[:, 0] > 0) & (info[:, 1] >= 1)
    win = mask[:, RON0:RON0 + 34].any(1) | mask[:, TSUMO0:TSUMO0 + 34].any(1)
    safe_d = mask[:, :34] & genb
    safe_r = mask[:, RED0:RED0 + 34] & genb
    has_safe = (safe_d | safe_r).any(1)
    rows = exposed & ~win & has_safe
    out[rows] = False
    out[rows, :34] = safe_d[rows]
    out[rows, RED0:RED0 + 34] = safe_r[rows]
    call = exposed & ~win & ~has_safe & mask[:, SKIP0] & ~mask[:, :34].any(1)
    out[call] = False
    out[call, SKIP0] = True
    return out


@torch.no_grad()
def collect_mixed(net, pool_nets, roles, probs, seeds, args, dev, variant):
    """One VecEnv pass. roles: list of role names aligned with probs ('L', 'F', pool names).
    Returns (episodes of ALL seats, stats dict)."""
    import riichi_rs
    env = riichi_rs.VecEnv([int(s) for s in seeds], int(args.concurrent), 1.0, False, 1.0, True, variant)
    q = float(riichi_rs.PLANE_Q)
    seed_index = {int(s): i for i, s in enumerate(seeds)}
    plan = np.stack([draw_roles(s, probs) for s in seeds])                      # [n_seeds, 4]
    i_f = roles.index("F") if "F" in roles else -1
    learner_roles = [i for i, r in enumerate(roles) if r in ("L", "F")]
    st = dict(rows=0, override=0, exp_rows=0, exp_override=0, far_rows=0, far_override=0,
              far_gap_sum=0.0, far_gap_n=0, a_abs=0.0, f_rows=0)
    games = []
    while not env.done():
        planes, scalars, mask, seats, gids = env.observe()
        n = planes.shape[0]
        if n == 0:
            env.step([], [])
            games.extend(env.drain_finished())
            continue
        genb, info = env.safe_info()
        slot_seed = np.asarray(env.slot_seeds(), dtype=np.int64)
        sidx = np.fromiter((seed_index.get(int(s), 0) for s in slot_seed), dtype=np.int64, count=len(slot_seed))
        role = plan[sidx[np.asarray(gids, dtype=np.int64)], np.asarray(seats, dtype=np.int64)]
        P = torch.from_numpy(planes).to(dev).view(n, -1, 34).float().div_(q)
        S = torch.from_numpy(scalars).to(dev)
        acts = np.zeros(n, dtype=np.int64)
        lsel = np.nonzero(np.isin(role, learner_roles))[0]
        if len(lsel):
            act_mask = mask[lsel]
            if i_f >= 0:
                fr = role[lsel] == i_f
                if fr.any():
                    act_mask[fr] = fold_mask(mask[lsel][fr], genb[lsel][fr], info[lsel][fr])
                    st["f_rows"] += int((act_mask[fr] != mask[lsel][fr]).any(1).sum())
            t = torch.from_numpy(lsel).to(dev)
            M = torch.from_numpy(mask[lsel]).to(dev)
            lp0, a, _ = net.parts(P[t], S[t], M)
            logits = lp0 + a / net.tau
            AM = torch.from_numpy(act_mask).to(dev)
            lg = logits.masked_fill(~AM, float("-inf"))
            if args.temperature <= 0:
                idx = lg.argmax(1)
            else:
                idx = torch.multinomial(torch.softmax(lg / args.temperature, 1), 1).squeeze(1)
            acts[lsel] = idx.cpu().numpy()
            # diagnostics on the UNRESTRICTED learner view of these states
            multi = M.sum(1) > 1
            g_anch, g_prior = logits.argmax(1), lp0.argmax(1)
            ov = (g_anch != g_prior) & multi
            inf_t = torch.from_numpy(info[lsel]).to(dev)
            expo = (inf_t[:, 0] > 0) & multi
            far = expo & (inf_t[:, 1] >= 2)
            st["rows"] += int(multi.sum()); st["override"] += int(ov.sum())
            st["exp_rows"] += int(expo.sum()); st["exp_override"] += int((ov & expo).sum())
            st["far_rows"] += int(far.sum()); st["far_override"] += int((ov & far).sum())
            st["a_abs"] += float(a.abs().sum() / M.sum().clamp_min(1) * int(multi.sum()))
            G = torch.from_numpy(genb[lsel]).to(dev)
            safe = (M[:, :34] & G)
            rows = far & safe.any(1) & M[:, :34].any(1)
            if rows.any():
                a_safe = a[:, :34].masked_fill(~safe, float("-inf")).max(1).values
                a_pri = a.gather(1, g_prior[:, None]).squeeze(1)
                st["far_gap_sum"] += float((a_safe - a_pri)[rows].sum()); st["far_gap_n"] += int(rows.sum())
        for j, name in enumerate(roles):
            if name in ("L", "F"):
                continue
            sel = np.nonzero(role == j)[0]
            if not len(sel):
                continue
            t = torch.from_numpy(sel).to(dev)
            idx, _ = pool_nets[name].act(P[t], S[t], torch.from_numpy(mask[sel]).to(dev),
                                         temperature=args.pool_temp, check=False)
            acts[sel] = idx.cpu().numpy()
        env.step(acts.tolist(), [0.0] * n)
        games.extend(env.drain_finished())
    episodes = []
    role_pts = {r: [0.0, 0] for r in roles}
    for g in games:
        pr = plan[seed_index[int(g["seed"])]]
        pts, sp = g.get("points"), g.get("start_points")
        for e in g["episodes"]:
            episodes.append(e)
        if pts is not None and sp is not None:
            for seat in range(4):
                rp = role_pts[roles[pr[seat]]]
                rp[0] += float(pts[seat] - sp[seat]); rp[1] += 1
    st["role_pts"] = {r: (v[0] / v[1] if v[1] else 0.0) for r, v in role_pts.items()}
    st["games"] = len(games)
    return episodes, st


def main():
    args = parse_args()
    os.makedirs(args.exp_dir, exist_ok=True)
    json.dump(vars(args), open(f"{args.exp_dir}/config.json", "w"), indent=2)
    dev = torch.device(args.device)
    from src.agents.dnn.arch_zoo import ZOO
    from src.agents.dnn.net import load_compatible
    from src.agents.dnn.parallel_rollout import _load_policy_ckpt

    net = ZOO[args.arch][0]()
    variant = net.encoder_variant
    games0, it0 = 0, 0
    if args.resume:
        blob = torch.load(args.resume, map_location="cpu", weights_only=False)
        net.load_state_dict(blob["state_dict"])
        games0, it0 = int(blob.get("games", 0)), int(blob.get("iter", 0))
        print(f"🔁 resume {args.resume} @ {games0} games", flush=True)
    else:
        blob = torch.load(args.init, map_location="cpu", weights_only=False)
        for part in (net.prior, net.q):
            skipped = load_compatible(part, blob["state_dict"])
            if skipped:
                raise SystemExit(f"--init does not fill {skipped}")
        last = net.q.head[-1]
        torch.nn.init.zeros_(last.weight); torch.nn.init.zeros_(last.bias)      # A = 0: the net IS the prior
        net.tau.fill_(args.tau); net.eps.fill_(args.eps)
    net = net.to(dev).eval()                # BatchNorm stays on the prior's running stats throughout
    target = copy.deepcopy(net).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    pool_nets = {}
    for name, path in parse_kv(args.pool, str).items():
        pn = _load_policy_ckpt(path).to(dev)
        if getattr(pn, "encoder_variant", None) != variant:
            raise SystemExit(f"pool {name}: encoder {pn.encoder_variant!r} != {variant!r}")
        pool_nets[name] = pn
    mix = parse_kv(args.mix)
    roles = list(mix)
    for r in roles:
        if r not in ("L", "F") and r not in pool_nets:
            raise SystemExit(f"--mix role {r!r} has no --pool entry")
    probs = np.array([mix[r] for r in roles], dtype=np.float64)
    probs = probs / probs.sum()
    print(f"🏗 {args.arch} prior+Q from {args.init}; tau={args.tau} eps={args.eps}; mix "
          + ", ".join(f"{r}={p:.2f}" for r, p in zip(roles, probs)), flush=True)

    head_params = list(net.q.head.parameters()) + list(net.q.value.parameters())
    head_ids = {id(p) for p in head_params}
    trunk_params = [p for p in net.q.parameters() if id(p) not in head_ids]
    opt = torch.optim.Adam([{"params": head_params, "lr": args.lr},
                            {"params": trunk_params, "lr": args.lr * args.trunk_lr_mult}])
    if args.resume and blob.get("optimizer"):
        opt.load_state_dict(blob["optimizer"])

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.join(args.exp_dir, "tensorboard"))
    rng = np.random.default_rng(args.seed)
    replay = None
    games, it, upd, t0 = games0, it0, 0, time.time()
    next_ms = (games // args.milestone_games + 1) * args.milestone_games
    log_rows = []

    def save(tag):
        torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "arch": args.arch,
                    "games": games, "iter": it, "optimizer": opt.state_dict()}, f"{args.exp_dir}/{tag}.pt")

    if not args.resume:
        save("games_0")
    mc_phase = games < args.mc_until
    for p in trunk_params:
        p.requires_grad_(not mc_phase)
    while games < args.total_games:
        it += 1
        if mc_phase and games >= args.mc_until:
            mc_phase = False
            for p in trunk_params:
                p.requires_grad_(True)
            target.load_state_dict(net.state_dict())
            print(f"🔁 MC->TD switch at {games} games: trunk unfrozen, target synced, n={args.nstep}", flush=True)
        lr_frac = 1.0 - (1.0 - args.lr_final_frac) * min(1.0, games / max(args.total_games, 1))
        opt.param_groups[0]["lr"] = args.lr * lr_frac
        opt.param_groups[1]["lr"] = args.lr * args.trunk_lr_mult * lr_frac
        t_r = time.time()
        seeds = [args.seed + it * 1_000_003 + d for d in range(args.games_per_iter)]
        episodes, st = collect_mixed(net, pool_nets, roles, probs, seeds, args, dev, variant)
        games += st["games"]
        if replay is None:
            e0 = episodes[0]
            replay = RingReplay(args.replay_cap, e0["planes"].shape[1], e0["scalars"].shape[1], e0["mask"].shape[1])
            print(f"🗃 replay cap {args.replay_cap}: {e0['planes'].shape[1]}x34 u8 planes", flush=True)
        new_steps = replay.add_block(episodes, args.reward_scale, args.nstep)
        del episodes
        rollout_s = time.time() - t_r

        t_u = time.time()
        n_upd = int(new_steps * args.replay_ratio / args.batch) if replay.size >= args.min_replay else 0
        td_sum = q_sum = y_sum = 0.0
        nb = 0
        for _ in range(n_upd):
            idx = replay.sample(args.batch, rng)
            P = torch.from_numpy(replay.planes[idx]).to(dev).float().div_(20.0)
            S = torch.from_numpy(replay.scal[idx]).to(dev)
            M = torch.from_numpy(replay.mask[idx]).to(dev)
            A = torch.from_numpy(replay.act[idx]).to(dev)
            _, a, v = net.parts(P, S, M)
            qsa = v + a.gather(1, A[:, None]).squeeze(1)
            if mc_phase:
                y = torch.from_numpy(replay.mcret[idx]).to(dev)
            else:
                y = torch.from_numpy(replay.nret[idx]).to(dev)
                boot = replay.nidx[idx]
                live = boot >= 0
                if live.any():
                    b = boot[live]
                    Pb = torch.from_numpy(replay.planes[b]).to(dev).float().div_(20.0)
                    Sb = torch.from_numpy(replay.scal[b]).to(dev)
                    Mb = torch.from_numpy(replay.mask[b]).to(dev)
                    with torch.no_grad():
                        lp0b, ab, _ = net.parts(Pb, Sb, Mb)
                        a_star = (lp0b + ab / net.tau).argmax(1)               # anchored greedy (online net)
                        _, at, vt = target.parts(Pb, Sb, Mb)
                        qt = vt + at.gather(1, a_star[:, None]).squeeze(1)     # evaluated by the target net
                    y[torch.from_numpy(live).to(dev)] += qt
            loss = F.smooth_l1_loss(qsa, y)
            if args.a_l2 > 0:
                loss = loss + args.a_l2 * ((a * a).sum(1) / M.sum(1).clamp_min(1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.q.parameters(), 10.0)
            opt.step()
            upd += 1
            if upd % args.target_every == 0:
                target.load_state_dict(net.state_dict())
            td_sum += loss.item(); q_sum += qsa.mean().item(); y_sum += y.mean().item(); nb += 1
        update_s = time.time() - t_u

        d = max(nb, 1)
        rate = lambda k, n: st[k] / max(st[n], 1)
        row = {"iter": it, "games": games, "wall_s": round(time.time() - t0, 1), "rollout_s": round(rollout_s, 1),
               "update_s": round(update_s, 1), "updates": upd, "td_loss": round(td_sum / d, 5),
               "q_mean": round(q_sum / d, 4), "target_mean": round(y_sum / d, 4),
               "a_abs": round(st["a_abs"] / max(st["rows"], 1), 5),
               "override": round(rate("override", "rows"), 4), "override_exposed": round(rate("exp_override", "exp_rows"), 4),
               "override_far": round(rate("far_override", "far_rows"), 4),
               "far_safe_gap": round(st["far_gap_sum"] / max(st["far_gap_n"], 1), 5),
               "f_rows": st["f_rows"], "role_pts": {k: round(v, 1) for k, v in st["role_pts"].items()},
               "replay": replay.size, "phase": "mc" if mc_phase else f"n{args.nstep}"}
        log_rows.append(row)
        json.dump(log_rows, open(f"{args.exp_dir}/train_log.json", "w"), indent=1)
        for k in ("td_loss", "q_mean", "target_mean", "a_abs", "override", "override_exposed", "override_far", "far_safe_gap"):
            writer.add_scalar(f"qs/{k}", row[k], games)
        for k, v in st["role_pts"].items():
            writer.add_scalar(f"qs_role_pts/{k}", v, games)
        writer.add_scalar("qs/games_per_s", (games - games0) / max(time.time() - t0, 1e-9), games)
        writer.add_scalar("qs/replay_size", replay.size, games)
        writer.add_scalar("qs/lr", args.lr * lr_frac, games)
        print(f"[{it:4d}] games={games:8d} {row['wall_s']/60:6.1f}min {(games-games0)/max(time.time()-t0,1e-9):5.1f}局/s "
              f"roll={rollout_s:.0f}s upd={update_s:.0f}s td={row['td_loss']:.4f} q={row['q_mean']:+.3f} y={row['target_mean']:+.3f} "
              f"|A|={row['a_abs']:.4f} ov={row['override']:.3f}/{row['override_exposed']:.3f}/{row['override_far']:.3f} "
              f"gap={row['far_safe_gap']:+.4f} L={st['role_pts'].get('L', 0):+.0f} {row['phase']}", flush=True)
        if it % args.ckpt_every == 0:
            save("latest")
        if games >= next_ms:
            save(f"games_{next_ms}")
            next_ms += args.milestone_games
    save("games_final")
    print(f"✅ {games} games in {(time.time()-t0)/60:.1f} min -> {args.exp_dir}", flush=True)


if __name__ == "__main__":
    main()
