"""Value-method line v1: Double DQN on the human-prior trunk (exp59).

Design: experiments/designs/design_dqn_value.md. The rollout stack is reused
unchanged — the policy head's logits are REINTERPRETED as Q(s,a): warm-started
from bc49 the argmax equals bc49's greedy action, so behaviour starts at prior
level, and net.act(temperature=1) gives Boltzmann exploration for free.

Sparse terminal rewards make 1-step TD propagate one hop per target sync, so
targets are n-step (default 10) Double DQN, with a pure-MC burn-in
(--mc_until) that also calibrates the head from log-policy scale to return
scale.

Usage (smoke):
  python scripts/train_dnn_dqn.py --arch convformer_m_v3r_m46 \
    --init experiments/_anchors_epoch6/bc49.pt --league league.json \
    --gpu_infer --gpu_infer_opponents --total_games 50000 \
    --exp_dir experiments/exp59_dqn_smoke_<ts>
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--init", required=True, help="warm-start ckpt (bc49)")
    ap.add_argument("--league", required=True,
                    help="json list of {name, path} frozen T=0 opponents; "
                         "entry 0 fills most seats (stationary env)")
    ap.add_argument("--total_games", type=int, default=50000)
    ap.add_argument("--games_per_iter", type=int, default=512)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--games_per_worker", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Boltzmann exploration over Q")
    ap.add_argument("--margin_coef", type=float, default=0.0,
                    help="DQfD large-margin loss vs the behavior policy's "
                         "greedy action: keeps the prior's action ORDERING "
                         "while TD fits magnitudes (v1.1 showed raw Q "
                         "regression erodes the supervised ordering faster "
                         "than 50k games of TD can rebuild it)")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="margin in scaled-return units")
    ap.add_argument("--margin_schedule", default=None,
                    help="step schedule 'games:coef,games:coef,...' overriding "
                         "--margin_coef once games pass each threshold — "
                         "anneal the prior anchor so TD may overrule it")
    ap.add_argument("--freeze_trunk_until", type=int, default=0,
                    help="LP-FT (Kumar et al. 2022): train ONLY the Q head with "
                         "the trunk frozen until this many games, then unfreeze "
                         "— a mis-scaled head must not backprop into the prior's "
                         "features (v1.1 lesson). 0 = never freeze")
    ap.add_argument("--trunk_lr_mult", type=float, default=1.0,
                    help="discriminative LR: trunk lr = lr * mult after unfreeze")
    ap.add_argument("--single_dev_p", type=float, default=0.0,
                    help="single-deviation exploration: per multi-choice "
                         "learner decision, probability of being THE one step "
                         "this game that samples at --single_dev_temp instead "
                         "of the seat temperature; at most once per game "
                         "(~28 learner decisions/game -> p=0.04 deviates in "
                         "~2/3 of games). Keeps hands coherent, unlike per-step "
                         "epsilon (0.9^110 ~ 0 intact hands)")
    ap.add_argument("--single_dev_temp", type=float, default=1.0)
    ap.add_argument("--self_frac", type=float, default=0.0,
                    help="fraction of iterations whose data is collected by the "
                         "CURRENT Q net (greedy at --temperature, plus single "
                         "deviations) instead of --behavior_ckpt. Optimistic Q "
                         "errors are only self-falsifying if the policy acts on "
                         "them; 1.0 = fully online DQN, 0.0 = pure off-policy")
    ap.add_argument("--store_dir", default=None,
                    help="exp60 learner mode: stream shards written by "
                         "scripts/value_actor.py instead of collecting rollouts "
                         "in-process; --updates_per_iter gradient steps per pass")
    ap.add_argument("--pool_dir", default=None,
                    help="exp60: where promoted gen_NNNN.pt land (actors read it)")
    ap.add_argument("--updates_per_iter", type=int, default=200)
    ap.add_argument("--promote_every", type=int, default=4000,
                    help="optimizer updates between promotion evals")
    ap.add_argument("--promote_pairs", type=int, default=1000,
                    help="duplicate pairs for the greedy promotion eval vs the "
                         "current newest gen; promote when share >= 0.5 - 1 SE")
    ap.add_argument("--all_seats", action="store_true",
                    help="train on all four seats' transitions (frozen T=0 "
                         "opponents included): 4x data per game and a second "
                         "expert's choices as plausible alternatives. Requires "
                         "every league member to share the learner's encoder "
                         "variant and action space")
    ap.add_argument("--behavior_ckpt", default=None,
                    help="freeze the ACTING policy to this ckpt (e.g. bc49) "
                         "while Q trains off-policy on its data — the v1 "
                         "smoke showed self-acting Boltzmann(Q) collapses to "
                         "near-uniform once TD rescales Q to return units")
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--replay_cap", type=int, default=1_000_000)
    ap.add_argument("--replay_ratio", type=float, default=2.0,
                    help="samples consumed per new env step")
    ap.add_argument("--nstep", type=int, default=10)
    ap.add_argument("--target_every", type=int, default=500,
                    help="hard target-net sync, in optimizer updates")
    ap.add_argument("--mc_until", type=int, default=20000,
                    help="pure Monte-Carlo targets until this many games "
                         "(scale calibration; no bootstrap)")
    ap.add_argument("--reward_scale", type=float, default=0.05)
    ap.add_argument("--gpu_infer", action="store_true")
    ap.add_argument("--gpu_infer_opponents", action="store_true")
    ap.add_argument("--infer_max_batch", type=int, default=512)
    ap.add_argument("--infer_wait_ms", type=float, default=0.0)
    ap.add_argument("--train_device", default="cuda")
    ap.add_argument("--ckpt_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7_000_000)
    ap.add_argument("--exp_dir", required=True)
    return ap.parse_args()


class Replay:
    """Ring buffer of transitions with insert-time n-step targets.

    Episodes are written contiguously; each step stores the ABSOLUTE slot of
    its bootstrap state (i+n within the episode, -1 if the episode ends
    first). A per-slot generation counter invalidates samples whose bootstrap
    slot has since been overwritten by the ring head.
    """

    def __init__(self, cap, n_planes, n_scal, n_act):
        self.cap = cap
        self.planes = np.zeros((cap, n_planes, 34), dtype=np.float16)
        self.scal = np.zeros((cap, n_scal), dtype=np.float32)
        self.mask = np.zeros((cap, n_act), dtype=bool)
        self.act = np.zeros(cap, dtype=np.int64)
        self.nret = np.zeros(cap, dtype=np.float32)     # n-step reward sum
        self.ndisc = np.zeros(cap, dtype=np.float32)    # gamma^k at bootstrap
        self.nidx = np.full(cap, -1, dtype=np.int64)    # bootstrap slot
        self.ngen = np.zeros(cap, dtype=np.int64)       # expected gen at nidx
        self.mcret = np.zeros(cap, dtype=np.float32)
        self.gen = np.zeros(cap, dtype=np.int64)
        self.head, self.size, self.write_no = 0, 0, 0

    def add_episode(self, planes, scal, mask, act, rew, mcret, gamma, nstep):
        T = len(act)
        slots = [(self.head + i) % self.cap for i in range(T)]
        nret = np.zeros(T, dtype=np.float32)
        ndisc = np.zeros(T, dtype=np.float32)
        nidx = np.full(T, -1, dtype=np.int64)
        for i in range(T):
            acc, g = 0.0, 1.0
            for k in range(min(nstep, T - i)):
                acc += g * rew[i + k]
                g *= gamma
            nret[i] = acc
            if i + nstep < T:
                ndisc[i] = g
                nidx[i] = slots[i + nstep]
        self.write_no += 1
        for i, s in enumerate(slots):
            self.planes[s] = planes[i]
            self.scal[s] = scal[i]
            self.mask[s] = mask[i]
            self.act[s] = act[i]
            self.nret[s] = nret[i]
            self.ndisc[s] = ndisc[i]
            self.nidx[s] = nidx[i]
            self.mcret[s] = mcret[i]
            self.gen[s] = self.write_no
            self.ngen[s] = self.write_no      # bootstrap slot is same episode
        self.head = (self.head + T) % self.cap
        self.size = min(self.size + T, self.cap)

    def sample(self, n, rng):
        idx = rng.integers(0, self.size, size=n)
        boot = self.nidx[idx]
        # drop rows whose bootstrap slot was overwritten by a later episode
        ok = (boot < 0) | (self.gen[np.clip(boot, 0, self.cap - 1)] == self.ngen[idx])
        return idx[ok]


def main():
    args = parse_args()
    os.makedirs(args.exp_dir, exist_ok=True)
    json.dump(vars(args), open(f"{args.exp_dir}/config.json", "w"), indent=2)
    dev = torch.device(args.train_device)

    from src.agents.dnn.action_space import space_of_arch
    from src.agents.dnn.arch_zoo import ZOO
    from src.agents.dnn.net import load_compatible
    from src.agents.dnn.parallel_rollout import collect_parallel

    net = ZOO[args.arch][0]().to(dev)
    blob = torch.load(args.init, map_location="cpu", weights_only=False)
    skipped = load_compatible(net, blob["state_dict"])
    print(f"🏗 arch {args.arch}, warm-start {args.init}"
          + (f" (fresh: {skipped})" if skipped else ""), flush=True)
    target = ZOO[args.arch][0]().to(dev)
    target.load_state_dict(net.state_dict())
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)
    beh_net = net
    if args.behavior_ckpt:
        beh_net = ZOO[args.arch][0]().to(dev)
        load_compatible(beh_net, torch.load(args.behavior_ckpt,
                                            map_location="cpu")["state_dict"])
        beh_net.eval()
        for p in beh_net.parameters():
            p.requires_grad_(False)
        print(f"🎭 frozen behavior: {args.behavior_ckpt}", flush=True)
    # the action head: MahjongPolicyNet has one `head`; ConvFormer46 scores
    # the 46 slots with four small layers (exp47 structured head)
    HEAD_NAMES = ("head", "type_head", "slot_tile", "slot_red", "slot_global")
    head_params = [p for name, m in net.named_children() if name in HEAD_NAMES
                   for p in m.parameters()]
    if not head_params:
        raise SystemExit(f"no action head found among {[n for n, _ in net.named_children()]}")
    head_ids = {id(p) for p in head_params}
    trunk_params = [p for p in net.parameters() if id(p) not in head_ids]
    opt = torch.optim.Adam([{"params": head_params, "lr": args.lr},
                            {"params": trunk_params,
                             "lr": args.lr * args.trunk_lr_mult}])
    trunk_frozen = args.freeze_trunk_until > 0
    if trunk_frozen:
        for p in trunk_params:
            p.requires_grad_(False)
        print(f"🧊 trunk frozen ({sum(p.numel() for p in trunk_params)} params) "
              f"until {args.freeze_trunk_until} games; head "
              f"{sum(p.numel() for p in head_params)} params trainable", flush=True)

    cfg = dict(channels=blob.get("channels", 64), blocks=blob.get("blocks", 3),
               arch=args.arch, temperature=args.temperature, gamma=args.gamma,
               games_per_worker=args.games_per_worker, rollout_temps=None,
               shaping=False, seed=args.seed, critic_feats="none",
               gpu_infer=args.gpu_infer,
               gpu_infer_opponents=args.gpu_infer_opponents,
               infer_max_batch=args.infer_max_batch,
               infer_wait_ms=args.infer_wait_ms, infer_device=args.train_device,
               bf16_infer=False, no_episodes=False,
               league=json.load(open(args.league)), league_frac=1.0,
               league_learner_seats=1, league_opp_temp=0.0, hanchan=False,
               hanchan_w_path=None, action_space=space_of_arch(args.arch),
               single_dev_p=args.single_dev_p, single_dev_temp=args.single_dev_temp,
               all_seats_episodes=bool(args.all_seats))
    print(f"🏟 league: {len(cfg['league'])} frozen T=0 opponents, learner x1",
          flush=True)

    replay = None
    rng = np.random.default_rng(args.seed)
    reader = None
    if args.store_dir:
        from src.agents.dnn.replay_store import StoreReader
        reader = StoreReader(args.store_dir)
        os.makedirs(args.pool_dir or args.exp_dir, exist_ok=True)
        print(f"📥 learner streams {args.store_dir}; promotions -> "
              f"{args.pool_dir or args.exp_dir}", flush=True)
    next_promote = args.promote_every
    promo_state = {"low_streak": 0, "frozen": False}

    def promotion_eval(games_now, it_now):
        """Greedy self vs the newest promoted gen (or --init when none yet):
        promote when not worse at 1 SE. The eval seeds advance per attempt so
        a repeated attempt never replays the same walls."""
        import glob as _glob
        import math
        from scripts.run_elo_league import play_pair_vector
        pool_dir = args.pool_dir or args.exp_dir
        gens = sorted(_glob.glob(os.path.join(pool_dir, "gen_*.pt")))
        ref = gens[-1] if gens else args.init
        cand = os.path.join(pool_dir, "candidate.pt")
        torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                    "arch": args.arch, "games": games_now, "iter": it_now,
                    "channels": cfg["channels"], "blocks": cfg["blocks"]}, cand)
        seed0 = 65_000_000 + it_now * 10_007
        sc, _, _ = play_pair_vector(cand, ref, args.promote_pairs, seed0, 20,
                                    args.train_device, temp_a=0.0, temp_b=0.0,
                                    hanchan=False)
        share = sum(sc) / len(sc)
        se = math.sqrt(max(share * (1 - share), 1e-9) / len(sc))
        # absolute scale: the same candidate vs the FIXED prior. A chain of
        # "not worse than the previous gen" promotions is non-transitive and
        # can drift down one noisy step at a time; the anchor read catches it.
        anchor = args.behavior_ckpt or args.init
        sca, _, _ = play_pair_vector(cand, anchor, args.promote_pairs, seed0 + 1, 20,
                                     args.train_device, temp_a=0.0, temp_b=0.0,
                                     hanchan=False)
        share_bc = sum(sca) / len(sca)
        promo_state["low_streak"] = promo_state["low_streak"] + 1 if share_bc < 0.48 else 0
        if promo_state["low_streak"] >= 2 and not promo_state["frozen"]:
            promo_state["frozen"] = True
            print(f"🚨 promotions FROZEN: vs {os.path.basename(anchor)} < 0.48 twice in a row "
                  f"({share_bc:.4f}); generations no longer advance", flush=True)
        ok = (share >= 0.5) and not promo_state["frozen"]     # point estimate, not 1-SE slack
        if ok:
            new = os.path.join(pool_dir, f"gen_{len(gens) + 1:04d}.pt")
            os.replace(cand, new)
        print(f"🏅 promotion eval @ {games_now} games: vs {os.path.basename(ref)} "
              f"share {share:.4f} ± {se:.4f} | vs {os.path.basename(anchor)} {share_bc:.4f} "
              f"-> {'PROMOTED ' + os.path.basename(new) if ok else ('FROZEN' if promo_state['frozen'] else 'held')}",
              flush=True)
        writer.add_scalar("dqn/promo_share_vs_latest", share, games_now)
        writer.add_scalar("dqn/promo_share_vs_bc49", share_bc, games_now)
        writer.add_scalar("dqn/promotion_frozen", int(promo_state["frozen"]), games_now)
        writer.add_scalar("dqn/generation", len(gens) + (1 if ok else 0), games_now)
        return ok
    games, it, upd, t0 = 0, 0, 0, time.time()
    log_rows = []
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.join(args.exp_dir, "tensorboard"))

    def save(tag, g, i):
        torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                    "arch": args.arch, "games": g, "iter": i,
                    "channels": cfg["channels"], "blocks": cfg["blocks"],
                    "optimizer": opt.state_dict()}, f"{args.exp_dir}/{tag}.pt")

    save("games_0", 0, 0)
    margin_coef = args.margin_coef
    m_sched = []
    if args.margin_schedule:
        m_sched = sorted((int(g), float(c)) for g, c in
                         (x.split(":") for x in args.margin_schedule.split(",")))
    mc_phase = True
    while games < args.total_games:
        it += 1
        for g_thr, coef in m_sched:
            if games >= g_thr:
                margin_coef = coef
        if trunk_frozen and games >= args.freeze_trunk_until:
            for p in trunk_params:
                p.requires_grad_(True)
            trunk_frozen = False
            target.load_state_dict(net.state_dict())
            print(f"🔥 trunk unfrozen at {games} games (trunk lr x{args.trunk_lr_mult}); "
                  f"target net synced", flush=True)
        if mc_phase and games > args.mc_until:
            # tranche-1 TB (2026-09-01): the first hard sync landed ~15k games
            # AFTER the MC->TD switch, so bootstrapping ran off the warm-start
            # LOGITS (scale +-5) and targets sat at ~+1.0 for 16k games. Sync
            # the calibrated net into the target the moment bootstrapping starts.
            target.load_state_dict(net.state_dict())
            mc_phase = False
            print(f"🔁 MC->TD switch at {games} games: target net synced", flush=True)
        net.eval()
        t_r = time.time()
        if args.store_dir:
            # exp60 learner: the actor process fills the store; ingest whatever
            # is new, and if nothing arrived yet keep training on the ring
            from src.agents.dnn.replay_store import read_shard
            episodes, metas = [], []
            while True:
                for p in reader.new_shards():
                    eps, meta = read_shard(p)
                    episodes.extend(eps)
                    metas.append(meta)
                if episodes or (replay is not None and replay.size >= args.batch):
                    break
                time.sleep(5)
            games += sum(m["games"] for m in metas)
            store_proxy = [m["learner_pts"] for m in metas if m.get("learner_pts") is not None]
        else:
            seeds = [args.seed + it * 100003 + d for d in range(args.games_per_iter)]
            act_net = net if rng.random() < args.self_frac else beh_net
            episodes, results = collect_parallel(act_net, len(seeds), cfg,
                                                 args.workers, seeds)
            games += len(results)
        rollout_s = time.time() - t_r

        new_steps, ret_sum, ret_n = 0, 0.0, 0
        for e in episodes:
            if e.get("planes_log") is not None:
                raise SystemExit("sparse planes unsupported in DQN v1")
            pl = e["planes"]
            rw = e["rewards"] * args.reward_scale
            mc = e["returns"] * args.reward_scale
            if replay is None:
                replay = Replay(args.replay_cap, pl.shape[1], e["scalars"].shape[1],
                                e["mask"].shape[1])
                print(f"🗃 replay: cap {args.replay_cap}, "
                      f"{pl.shape[1]}x34 planes fp16", flush=True)
            replay.add_episode(pl, e["scalars"], e["mask"], e["actions"],
                               rw, mc, args.gamma, args.nstep)
            new_steps += len(e["actions"])
            ret_sum += float(mc[0]) if len(mc) else 0.0
            ret_n += 1

        # learner mean end-points vs 25k start — free strength proxy vs the
        # frozen anchors (positive = holding its own at the league table);
        # per-game facts live in last_games, `results` is just result strings
        if args.store_dir:
            proxy = float(np.mean(store_proxy)) if store_proxy else 0.0
            n_upd = args.updates_per_iter
        else:
            diffs = [g["points"][g["learner_seats"][0]] - 25000
                     for g in collect_parallel.last_games
                     if g.get("learner_seats") and g.get("points")]
            proxy = float(np.mean(diffs)) if diffs else 0.0
            n_upd = max(1, int(new_steps * args.replay_ratio / args.batch))
        net.train()
        t_u = time.time()
        td_sum, q_sum, y_sum, nb = 0.0, 0.0, 0.0, 0
        for _ in range(n_upd):
            idx = replay.sample(args.batch, rng)
            if len(idx) < 32:
                continue
            P = torch.from_numpy(replay.planes[idx]).to(dev).float()
            S = torch.from_numpy(replay.scal[idx]).to(dev)
            M = torch.from_numpy(replay.mask[idx]).to(dev)      # bool
            A = torch.from_numpy(replay.act[idx]).to(dev)
            q_all = net(P, S, M)
            q = q_all.gather(1, A[:, None]).squeeze(1)
            if games <= args.mc_until:
                y = torch.from_numpy(replay.mcret[idx]).to(dev)
            else:
                y = torch.from_numpy(replay.nret[idx]).to(dev)
                boot = replay.nidx[idx]
                live = boot >= 0
                if live.any():
                    b = boot[live]
                    Pb = torch.from_numpy(replay.planes[b]).to(dev).float()
                    Sb = torch.from_numpy(replay.scal[b]).to(dev)
                    Mb = torch.from_numpy(replay.mask[b]).to(dev)
                    with torch.no_grad():
                        # forward already sets illegal slots to -inf via mask
                        a_star = net(Pb, Sb, Mb).argmax(1)
                        qt = target(Pb, Sb, Mb).gather(1, a_star[:, None]).squeeze(1)
                    disc = torch.from_numpy(replay.ndisc[idx][live]).to(dev)
                    y[torch.from_numpy(live).to(dev)] += disc * qt
            loss = F.smooth_l1_loss(q, y)
            if margin_coef > 0:
                # J_E(Q) = max_a[Q + m*1(a != a_E)] - Q(s, a_E), a_E = the
                # frozen prior's greedy action on this state (DQfD eq. 2)
                with torch.no_grad():
                    a_e = beh_net(P, S, M).argmax(1)
                pad = torch.full_like(q_all, args.margin)
                pad.scatter_(1, a_e[:, None], 0.0)
                aug = (q_all + pad).masked_fill(~M, float("-inf")).max(1).values
                q_e = q_all.gather(1, a_e[:, None]).squeeze(1)
                loss = loss + margin_coef * (aug - q_e).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            upd += 1
            if upd % args.target_every == 0:
                target.load_state_dict(net.state_dict())
            td_sum += float(loss)
            q_sum += float(q.mean())
            y_sum += float(y.mean())
            nb += 1
        update_s = time.time() - t_u

        ret0 = ret_sum / max(ret_n, 1)
        row = {"iter": it, "games": games, "wall_s": round(time.time() - t0, 1),
               "rollout_s": round(rollout_s, 1), "update_s": round(update_s, 1),
               "updates": upd, "td_loss": round(td_sum / max(nb, 1), 5),
               "q_mean": round(q_sum / max(nb, 1), 4),
               "target_mean": round(y_sum / max(nb, 1), 4),
               "ep_return0": round(ret0, 4), "league_pts": round(proxy, 1),
               "replay": replay.size, "margin_coef": margin_coef,
               "phase": ("mc" if games <= args.mc_until else f"n{args.nstep}")}
        log_rows.append(row)
        json.dump(log_rows, open(f"{args.exp_dir}/train_log.json", "w"), indent=1)
        # in store mode the actor owns league_pts (it measures it); the learner
        # logs only what it computes itself
        keys = ("td_loss", "q_mean", "target_mean", "ep_return0") + (() if args.store_dir else ("league_pts",))
        for k in keys:
            writer.add_scalar(f"dqn/{k}", row[k], games)
        writer.add_scalar("dqn/replay_size", replay.size, games)
        writer.add_scalar("dqn/margin_coef", margin_coef, games)
        writer.add_scalar("dqn/updates_per_s", nb / max(update_s, 1e-9), games)
        writer.add_scalar("dqn/samples_per_s", nb * args.batch / max(update_s, 1e-9), games)
        writer.add_scalar("dqn/ingest_steps_per_s", new_steps / max(rollout_s, 1e-9), games)
        writer.add_scalar("dqn/replay_ratio_eff", nb * args.batch / max(new_steps, 1), games)
        writer.add_scalar("dqn/updates_total", upd, games)
        gs = games / max(time.time() - t0, 1e-9)
        print(f"[{it:4d}] games={games:7d} {row['wall_s']/60:5.1f}min "
              f"{gs:5.1f}局/s td={row['td_loss']:.4f} q={row['q_mean']:+.3f} "
              f"y={row['target_mean']:+.3f} lg={row['league_pts']:+6.0f} "
              f"buf={replay.size} {row['phase']}", flush=True)
        if it % args.ckpt_every == 0:
            save(f"games_{games}", games, it)
        if args.store_dir and upd >= next_promote:
            net.eval()
            promotion_eval(games, it)
            net.train()
            next_promote += args.promote_every
    save("games_final", games, it)
    print(f"✅ {games} games in {(time.time()-t0)/60:.1f} min -> {args.exp_dir}",
          flush=True)


if __name__ == "__main__":     # REQUIRED: spawn re-imports __main__
    main()
