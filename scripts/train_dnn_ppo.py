"""PPO + critic arm for the conventional DNN, matched to the REINFORCE arm.

Why PPO here (decided from measurements, not defaults):
  * NOT for variance reduction. Four probes put the return's explained
    variance at 0.02-0.03 regardless of representation (LLM hidden state,
    engine ground truth, DNN encoding at random and at teacher level), so
    a critic cannot cancel much noise.
  * FOR sample efficiency. REINFORCE is only valid at the sampling policy,
    so that arm can take exactly ONE update per rollout batch (taking ~88
    collapsed entropy 2.007 -> 0.441 inside one iteration). Clipping makes
    several epochs per batch safe, and rollout — not gradient — is the
    7-hours-per-600k-games bottleneck.
  * FOR making draws informative. V's spread is ~15% of the return's, so a
    0-return draw gets an advantage of ~0.6 instead of exactly 0 (typical
    |advantage| is ~2.9). That is the principled version of "stop throwing
    draws away" — give them a baseline rather than more of them.

Everything else (engine, reward, gamma, duplicate deals, group baseline,
zero-return filter) is identical to the REINFORCE arm so the two are
directly comparable.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.dnn.net import MahjongPolicyNet                      # noqa: E402
from src.agents.dnn.parallel_rollout import (apply_group_baseline,   # noqa: E402
                                             collect_parallel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=None)
    ap.add_argument("--bc_anchor", default=None,
                    help="frozen BC checkpoint; adds bc_kl_coef * KL(pi || pi_BC) "
                         "to the loss (exp46: keep the human prior while improving)")
    ap.add_argument("--bc_kl_coef", type=float, default=0.0)
    ap.add_argument("--total_games", type=int, default=600000)
    ap.add_argument("--games_per_iter", type=int, default=2048)
    ap.add_argument("--dup_k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--games_per_worker", type=int, default=1,
                    help="perf 2026-08-23: K games interleaved per rollout "
                         "process, one batched RPC per round (gpu_infer only)")
    ap.add_argument("--rollout_temps", default=None,
                    help="exp28: comma-separated temperatures; each seat of each "
                         "rollout game samples at one of them (behaviour logprob "
                         "is recorded, so PPO's ratio becomes pi/b). Default: "
                         "single --temperature.")
    ap.add_argument("--entropy_coef", type=float, default=0.03)
    ap.add_argument("--value_coef", type=float, default=0.5)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--ppo_epochs", type=int, default=4)
    ap.add_argument("--target_kl", type=float, default=0.02)
    ap.add_argument("--arch", default=None,
                    help="arch-zoo model name (e.g. vit_small); overrides "
                         "channels/blocks")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--league", default=None,
                    help="exp22: json list of {name, path} frozen PURE-lineage "
                         "opponents; a fraction of deals seats the learner "
                         "against them (design docs/design_league_exp22.md)")
    ap.add_argument("--league_frac", type=float, default=0.5)
    ap.add_argument("--league_learner_seats", type=int, default=0,
                    help="fixed learner seat count in league games "
                         "(0 = legacy random 1-2)")
    ap.add_argument("--bf16_infer", action="store_true",
                    help="bf16 autocast in the rollout inference server "
                         "(2x-ish on attention archs; logits sampled in fp32)")
    ap.add_argument("--tf32", type=int, default=1,
                    help="TF32 matmul for the fp32 update path (default on)")
    ap.add_argument("--gpu_infer", action="store_true",
                    help="batched GPU inference server for rollouts (user rule "
                         "2026-08-22: all future runs); see infer_server.py")
    ap.add_argument("--gpu_infer_opponents", action="store_true",
                    help="host the league pool on the GPU server too (model_id "
                         "per slot); needed once opponents are >=10M nets")
    ap.add_argument("--infer_max_batch", type=int, default=128)
    ap.add_argument("--infer_wait_ms", type=float, default=0.0,
                    help="accumulation window for the inference server. 0 = "
                         "take whatever is pending and let the GPU's own "
                         "turnaround size the next batch (requests pile up "
                         "during the previous forward, so batch size "
                         "self-balances at arrival_rate x forward_time). "
                         "Measured 2026-08-25: 0 beats the old 4.0 default by "
                         "~10% on both a slow model (mortal_full 3599->3974 "
                         "decisions/s) and a fast one (cnn_m_r 143->157 "
                         "games/s), and is neutral at low concurrency (batch "
                         "is 2.2 either way with 3 workers). Waiting buys a "
                         "bigger batch, but the forward cost scales with batch "
                         "size, so the wait is net dead time.")
    ap.add_argument("--amp_update", action="store_true",
                    help="bf16 autocast around the PPO forward pass in the "
                         "update loop (matmuls only; logprob/ratio math stays "
                         "fp32). Local bench 2026-08-24, HRF, B=512: 258.9ms->"
                         "171.3ms (1.51x) and 9.3GB->5.4GB peak activation mem "
                         "(matters most for wide attention archs at real batch)")
    ap.add_argument("--warmup_updates", type=int, default=0,
                    help="linear LR warmup over N optimizer updates "
                         "(0 = off; transformers in RL want ~1000)")
    ap.add_argument("--adv_clamp", type=float, default=5.0,
                    help="normalized-advantage winsorize bound (sigma); a "
                         "yakuman event sits near 8 sigma, so the 5.0 "
                         "default caps big-hand policy gradients")
    ap.add_argument("--drop_zero_return", action="store_true")
    ap.add_argument("--gae_lambda", type=float, default=None,
                    help="Use GAE(lambda) advantages instead of Monte-Carlo "
                         "G - V(s). With a bootstrapped advantage, knowledge "
                         "held by the value model becomes a PER-ACTION signal "
                         "(e.g. a call that kills the menzen package is "
                         "penalized immediately via gamma*V(s')-V(s), without "
                         "waiting for the settlement). MC advantages reduce "
                         "the critic to variance reduction only.")
    ap.add_argument("--entropy_schedule", default=None,
                    help="step schedule 'games:coef,games:coef' overriding "
                         "--entropy_coef once the games counter passes each "
                         "threshold (exp13 hand-ladder arm)")
    ap.add_argument("--entropy_auto", action="store_true",
                    help="Lagrangian auto-anneal (exp13): treat the entropy "
                         "coefficient as a dual variable on the constraint "
                         "H >= entropy_floor. alpha decays exponentially "
                         "while entropy is safely above the floor and rises "
                         "when it approaches, so the annealing CURVE is "
                         "emergent — the only hand number is the safety "
                         "floor (exp8: collapse at H=0.44; champion healthy "
                         "at 0.76).")
    ap.add_argument("--entropy_floor", type=float, default=0.5)
    ap.add_argument("--entropy_floor_schedule", default=None,
                    help="games:target pairs, linear between knots (e.g. "
                         "'0:0.5,1000000:0.25'); overrides --entropy_floor "
                         "as the dual-control target (exp31)")
    ap.add_argument("--entropy_dual_lr", type=float, default=0.1)
    ap.add_argument("--entropy_abort", type=float, default=None,
                    help="hard safety: if post-update entropy falls below "
                         "this, snapshot and stop (exp8: policy died at "
                         "H=0.44; guard for aggressive anneal arms)")
    ap.add_argument("--critic_feats", choices=["none", "profile", "hazard"],
                    default="none",
                    help="Privileged CRITIC-ONLY features (exp11; the policy "
                         "never sees them). profile = 4-dim value-distance "
                         "profile concatenated into the value head (A1). "
                         "hazard = 9x5 per-family dynamics rows through a "
                         "shared completion-hazard head, V = sum P_y*value_y "
                         "+ residual, with a supervised BCE channel on "
                         "did-this-family-complete labels (A2).")
    ap.add_argument("--hazard_coef", type=float, default=0.5,
                    help="weight of the hazard head's BCE loss (A2 only)")
    ap.add_argument("--milestones", type=str, default="20000,80000,240000,600000")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_device",
                    default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device for the UPDATE (and, with --gpu_infer, the "
                         "batched rollout inference server). Without "
                         "--gpu_infer rollout stays on CPU workers: batch-1 "
                         "forwards of a 1.4M net are launch-bound, so 16 CPU "
                         "processes beat one GPU; the 2026-08-22 server "
                         "batches across workers + CUDA graphs and wins for "
                         ">=10M nets (9.5x on 192x40).")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to continue from; restores weights, "
                         "optimizer moments and the games counter when the "
                         "checkpoint carries them (older ones only have "
                         "weights — Adam moments then rebuild in a few iters)")
    ap.add_argument("--ckpt_every", type=int, default=25,
                    help="also snapshot every N iterations, so a restart "
                         "never loses more than that")
    ap.add_argument("--exp_dir", default=None)
    args = ap.parse_args()

    from src.agents.dnn.action_space import space_of_arch
    if args.tf32:
        torch.set_float32_matmul_precision("high")
    if args.entropy_schedule and args.entropy_auto:
        raise SystemExit("--entropy_schedule and --entropy_auto are exclusive")
    ent_schedule = []
    if args.entropy_schedule:
        ent_schedule = sorted((int(g), float(c)) for g, c in
                              (p.split(":") for p in
                               args.entropy_schedule.split(",")))

    torch.set_num_threads(max(1, os.cpu_count() // 3))
    torch.manual_seed(args.seed); random.seed(args.seed)
    exp_dir_guard = None
    exp_dir = args.exp_dir or f"experiments/dnn_ppo_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(exp_dir, exist_ok=True)
    # Single-writer lock. Two trainers sharing an exp_dir race on
    # train_log.json, checkpoints AND event files — observed as duplicated
    # TensorBoard steps after a launch script was run twice.
    lock = os.path.join(exp_dir, "RUNNING.lock")
    if os.path.exists(lock):
        try:
            old = int(open(lock).read().split()[0])
            alive = os.path.exists(f"/proc/{old}")
        except Exception:
            old, alive = -1, False
        if alive:
            raise SystemExit(
                f"refusing to start: pid {old} already writes {exp_dir} "
                f"(remove {lock} if that process is gone)")
        print(f"   stale lock from pid {old} removed", flush=True)
    with open(lock, "w") as f:
        f.write(f"{os.getpid()} {time.strftime('%F %T')}\n")
    import atexit
    atexit.register(lambda: os.path.exists(lock) and os.remove(lock))

    json.dump(vars(args), open(f"{exp_dir}/config.json", "w"), indent=2)
    milestones = sorted(int(x) for x in args.milestones.split(","))

    dev = torch.device(args.train_device)
    use_cf = args.critic_feats != "none"
    if args.arch:
        if use_cf:
            raise SystemExit("--critic_feats requires the default CNN "
                             "(zoo nets don't carry the critic variants)")
        from src.agents.dnn.arch_zoo import ZOO
        net = ZOO[args.arch][0]().to(dev)
        print(f"🏗 arch: {args.arch}", flush=True)
    else:
        from src.agents.dnn.selfplay import CFEAT_DIM
        net = MahjongPolicyNet(
            channels=args.channels, blocks=args.blocks,
            critic_feat_dim=CFEAT_DIM["profile"] if args.critic_feats == "profile" else 0,
            hazard=args.critic_feats == "hazard").to(dev)
        if use_cf:
            print(f"🔭 critic_feats: {args.critic_feats}", flush=True)
    anchor = None
    if args.bc_anchor:
        from src.agents.dnn.arch_zoo import ZOO as _ZOO
        from src.agents.dnn.net import load_compatible
        anchor = _ZOO[args.arch][0]().to(dev)
        load_compatible(anchor, torch.load(args.bc_anchor,
                                           map_location=dev)["state_dict"])
        anchor.eval()
        for q in anchor.parameters():
            q.requires_grad_(False)
        print(f"⚓ BC anchor {args.bc_anchor} (coef {args.bc_kl_coef})", flush=True)
    if args.init:
        from src.agents.dnn.net import load_compatible
        skipped = load_compatible(net, torch.load(args.init,
                                                  map_location="cpu")["state_dict"])
        print(f"⚓ warm-start {args.init}"
              + (f" (fresh: {skipped})" if skipped else ""), flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    n_upd = 0                      # optimizer updates, for --warmup_updates
    ent_alpha = args.entropy_coef      # live coefficient (schedule/auto laws)

    def save(tag, games, it=0):
        torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                    "channels": args.channels, "blocks": args.blocks,
                    "arch": args.arch, "critic_feats": args.critic_feats,
                    "games": games, "iter": it, "entropy_alpha": ent_alpha,
                    "optimizer": opt.state_dict()}, f"{exp_dir}/{tag}.pt")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.join(exp_dir, "tensorboard"))
    # TB display names only. "win_rate" is the share of self-play games ANY
    # seat wins (1 - draw rate): a style metric, not strength — it has misled
    # twice. The JSON field keeps its old name for downstream compatibility.
    TB_TAG = {"win_rate": "decisive_rate"}

    start_games, start_iter, log = 0, 0, []
    if args.resume:
        blob = torch.load(args.resume, map_location=dev)
        widened = any(k in blob["state_dict"] and v.dim() > 0
                      and tuple(blob["state_dict"][k].shape) != tuple(v.shape)
                      for k, v in net.state_dict().items())
        if widened:
            # pre-red (272-action) checkpoint: policy head widened to the
            # 374-action space (net.load_compatible); Adam moments no longer
            # match the parameter shapes, so they restart.
            from src.agents.dnn.net import load_compatible
            load_compatible(net, blob["state_dict"])
            print("   legacy action head widened 272->374 (optimizer moments restart)",
                  flush=True)
        else:
            net.load_state_dict(blob["state_dict"])
        if "optimizer" in blob and not widened:
            opt.load_state_dict(blob["optimizer"])
            print("   optimizer moments restored", flush=True)
        elif not widened:
            print("   (checkpoint has no optimizer state; Adam moments restart)",
                  flush=True)
        start_games = int(blob.get("games", 0))
        start_iter = int(blob.get("iter", 0))
        ent_alpha = float(blob.get("entropy_alpha", ent_alpha) or ent_alpha)
        old_log = os.path.join(exp_dir, "train_log.json")
        if os.path.exists(old_log):
            try:
                log = [r for r in json.load(open(old_log))
                       if r.get("games", 0) <= start_games]
            except Exception:
                log = []
        print(f"⏩ resume from {args.resume}: {start_games} games, "
              f"{len(log)} log rows kept", flush=True)
        for r in log:                       # replay history into TensorBoard
            for k, v in r.items():
                if k not in ("iter", "games", "wall_s") and isinstance(v, (int, float)):
                    writer.add_scalar(TB_TAG.get(k, k), float(v), int(r["games"]))
    else:
        save("games_0", 0)
    cfg = dict(channels=args.channels, blocks=args.blocks, arch=args.arch,
               temperature=args.temperature, gamma=args.gamma,
               games_per_worker=args.games_per_worker,
               rollout_temps=([float(x) for x in args.rollout_temps.split(",")]
                              if args.rollout_temps else None),
               shaping=False, seed=args.seed,
               critic_feats=args.critic_feats,
               gpu_infer=args.gpu_infer, gpu_infer_opponents=args.gpu_infer_opponents,
               infer_max_batch=args.infer_max_batch,
               infer_wait_ms=args.infer_wait_ms, infer_device=args.train_device,
               bf16_infer=args.bf16_infer,
               action_space=space_of_arch(args.arch))
    if args.league:
        cfg["league"] = json.load(open(args.league))
        cfg["league_frac"] = args.league_frac
        cfg["league_learner_seats"] = args.league_learner_seats
        print(f"🏟 league: {len(cfg['league'])} frozen opponents, frac {args.league_frac}, "
              f"learner seats {args.league_learner_seats or 'rand 1-2'}", flush=True)
    if args.gpu_infer:
        print(f"🚀 gpu_infer: batched rollout inference on {args.train_device} "
              f"(max_batch {args.infer_max_batch}, wait {args.infer_wait_ms} ms)", flush=True)

    games, it, t0, next_ms = start_games, start_iter, time.time(), 0
    while next_ms < len(milestones) and milestones[next_ms] <= start_games:
        next_ms += 1
    while games < args.total_games:
        it += 1
        for g_thr, coef in ent_schedule:      # step function on the counter
            if games >= g_thr:
                ent_alpha = coef
        n_deals = max(1, args.games_per_iter // args.dup_k)
        base = 6_000_000 + it * 9973
        seeds = [base + d for d in range(n_deals) for _ in range(args.dup_k)]
        net.eval()
        t_roll0 = time.time()
        episodes, results = collect_parallel(net, len(seeds), cfg, args.workers, seeds)
        apply_group_baseline(episodes, args.gamma)
        rollout_s = time.time() - t_roll0
        games += len(results)

        cat = lambda k: np.concatenate([e[k] for e in episodes])
        if episodes and episodes[0].get("planes_log") is not None:
            # wide observations arrive as the encoder's write log (98x smaller
            # over the pickle boundary); expand them straight onto the training
            # device with two scatters -- the dense tensor never exists on the
            # host, which is what OOM-killed the first attempt.
            from src.agents.dnn.mortal_obs import densify
            planes = densify([l for e in episodes for l in e["planes_log"]],
                             device=dev)
        else:
            # episodes ship planes as float16 (see _package_game); widen here
            planes = torch.from_numpy(cat("planes")).to(dev).float()
        scal = torch.from_numpy(cat("scalars")).to(dev)
        mask = torch.from_numpy(cat("mask")).to(dev)
        acts = torch.from_numpy(cat("actions")).to(dev)
        rets = torch.from_numpy(cat("returns")).to(dev)
        old_lp = torch.from_numpy(cat("old_logprobs")).to(dev)
        cfe = torch.from_numpy(cat("cfeats")).to(dev) if use_cf else None
        if args.critic_feats == "hazard":
            # per-(game,seat) settled-fact label, broadcast over the episode
            hlab = torch.from_numpy(np.concatenate(
                [np.repeat(e["hlabels"][None], len(e["returns"]), 0)
                 for e in episodes])).to(dev)

        if args.drop_zero_return:
            lens = [len(e["returns"]) for e in episodes]
            nz = [bool(np.abs(e["returns"]).max() > 1e-6) for e in episodes]
            idx_keep = torch.nonzero(torch.from_numpy(np.repeat(nz, lens)).to(dev),
                                     as_tuple=True)[0]
        else:
            idx_keep = torch.arange(len(acts), device=dev)
        n_eff = len(idx_keep)
        if n_eff < args.batch:
            continue

        with torch.no_grad():
            probe = idx_keep[torch.randperm(n_eff, device=dev)[:2048]]
            lp0 = torch.log_softmax(net(planes[probe], scal[probe], mask[probe]), 1)
            s0 = torch.where(torch.isfinite(lp0), lp0, torch.zeros_like(lp0))
            ent_before = float(-(lp0.exp() * s0).sum(1).mean())
            # V(s) for the whole batch under the SAMPLING policy
            vals = torch.cat([
                net.forward_with_value(planes[i:i + 8192], scal[i:i + 8192],
                                       mask[i:i + 8192],
                                       cfeats=cfe[i:i + 8192] if use_cf else None)[1]
                for i in range(0, len(acts), 8192)])
        if args.gae_lambda is not None:
            # per-episode GAE over the SAME per-step V used elsewhere
            adv_raw = torch.zeros_like(rets)
            off = 0
            lam, gam = args.gae_lambda, args.gamma
            for e in episodes:
                n = len(e["returns"])
                r = torch.from_numpy(e["rewards"]).to(vals.device)
                v = vals[off:off + n]
                gae = 0.0
                for t in range(n - 1, -1, -1):
                    v_next = v[t + 1] if t + 1 < n else 0.0   # terminal V := 0
                    delta = r[t] + gam * v_next - v[t]
                    gae = delta + gam * lam * gae
                    adv_raw[off + t] = gae
                off += n
        else:
            adv_raw = rets - vals
        adv = ((adv_raw - adv_raw[idx_keep].mean())
               / (adv_raw[idx_keep].std() + 1e-8)).clamp(-args.adv_clamp,
                                                        args.adv_clamp)

        net.train()
        t_upd0 = time.time()
        stop, passes, kls, closs, vloss, hloss, bkls = False, 0, [], [], [], [], []
        for ep in range(args.ppo_epochs):
            order = idx_keep[torch.randperm(n_eff, device=dev)]
            pass_kl = []
            for lo in range(0, n_eff, args.batch):
                sel = order[lo:lo + args.batch]
                with torch.autocast("cuda", torch.bfloat16,
                                    enabled=args.amp_update and dev.type == "cuda"):
                    logits, v = net.forward_with_value(
                        planes[sel], scal[sel], mask[sel],
                        cfeats=cfe[sel] if use_cf else None)
                # ratio/KL/loss math stays fp32 regardless of amp_update —
                # only the forward matmuls run in bf16 (bench: numerically
                # safe, matches the rollout inference server's pattern)
                logits, v = logits.float(), v.float()
                logp = torch.log_softmax(logits, 1)
                chosen = logp.gather(1, acts[sel][:, None]).squeeze(1)
                ratio = torch.exp(chosen - old_lp[sel])
                a = adv[sel]
                unclipped = ratio * a
                clipped = ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * a
                pg = -torch.min(unclipped, clipped).mean()
                safe = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))
                ent = -(logp.exp() * safe).sum(1).mean()
                vl = torch.nn.functional.mse_loss(v, rets[sel])
                loss = pg + args.value_coef * vl - ent_alpha * ent
                if anchor is not None and args.bc_kl_coef > 0:
                    with torch.no_grad():
                        alogits, _ = anchor.forward_with_value(
                            planes[sel], scal[sel], mask[sel],
                            cfeats=cfe[sel] if use_cf else None)
                        alogp = torch.log_softmax(alogits.float(), 1)
                    # sanitize BEFORE the multiply: where() keeps the un-taken
                    # branch in the graph, so (-inf)-(-inf)=nan on masked slots
                    # poisons the BACKWARD even though the forward is finite
                    fin = torch.isfinite(logp) & torch.isfinite(alogp)
                    diff = torch.where(fin, logp - alogp, torch.zeros_like(logp))
                    bc_kl = (logp.exp() * diff).sum(1).mean()
                    loss = loss + args.bc_kl_coef * bc_kl
                    bkls.append(float(bc_kl.detach()))
                if args.critic_feats == "hazard":
                    # supervised channel: completion is settled fact, so this
                    # loss is exempt from the on-policy/reuse constraints
                    hz = torch.nn.functional.binary_cross_entropy_with_logits(
                        net.hazard_head(cfe[sel]), hlab[sel])
                    loss = loss + args.hazard_coef * hz
                    hloss.append(hz.item())
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                if args.warmup_updates:
                    n_upd += 1
                    scale = min(1.0, n_upd / args.warmup_updates)
                    for g in opt.param_groups:
                        g["lr"] = args.lr * scale
                opt.step()
                with torch.no_grad():
                    lr_ = chosen - old_lp[sel]
                    pass_kl.append(float(((lr_.exp() - 1) - lr_).mean()))
                closs.append(pg.item()); vloss.append(vl.item())
            passes += 1
            kl = float(np.mean(pass_kl)); kls.append(kl)
            if kl > args.target_kl:
                stop = True
            if stop:
                break

        with torch.no_grad():
            net.eval()
            lp1 = torch.log_softmax(net(planes[probe], scal[probe], mask[probe]), 1)
            s1 = torch.where(torch.isfinite(lp1), lp1, torch.zeros_like(lp1))
            ent_after = float(-(lp1.exp() * s1).sum(1).mean())
            ev = float(1 - (rets[idx_keep] - vals[idx_keep]).var()
                       / (rets[idx_keep].var() + 1e-9))
        if args.entropy_auto and args.entropy_floor_schedule:
            knots = sorted((int(g), float(v)) for g, v in
                           (p.split(":") for p in args.entropy_floor_schedule.split(",")))
            t = games
            lo = max((k for k in knots if k[0] <= t), default=knots[0])
            hi = min((k for k in knots if k[0] > t), default=None)
            if hi is None:
                args.entropy_floor = lo[1]
            else:
                w = (t - lo[0]) / max(hi[0] - lo[0], 1)
                args.entropy_floor = lo[1] + w * (hi[1] - lo[1])
        if args.entropy_auto:
            # dual ascent on the constraint H >= floor: decay while safe,
            # rise when the floor approaches; the trajectory is emergent
            import math
            ent_alpha = min(0.1, max(1e-5, ent_alpha * math.exp(
                args.entropy_dual_lr * (args.entropy_floor - ent_after)
                / max(args.entropy_floor, 1e-6))))

        # NOTE: this is the DECISIVE-GAME rate (a hand ended in ron/tsumo by
        # anyone) = 1 - draw rate. It is NOT a per-agent win rate: all four
        # seats share these weights, so a per-agent figure is roughly this
        # divided by four (measured: 77.5% decisive <-> 19.4% per seat).
        # Ceiling is 100%, not 25%. It tracks how fast the shared policy
        # completes hands, i.e. tile efficiency — competitive strength only
        # comes from the arena, where different policies actually meet.
        update_s = time.time() - t_upd0
        win = sum(1 for r in results if "荣和" in r or "自摸" in r) / max(len(results), 1)
        lg = getattr(collect_parallel, "last_league", None)
        if lg:
            # rollout ratings (exp46-C rev3): learner-vs-pool point-share per
            # opponent, one jsonl row per iteration; chunk drivers aggregate
            # these to pick the next pool (strongest = lowest learner_share)
            with open(f"{exp_dir}/league_stats.jsonl", "a") as f:
                f.write(json.dumps({"iter": it, "games": games, "vs": lg}) + "\n")
        el = time.time() - t0
        row = {"iter": it, "games": games, "wall_s": round(el, 1),
               "rollout_s": round(rollout_s, 1), "update_s": round(update_s, 1),
               "pg_loss": float(np.mean(closs)), "value_loss": float(np.mean(vloss)),
               "entropy_before": ent_before, "entropy": ent_after,
               "approx_kl": kls[-1] if kls else 0.0, "ppo_passes": passes,
               "explained_var": ev, "win_rate": win,
               "bc_kl": float(np.mean(bkls)) if bkls else None,
               "entropy_coef": ent_alpha,
               "n_effective": n_eff, "n_raw": int(len(acts))}
        if hloss:
            row["hazard_bce"] = float(np.mean(hloss))
        log.append(row)
        for k, v in row.items():
            if k not in ("iter", "games", "wall_s") and isinstance(v, (int, float)):
                writer.add_scalar(TB_TAG.get(k, k), float(v), games)
        from src.agents.dnn.style_stats import summarize as _style_sum
        _sty = getattr(collect_parallel, "last_style", None)
        if _sty:
            for k, v in _style_sum(_sty).items():
                if k != "games":
                    writer.add_scalar(f"style/{k}", float(v), games)
        writer.add_scalar("games_per_sec",
                          (games - start_games) / max(el, 1e-9), games)
        writer.flush()
        if it % 5 == 0 or it == 1:
            print(f"[{it:4d}] games={games:7d} {el/60:6.1f}min "
                  f"{(games-start_games)/max(el,1):5.1f}局/s pg={row['pg_loss']:+.4f} "
                  f"H={ent_before:.3f}->{ent_after:.3f} win={win:.1%} "
                  f"kl={row['approx_kl']:.4f} passes={passes} EV={ev:+.3f} "
                  f"eff={n_eff}/{len(acts)}", flush=True)
            json.dump(log, open(f"{exp_dir}/train_log.json", "w"), indent=1)

        if args.entropy_abort is not None and ent_after < args.entropy_abort:
            save("entropy_abort", games, it)
            print(f"🛑 entropy {ent_after:.3f} < abort {args.entropy_abort}; "
                  f"snapshot saved, stopping", flush=True)
            break

        if args.ckpt_every and it % args.ckpt_every == 0:
            save("latest", games, it)

        while next_ms < len(milestones) and games >= milestones[next_ms]:
            save(f"games_{milestones[next_ms]}", games, it)
            print(f"📌 milestone {milestones[next_ms]} (actual {games})", flush=True)
            next_ms += 1

    save("games_final", games, it)
    json.dump(log, open(f"{exp_dir}/train_log.json", "w"), indent=1)
    writer.close()
    print(f"✅ {games} games in {(time.time()-t0)/60:.1f} min -> {exp_dir}")


if __name__ == "__main__":
    main()
