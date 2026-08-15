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
    ap.add_argument("--total_games", type=int, default=600000)
    ap.add_argument("--games_per_iter", type=int, default=2048)
    ap.add_argument("--dup_k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--entropy_coef", type=float, default=0.03)
    ap.add_argument("--value_coef", type=float, default=0.5)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--ppo_epochs", type=int, default=4)
    ap.add_argument("--target_kl", type=float, default=0.02)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--drop_zero_return", action="store_true")
    ap.add_argument("--milestones", type=str, default="20000,80000,240000,600000")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp_dir", default=None)
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() // 3))
    torch.manual_seed(args.seed); random.seed(args.seed)
    exp_dir = args.exp_dir or f"experiments/dnn_ppo_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(exp_dir, exist_ok=True)
    json.dump(vars(args), open(f"{exp_dir}/config.json", "w"), indent=2)
    milestones = sorted(int(x) for x in args.milestones.split(","))

    net = MahjongPolicyNet(channels=args.channels, blocks=args.blocks)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location="cpu")["state_dict"],
                            strict=False)
        print(f"⚓ warm-start {args.init}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def save(tag, games):
        torch.save({"state_dict": net.state_dict(), "channels": args.channels,
                    "blocks": args.blocks, "games": games}, f"{exp_dir}/{tag}.pt")

    save("games_0", 0)
    cfg = dict(channels=args.channels, blocks=args.blocks,
               temperature=args.temperature, gamma=args.gamma,
               shaping=False, seed=args.seed)

    games, it, t0, log, next_ms = 0, 0, time.time(), [], 0
    while games < args.total_games:
        it += 1
        n_deals = max(1, args.games_per_iter // args.dup_k)
        base = 6_000_000 + it * 9973
        seeds = [base + d for d in range(n_deals) for _ in range(args.dup_k)]
        net.eval()
        episodes, results = collect_parallel(net, len(seeds), cfg, args.workers, seeds)
        apply_group_baseline(episodes, args.gamma)
        games += len(results)

        cat = lambda k: np.concatenate([e[k] for e in episodes])
        planes = torch.from_numpy(cat("planes"))
        scal = torch.from_numpy(cat("scalars"))
        mask = torch.from_numpy(cat("mask"))
        acts = torch.from_numpy(cat("actions"))
        rets = torch.from_numpy(cat("returns"))
        old_lp = torch.from_numpy(cat("old_logprobs"))

        if args.drop_zero_return:
            lens = [len(e["returns"]) for e in episodes]
            nz = [bool(np.abs(e["returns"]).max() > 1e-6) for e in episodes]
            idx_keep = torch.nonzero(torch.from_numpy(np.repeat(nz, lens)),
                                     as_tuple=True)[0]
        else:
            idx_keep = torch.arange(len(acts))
        n_eff = len(idx_keep)
        if n_eff < args.batch:
            continue

        with torch.no_grad():
            probe = idx_keep[torch.randperm(n_eff)[:2048]]
            lp0 = torch.log_softmax(net(planes[probe], scal[probe], mask[probe]), 1)
            s0 = torch.where(torch.isfinite(lp0), lp0, torch.zeros_like(lp0))
            ent_before = float(-(lp0.exp() * s0).sum(1).mean())
            # V(s) for the whole batch under the SAMPLING policy
            vals = torch.cat([
                net.forward_with_value(planes[i:i + 8192], scal[i:i + 8192],
                                       mask[i:i + 8192])[1]
                for i in range(0, len(acts), 8192)])
        adv_raw = rets - vals
        adv = ((adv_raw - adv_raw[idx_keep].mean())
               / (adv_raw[idx_keep].std() + 1e-8)).clamp(-5, 5)

        net.train()
        stop, passes, kls, closs, vloss = False, 0, [], [], []
        for ep in range(args.ppo_epochs):
            order = idx_keep[torch.randperm(n_eff)]
            pass_kl = []
            for lo in range(0, n_eff, args.batch):
                sel = order[lo:lo + args.batch]
                logits, v = net.forward_with_value(planes[sel], scal[sel], mask[sel])
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
                loss = pg + args.value_coef * vl - args.entropy_coef * ent
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
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

        win = sum(1 for r in results if "荣和" in r or "自摸" in r) / max(len(results), 1)
        el = time.time() - t0
        row = {"iter": it, "games": games, "wall_s": round(el, 1),
               "pg_loss": float(np.mean(closs)), "value_loss": float(np.mean(vloss)),
               "entropy_before": ent_before, "entropy": ent_after,
               "approx_kl": kls[-1] if kls else 0.0, "ppo_passes": passes,
               "explained_var": ev, "win_rate": win,
               "n_effective": n_eff, "n_raw": int(len(acts))}
        log.append(row)
        if it % 5 == 0 or it == 1:
            print(f"[{it:4d}] games={games:7d} {el/60:6.1f}min "
                  f"{games/max(el,1):5.1f}局/s pg={row['pg_loss']:+.4f} "
                  f"H={ent_before:.3f}->{ent_after:.3f} win={win:.1%} "
                  f"kl={row['approx_kl']:.4f} passes={passes} EV={ev:+.3f} "
                  f"eff={n_eff}/{len(acts)}", flush=True)
            json.dump(log, open(f"{exp_dir}/train_log.json", "w"), indent=1)

        while next_ms < len(milestones) and games >= milestones[next_ms]:
            save(f"games_{milestones[next_ms]}", games)
            print(f"📌 milestone {milestones[next_ms]} (actual {games})", flush=True)
            next_ms += 1

    save("games_final", games)
    json.dump(log, open(f"{exp_dir}/train_log.json", "w"), indent=1)
    print(f"✅ {games} games in {(time.time()-t0)/60:.1f} min -> {exp_dir}")


if __name__ == "__main__":
    main()
