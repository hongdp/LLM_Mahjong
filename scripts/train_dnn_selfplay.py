"""Train the conventional-DNN baseline by self-play policy gradient.

Same engine, same settlement reward, same gamma as the LLM runs, so the
arena comparison against the shared SFT anchor is apples-to-apples on
everything except the agent itself.

Two honest asymmetries to keep in mind when reporting:
  1. The DNN CANNOT emit an illegal action (the head is masked), so it
     never pays the format penalty the LLM sometimes pays. That is a real
     advantage of the conventional design, not a measurement artifact.
  2. The DNN plays orders of magnitude more games per wall-clock hour, so
     "same games" and "same hours" are very different comparisons. Both
     are logged (games_played, wall_clock_s) so either can be quoted.
"""

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.dnn.net import MahjongPolicyNet          # noqa: E402
from src.agents.dnn.selfplay import play_game, returns_to_go   # noqa: E402


def collect_batch(net, n_games, gamma, temperature, device, dup_k, base_seed):
    """Returns (steps, returns, stats). With dup_k>1 each wall is replayed
    dup_k times and the leave-one-out group mean of G0 is subtracted per
    (deal, seat) — the empirical baseline exp4's probe pointed us to."""
    games, seeds = [], []
    if dup_k > 1:
        n_deals = max(1, n_games // dup_k)
        for d in range(n_deals):
            for _ in range(dup_k):
                seeds.append(base_seed + d)
    else:
        seeds = [None] * n_games

    for s in seeds:
        games.append((play_game(net, temperature=temperature, device=device,
                                deal_seed=s), s))

    all_steps, all_returns, group_of = [], [], []
    per_ep = []
    for g, seed in games:
        for pid in range(4):
            steps = g.trajectories[pid]
            if not steps:
                continue
            rets = returns_to_go(steps, gamma)
            per_ep.append((steps, rets, (seed, pid)))

    if dup_k > 1:
        from collections import defaultdict
        groups = defaultdict(list)
        for i, (_, rets, key) in enumerate(per_ep):
            groups[key].append(i)
        for key, idxs in groups.items():
            if len(idxs) < 2:
                continue
            g0 = [per_ep[i][1][0] for i in idxs]
            total = sum(g0)
            for j, i in enumerate(idxs):
                loo = (total - g0[j]) / (len(idxs) - 1)
                steps, rets, key2 = per_ep[i]
                n = len(rets)
                d = loo / n
                fixed = []
                for t, r in enumerate(rets):
                    m = n - t
                    tail = m if gamma == 1.0 else (1 - gamma ** m) / (1 - gamma)
                    fixed.append(r - d * tail)
                per_ep[i] = (steps, fixed, key2)

    for steps, rets, _ in per_ep:
        all_steps.extend(steps)
        all_returns.extend(rets)

    results = [g.result or "" for g, _ in games]
    stats = {
        "games": len(games),
        "win_rate": sum(1 for r in results if "荣和" in r or "自摸" in r) / max(len(games), 1),
        "draw_rate": sum(1 for r in results if "流局" in r) / max(len(games), 1),
        "riichi_per_game": sum(
            1 for g, _ in games for pid in range(4) for s in g.trajectories[pid]
            if s.action_idx // 34 == 1) / max(len(games), 1),
        "meld_per_game": sum(
            1 for g, _ in games for pid in range(4) for s in g.trajectories[pid]
            if s.action_idx // 34 in (2, 3, 4)) / max(len(games), 1),
    }
    return all_steps, torch.tensor(all_returns, dtype=torch.float32), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--games_per_iter", type=int, default=32)
    ap.add_argument("--dup_k", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--exp_dir", default=None)
    ap.add_argument("--ckpt_every", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    exp_dir = args.exp_dir or f"experiments/dnn_baseline_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(exp_dir, exist_ok=True)
    json.dump(vars(args), open(f"{exp_dir}/config.json", "w"), indent=2)

    net = MahjongPolicyNet(channels=args.channels, blocks=args.blocks).to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"📦 params={sum(p.numel() for p in net.parameters()):,}  dir={exp_dir}")

    t0 = time.time()
    games_played = 0
    log = []
    for it in range(1, args.iters + 1):
        net.eval()
        steps, returns, stats = collect_batch(
            net, args.games_per_iter, args.gamma, args.temperature,
            args.device, args.dup_k, base_seed=1_000_000 + it * 997)
        games_played += stats["games"]
        if not steps:
            continue

        adv = (returns - returns.mean()) / (returns.std() + 1e-8)
        adv = adv.clamp(-5, 5).to(args.device)

        net.train()
        order = torch.randperm(len(steps))
        losses, ents = [], []
        for lo in range(0, len(order), args.batch):
            sel = order[lo:lo + args.batch]
            planes = torch.stack([steps[i].planes for i in sel]).to(args.device)
            scal = torch.stack([steps[i].scalars for i in sel]).to(args.device)
            mask = torch.stack([steps[i].mask for i in sel]).to(args.device)
            acts = torch.tensor([steps[i].action_idx for i in sel], device=args.device)
            a = adv[sel]

            logits = net(planes, scal, mask)
            logp = torch.log_softmax(logits, dim=1)
            chosen = logp.gather(1, acts[:, None]).squeeze(1)
            p = logp.exp()
            # entropy over legal actions only (masked entries are -inf -> p=0)
            ent = -(p * torch.where(torch.isfinite(logp), logp,
                                    torch.zeros_like(logp))).sum(1).mean()
            loss = -(chosen * a).mean() - args.entropy_coef * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(loss.item()); ents.append(ent.item())

        el = time.time() - t0
        row = {"iter": it, "games": games_played, "wall_s": round(el, 1),
               "loss": sum(losses) / len(losses), "entropy": sum(ents) / len(ents),
               "mean_return": returns.mean().item(), **stats}
        log.append(row)
        if it % 10 == 0 or it == 1:
            print(f"[{it:4d}] games={games_played:6d} {el/60:6.1f}min "
                  f"loss={row['loss']:+.4f} H={row['entropy']:.3f} "
                  f"win={stats['win_rate']:.1%} draw={stats['draw_rate']:.1%} "
                  f"riichi/局={stats['riichi_per_game']:.2f} 副露/局={stats['meld_per_game']:.2f}",
                  flush=True)
        if it % args.ckpt_every == 0 or it == args.iters:
            torch.save({"state_dict": net.state_dict(),
                        "channels": args.channels, "blocks": args.blocks,
                        "iter": it, "games": games_played},
                       f"{exp_dir}/ckpt_iter{it}.pt")
            json.dump(log, open(f"{exp_dir}/train_log.json", "w"), indent=1)
    json.dump(log, open(f"{exp_dir}/train_log.json", "w"), indent=1)
    print(f"✅ done: {games_played} games in {(time.time()-t0)/60:.1f} min -> {exp_dir}")


if __name__ == "__main__":
    main()
