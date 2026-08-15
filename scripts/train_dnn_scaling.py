"""Does MORE self-play RL buy strength? Measure the scaling curve.

Motivating question (user, 2026-08-14): "what if we just use an enormous
number of RL games to beat the variance?" For the LLM that is unanswerable
— 600 games cost 27 A100-hours. For the conventional DNN it is a few
hours, so this script answers it empirically instead of by argument:
train from a fixed init and snapshot at exponentially spaced game counts,
so each snapshot can be played off against the init in a big arena.

Reads the same reward/discount/group-baseline setup as every other run so
the resulting points sit on the same axis as the LLM results.
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
    ap.add_argument("--init", default=None, help="warm-start checkpoint")
    ap.add_argument("--total_games", type=int, default=320000)
    ap.add_argument("--games_per_iter", type=int, default=256)
    ap.add_argument("--dup_k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--entropy_coef", type=float, default=0.02)
    ap.add_argument("--arch", default=None,
                    help="arch-zoo model name (e.g. vit_small); overrides "
                         "channels/blocks")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4096,
                    help="MICRO-batch for memory only. The gradient is "
                         "accumulated over EVERY effective sample in the "
                         "iteration, so the statistical batch size is the "
                         "effective-sample count, not this number.")
    ap.add_argument("--shaping", action="store_true")
    ap.add_argument("--drop_zero_return", action="store_true",
                    help="Skip episodes whose return is exactly 0 (an "
                         "exhaustive draw with no point transfer). Measured: "
                         "77%% of episodes, but only 2%% of the |advantage| "
                         "mass — dropping them makes the few gradient steps "
                         "per iteration land on signal-bearing data. Loss is "
                         "renormalised by the ORIGINAL sample count so the "
                         "gradient magnitude (and hence the effective lr) is "
                         "unchanged.")
    ap.add_argument("--milestones", type=str,
                    default="5000,20000,80000,320000",
                    help="cumulative game counts to snapshot at")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_device",
                    default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device for the UPDATE only. Rollout always stays on "
                         "CPU worker processes: batch-1 forwards of a 1.4M net "
                         "are launch-bound, so 16 CPU processes beat one GPU. "
                         "The update is the opposite shape (batch 4-8k) and "
                         "measured 148x faster on GPU (6439ms -> 43.6ms/step).")
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

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    exp_dir_guard = None
    exp_dir = args.exp_dir or f"experiments/dnn_scaling_{time.strftime('%Y%m%d_%H%M%S')}"
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
    if args.arch:
        from src.agents.dnn.arch_zoo import ZOO
        net = ZOO[args.arch][0]().to(dev)
        print(f"🏗 arch: {args.arch}", flush=True)
    else:
        net = MahjongPolicyNet(channels=args.channels, blocks=args.blocks).to(dev)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location="cpu")["state_dict"])
        print(f"⚓ warm-start {args.init}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def save(tag, games, it=0):
        torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                    "channels": args.channels, "blocks": args.blocks,
                    "arch": args.arch,
                    "games": games, "iter": it,
                    "optimizer": opt.state_dict()}, f"{exp_dir}/{tag}.pt")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.join(exp_dir, "tensorboard"))

    start_games, start_iter, log = 0, 0, []
    if args.resume:
        blob = torch.load(args.resume, map_location=dev)
        net.load_state_dict(blob["state_dict"])
        if "optimizer" in blob:
            opt.load_state_dict(blob["optimizer"])
            print("   optimizer moments restored", flush=True)
        else:
            print("   (checkpoint has no optimizer state; Adam moments restart)",
                  flush=True)
        start_games = int(blob.get("games", 0))
        start_iter = int(blob.get("iter", 0))
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
                    writer.add_scalar(k, float(v), int(r["games"]))
    else:
        save("games_0", 0)          # the init itself is the arena's reference
    cfg = dict(channels=args.channels, blocks=args.blocks, arch=args.arch,
               temperature=args.temperature, gamma=args.gamma,
               shaping=args.shaping, seed=args.seed)

    games, it, t0 = start_games, start_iter, time.time()
    next_ms = 0
    while next_ms < len(milestones) and milestones[next_ms] <= start_games:
        next_ms += 1
    while games < args.total_games:
        it += 1
        n_deals = max(1, args.games_per_iter // args.dup_k)
        base = 5_000_000 + it * 9973
        seeds = [base + d for d in range(n_deals) for _ in range(args.dup_k)]
        net.eval()
        episodes, results = collect_parallel(net, len(seeds), cfg,
                                             args.workers, seeds)
        apply_group_baseline(episodes, args.gamma)
        games += len(results)

        planes = torch.from_numpy(np.concatenate([e["planes"] for e in episodes])).to(dev)
        scal = torch.from_numpy(np.concatenate([e["scalars"] for e in episodes])).to(dev)
        mask = torch.from_numpy(np.concatenate([e["mask"] for e in episodes])).to(dev)
        acts = torch.from_numpy(np.concatenate([e["actions"] for e in episodes])).to(dev)
        rets = torch.from_numpy(np.concatenate([e["returns"] for e in episodes])).to(dev)
        adv = ((rets - rets.mean()) / (rets.std() + 1e-8)).clamp(-5, 5)

        with torch.no_grad():
            # entropy of the SAMPLING policy, before any update touches it
            probe = torch.randperm(len(acts), device=dev)[:2048]
            lp0 = torch.log_softmax(net(planes[probe], scal[probe], mask[probe]), 1)
            safe0 = torch.where(torch.isfinite(lp0), lp0, torch.zeros_like(lp0))
            ent_before = float(-(lp0.exp() * safe0).sum(1).mean())

        net.train()
        # --- select the EFFECTIVE samples ---------------------------------
        # An exhaustive draw with no point transfer has return exactly 0 and
        # therefore contributes exactly zero gradient (verified: filtered vs
        # full-batch gradient cosine 1.000, norm ratio 1.0002). Measured on
        # a random policy: 77% of episodes, 2% of the |advantage| mass.
        if args.drop_zero_return:
            lens = [len(e["returns"]) for e in episodes]
            nonzero_ep = [bool(np.abs(e["returns"]).max() > 1e-6) for e in episodes]
            idx_keep = torch.nonzero(
                torch.from_numpy(np.repeat(nonzero_ep, lens)).to(dev),
                as_tuple=True)[0]
        else:
            idx_keep = torch.arange(len(acts), device=dev)
        n_eff = len(idx_keep)
        if n_eff == 0:
            continue
        order = idx_keep[torch.randperm(n_eff, device=dev)]

        # --- ONE update per iteration over ALL effective samples ----------
        # Gradient accumulated in micro-batches; the statistical batch size
        # is n_eff. Normalising by n_eff (not by the raw sample count) keeps
        # the update size independent of the draw rate, which falls as the
        # policy improves — dividing by the raw count instead would silently
        # inflate the effective learning rate over training.
        opt.zero_grad()
        losses, ents = [], []
        for lo in range(0, n_eff, args.batch):
            sel = order[lo:lo + args.batch]
            logits = net(planes[sel], scal[sel], mask[sel])
            logp = torch.log_softmax(logits, dim=1)
            chosen = logp.gather(1, acts[sel][:, None]).squeeze(1)
            safe = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))
            ent = -(logp.exp() * safe).sum(1).mean()
            w = len(sel) / n_eff                      # micro-batch weight
            loss = (-(chosen * adv[sel]).mean() - args.entropy_coef * ent) * w
            loss.backward()
            losses.append(loss.item() / max(w, 1e-9)); ents.append(ent.item())
        gnorm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        opt.step()

        with torch.no_grad():
            # measured AFTER the step, on the SAME probe rows as ent_before —
            # during accumulation the weights never move, so the per-micro-
            # batch entropies say nothing about the update's effect.
            net.eval()
            lp1 = torch.log_softmax(net(planes[probe], scal[probe], mask[probe]), 1)
            safe1 = torch.where(torch.isfinite(lp1), lp1, torch.zeros_like(lp1))
            ent_after = float(-(lp1.exp() * safe1).sum(1).mean())
            net.train()

        # NOTE: this is the DECISIVE-GAME rate (a hand ended in ron/tsumo by
        # anyone) = 1 - draw rate. It is NOT a per-agent win rate: all four
        # seats share these weights, so a per-agent figure is roughly this
        # divided by four (measured: 77.5% decisive <-> 19.4% per seat).
        # Ceiling is 100%, not 25%. It tracks how fast the shared policy
        # completes hands, i.e. tile efficiency — competitive strength only
        # comes from the arena, where different policies actually meet.
        win = sum(1 for r in results if "荣和" in r or "自摸" in r) / max(len(results), 1)
        el = time.time() - t0
        row = {"iter": it, "games": games, "wall_s": round(el, 1),
               "loss": float(np.mean(losses)), "entropy": ent_after,
               "entropy_before": ent_before,
               "win_rate": win, "mean_return": float(rets.mean()),
               "n_effective": n_eff, "n_raw": int(len(acts)),
               "eff_frac": n_eff / max(len(acts), 1), "grad_norm": gnorm}
        log.append(row)
        for k, v in row.items():
            if k not in ("iter", "games", "wall_s") and isinstance(v, (int, float)):
                writer.add_scalar(k, float(v), games)
        writer.add_scalar("games_per_sec",
                          (games - start_games) / max(el, 1e-9), games)
        writer.flush()
        if it % 5 == 0 or it == 1:
            print(f"[{it:4d}] games={games:7d} {el/60:6.1f}min "
                  f"{(games-start_games)/max(el,1):5.1f}局/s loss={row['loss']:+.4f} "
                  f"H={ent_before:.3f}->{row['entropy']:.3f} win={win:.1%} "
                  f"eff={n_eff}/{len(acts)} ({row['eff_frac']:.0%}) "
                  f"|g|={gnorm:.2f}", flush=True)
            json.dump(log, open(f"{exp_dir}/train_log.json", "w"), indent=1)

        if args.ckpt_every and it % args.ckpt_every == 0:
            save("latest", games, it)

        while next_ms < len(milestones) and games >= milestones[next_ms]:
            save(f"games_{milestones[next_ms]}", games, it)
            print(f"📌 milestone {milestones[next_ms]} games -> checkpoint "
                  f"(actual {games})", flush=True)
            next_ms += 1

    save("games_final", games, it)
    json.dump(log, open(f"{exp_dir}/train_log.json", "w"), indent=1)
    writer.close()
    print(f"✅ {games} games in {(time.time()-t0)/60:.1f} min -> {exp_dir}")


if __name__ == "__main__":
    main()
