"""exp109 Stage 0: train SEPARATE Q critics to convergence against a FROZEN policy (no PPO), locally, $0.

Three critics share the recipe (policy-trunk warm start, own Adam, several passes per fresh batch):
  BQ   blind Q(s,a)                       target R_t
  OQ   oracle Q(s,a) (319-dim hidden vec)  target R_t
  ODQ  oracle Q_rest(s,a) + engine ron cost: target R_t - c_ron(s,a_t); Q_full = Q_rest + c_ron  (c_ron exact for ALL discards)

The gate (exp108 lesson) is RANKING quality, not EV: on exposed turn decisions (an opponent in riichi, both wait and
non-wait discards legal) — argmax-avoids-waits rate vs the policy's own, Q(wait) - Q(safe) gap; on all multi-action
states — agreement with the policy argmax; plus held-out EV of Q(s, a_t). Diagnostics run on each fresh batch BEFORE
training on it (so they are out-of-sample) every --diag_every iterations. Checkpoints + metrics.jsonl in --exp_dir.
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
import torch

from src.agents.dnn.encoder import TILE_TYPES, TYPE_TO_ID
from src.agents.dnn.oracle_critic import SeparateCritic, explained_variance
from src.agents.dnn.rust_rollout import collect_rust
from scripts.run_elo_league import load_dnn

D_TYPES = (TYPE_TO_ID["discard"], TYPE_TO_ID["discard0"])


def ron_cost_full(ron68: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B,68] per-tile cost ([0:34] plain, [34:68] red) -> [B,374] per-action cost on the discard / discard0 slots."""
    B = ron68.shape[0]
    out = torch.zeros(B, mask.shape[1], device=ron68.device)
    t0, t1 = TYPE_TO_ID["discard"], TYPE_TO_ID["discard0"]
    out[:, t0 * TILE_TYPES:(t0 + 1) * TILE_TYPES] = ron68[:, :TILE_TYPES]
    out[:, t1 * TILE_TYPES:(t1 + 1) * TILE_TYPES] = ron68[:, TILE_TYPES:]
    return out * mask


def diagnostics(net, crits, planes, scal, mask, cfe, ron68, acts, rets, dev):
    out = {}
    with torch.no_grad():
        lg = torch.cat([net(planes[i:i + 8192], scal[i:i + 8192], mask[i:i + 8192]).float() for i in range(0, len(acts), 8192)])
        cfull = ron_cost_full(ron68, mask)
        waits = (cfe[:, 111:213].view(-1, 3, TILE_TYPES).amax(1) > 0.5)                       # any riichi opponent waits on tile
        disc = mask.view(-1, 11, TILE_TYPES)[:, list(D_TYPES), :].any(1)                         # discardable tiles
        exposed = (cfe[:, 111:213].sum(1) > 0) & (disc.sum(1) >= 2) & (waits & disc).any(1) & (~waits & disc).any(1)
        multi = mask.sum(1) > 1
        pa = lg.argmax(1)
        pol_avoid = (~waits[exposed].gather(1, (pa[exposed] % TILE_TYPES)[:, None]).squeeze(1)).float().mean()
        out["n_exposed"] = int(exposed.sum()); out["n_multi"] = int(multi.sum()); out["policy_avoid_waits"] = float(pol_avoid)
        for name, c in crits.items():
            q = torch.cat([c(planes[i:i + 8192], scal[i:i + 8192], cfe[i:i + 8192]).float() for i in range(0, len(acts), 8192)])
            if name == "ODQ":
                q = q - cfull                                                                     # Q_full = Q_rest + c_ron (c_ron <= 0 as a cost -> subtract)
            qm = q.masked_fill(~mask, float("-inf"))
            qa = qm.argmax(1)
            out[f"{name}/ev"] = explained_variance(q.gather(1, acts[:, None]).squeeze(1), rets)
            out[f"{name}/agree_policy"] = float((qa == pa)[multi].float().mean())
            qt = qm.view(-1, 11, TILE_TYPES)[:, list(D_TYPES), :].amax(1)                       # best per tile
            qt = qt.masked_fill(~disc, float("nan"))
            ex = exposed
            gap = torch.nanmean(torch.where(waits[ex], qt[ex], torch.nan), 1) - torch.nanmean(torch.where(~waits[ex], qt[ex], torch.nan), 1)
            out[f"{name}/wait_gap_pts"] = float(torch.nanmean(gap)) * 1000.0
            qa_t = qt[ex].nan_to_num(float("-inf")).argmax(1)
            out[f"{name}/avoid_waits"] = float((~waits[ex].gather(1, qa_t[:, None]).squeeze(1)).float().mean())
            # top-2 margin inside exposed states (how decisive the ranking is)
            top2 = torch.topk(qt[ex].nan_to_num(float("-inf")), 2, dim=1).values
            out[f"{name}/top2_gap_pts"] = float((top2[:, 0] - top2[:, 1]).median()) * 1000.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--games_per_iter", type=int, default=2048)
    ap.add_argument("--games_per_worker", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--diag_every", type=int, default=25)
    ap.add_argument("--ckpt_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1091)
    ap.add_argument("--arms", default="BQ,OQ,ODQ")
    args = ap.parse_args()
    os.makedirs(args.exp_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.exp_dir, "config.json"), "w"), indent=1)
    dev = torch.device("cuda")
    net = load_dnn(args.policy, "cuda").eval()
    for p in net.parameters():
        p.requires_grad_(False)
    trunk = {k: v for k, v in net.state_dict().items() if k.startswith(("stem.", "blocks.", "scalar_fc."))}
    crits, opts = {}, {}
    for name in args.arms.split(","):
        c = SeparateCritic("q", args.channels, args.blocks, blind=(name == "BQ")).to(dev)
        c.load_state_dict(trunk, strict=False)
        crits[name] = c; opts[name] = torch.optim.Adam(c.parameters(), lr=args.lr)
    print(f"critics {list(crits)} params {sum(p.numel() for p in crits['BQ' if 'BQ' in crits else list(crits)[0]].parameters())/1e6:.2f}M each", flush=True)
    cfg = {"temperature": 1.0, "gamma": 0.995, "games_per_worker": args.games_per_worker, "encoder_variant": "v3r",
           "oracle_feats": True, "ron_cost": True}
    import riichi_rs
    q = float(riichi_rs.PLANE_Q)
    mlog = open(os.path.join(args.exp_dir, "metrics.jsonl"), "a")
    t0 = time.time(); games = 0
    for it in range(1, args.iters + 1):
        base = 9_500_000 + it * 9973 + args.seed
        seeds = [base + d for d in range(args.games_per_iter)]
        tr = time.time()
        eps, _ = collect_rust(net, len(seeds), cfg, 1, seeds, device="cuda")
        games += len(seeds)
        cat = lambda k: np.concatenate([e[k] for e in eps])
        planes = torch.from_numpy(cat("planes")).to(dev).float().div_(q)
        scal = torch.from_numpy(cat("scalars")).to(dev); mask = torch.from_numpy(cat("mask")).to(dev)
        acts = torch.from_numpy(cat("actions")).to(dev).long(); rets = torch.from_numpy(cat("returns")).to(dev).float()
        cfe = torch.from_numpy(cat("cfeats")).to(dev).float(); ron68 = torch.from_numpy(cat("ron_cost")).to(dev).float()
        roll_s = time.time() - tr
        row = {"iter": it, "games": games}
        if it % args.diag_every == 1:
            for c in crits.values(): c.eval()
            row.update(diagnostics(net, crits, planes, scal, mask, cfe, ron68, acts, rets, dev))
        # targets: taken-action cost for ODQ
        c_taken = ron_cost_full(ron68, mask).gather(1, acts[:, None]).squeeze(1)
        tu = time.time(); N = len(acts)
        for name, c in crits.items():
            c.train(); losses = []
            target = rets + c_taken if name == "ODQ" else rets                                  # Q_rest learns what is left after the exact ron cost
            for _ in range(args.epochs):
                order = torch.randperm(N, device=dev)
                for lo in range(0, N, args.batch):
                    sel = order[lo:lo + args.batch]
                    with torch.autocast("cuda", torch.bfloat16):
                        out = c(planes[sel], scal[sel], cfe[sel])
                    pred = out.float().gather(1, acts[sel][:, None]).squeeze(1)
                    loss = torch.nn.functional.mse_loss(pred, target[sel])
                    opts[name].zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(c.parameters(), 1.0); opts[name].step(); losses.append(loss.item())
            row[f"{name}/loss"] = float(np.mean(losses))
        row["rollout_s"] = round(roll_s, 1); row["update_s"] = round(time.time() - tu, 1); row["wall_s"] = round(time.time() - t0, 1)
        mlog.write(json.dumps(row) + "\n"); mlog.flush()
        if it % args.diag_every == 1 or it % 10 == 0:
            print(f"[{it}] games={games} {len(seeds)/roll_s:.0f}局/s  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items() if "/" in k or k in ("policy_avoid_waits",)), flush=True)
        if it % args.ckpt_every == 0 or it == args.iters:
            for name, c in crits.items():
                torch.save({"state_dict": {k: v.cpu() for k, v in c.state_dict().items()}, "optimizer": opts[name].state_dict(),
                            "mode": "q", "blind": c.blind, "channels": args.channels, "blocks": args.blocks, "games": games,
                            "decomposed": name == "ODQ"}, os.path.join(args.exp_dir, f"critic_{name}_latest.pt"))
    print("✅ stage0 done", flush=True)


if __name__ == "__main__":
    main()
