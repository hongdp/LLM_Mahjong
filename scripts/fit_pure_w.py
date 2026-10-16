"""exp112: fit the placement potential W on PURE self-play (zero human data).

Plays N mirror hanchan matches with the given policy (Rust VecEnv, credit none), records every (deal start
state, seat) -> final uma, and fits PlacementValue (rank-uma analytic + MLP residual, same family/feature layout
as exp55-D's human W). Holdout split by match seed; reports MAE vs the rank-uma and zero baselines.
usage: PYTHONPATH=. python scripts/fit_pure_w.py --policy <ckpt> --out experiments/placement_value/w_pure.pt --matches 300000
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agents.dnn.hanchan_credit import deal_features, rank_uma, W_DIM        # noqa: E402
from scripts.train_placement_value import PlacementValue                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--matches", type=int, default=300000)
    ap.add_argument("--per_iter", type=int, default=2048)
    ap.add_argument("--games_per_worker", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=12_000_000)
    ap.add_argument("--data_out", default=None, help="also save the (X, y, seed) arrays as .npz")
    ap.add_argument("--from_npz", default=None, help="skip collection; load X/y/base/seed from this .npz")
    ap.add_argument("--init", default=None, help="continue training from this PlacementValue checkpoint")
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    from src.agents.dnn.rust_rollout import collect_rust
    from scripts.run_elo_league import load_dnn
    if args.from_npz:
        z = np.load(args.from_npz); X, Y, B, S = z["X"], z["y"], z["base"], z["seed"]
        print(f"loaded {len(Y)} rows from {args.from_npz}", flush=True)
        return fit(args, X, Y, B, S)
    net = load_dnn(args.policy, "cuda").eval()
    cfg = {"temperature": 1.0, "gamma": 0.995, "games_per_worker": args.games_per_worker,
           "encoder_variant": getattr(net, "encoder_variant", "v3r"), "hanchan": True, "hanchan_pure": True,
           "hanchan_credit": "none", "shaping": False}
    X, Y, B, S = [], [], [], []
    t0 = time.time(); done = 0
    while done < args.matches:
        n = min(args.per_iter, args.matches - done)
        seeds = [args.seed0 + done + i for i in range(n)]
        collect_rust(net, n, cfg, 1, seeds, device="cuda")
        for g in collect_rust.last_games:
            uma = g["hanchan"]["uma_points"]
            for d in g["deals"]:
                for seat in range(4):
                    X.append(deal_features(d["start_points"], d["dealer"], d["rw"], d["honba"], d["kyotaku"], seat))
                    B.append(rank_uma(d["start_points"], seat)); Y.append(float(uma[seat])); S.append(int(g["seed"]))
        done += n
        el = time.time() - t0
        print(f"[{done}/{args.matches}] {done/el:.0f} matches/s  rows {len(Y)}", flush=True)
    X = np.stack(X).astype(np.float32); Y = np.asarray(Y, np.float32); B = np.asarray(B, np.float32); S = np.asarray(S)
    if args.data_out:
        np.savez_compressed(args.data_out, X=X, y=Y, base=B, seed=S)
    return fit(args, X, Y, B, S)


def fit(args, X, Y, B, S):
    hold = (S % 10) == 0
    resid = (Y - B) / 1000.0                                            # residual target in k-points (as exp55)
    dev = torch.device("cuda")
    model = PlacementValue(d_in=W_DIM).to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev)["state_dict"]); print(f"init from {args.init}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    Xt, Rt = torch.from_numpy(X[~hold]).to(dev), torch.from_numpy(resid[~hold]).to(dev)
    Xh, Rh = torch.from_numpy(X[hold]).to(dev), torch.from_numpy(resid[hold]).to(dev)
    # placement labels for the auxiliary head (final rank from uma sign/order is not recoverable per row; use uma bucket)
    def rank_from_uma(u):  # uma_points = final-25000+UMA: 1st >= +15000-ish ... use nearest UMA bucket of the bonus part
        return torch.zeros_like(u, dtype=torch.long)
    n = len(Xt); bs = 8192
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev); tot = 0.0
        for lo in range(0, n, bs):
            sel = perm[lo:lo + bs]
            _, u = model(Xt[sel]); loss = torch.nn.functional.smooth_l1_loss(u, Rt[sel])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(sel)
        with torch.no_grad():
            _, uh = model(Xh)
            mae = (uh - Rh).abs().mean().item() * 1000
        print(f"epoch {ep+1}: train {tot/n:.4f}  holdout MAE {mae:.0f} pts  (rank-uma baseline {np.abs(Y[hold]-B[hold]).mean():.0f}, zero baseline {np.abs(Y[hold]).mean():.0f})", flush=True)
    with torch.no_grad():
        _, uh = model(Xh); mae = float((uh - Rh).abs().mean().item() * 1000)
    base_mae = float(np.abs(Y[hold] - B[hold]).mean()); zero_mae = float(np.abs(Y[hold]).mean())
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "d_in": W_DIM, "policy": args.policy,
                "matches": args.matches, "init": args.init, "rows": int(len(Y)), "holdout_mae": mae, "rank_uma_mae": base_mae, "zero_mae": zero_mae}, args.out)
    print(json.dumps({"out": args.out, "rows": int(len(Y)), "holdout_mae": mae, "rank_uma_mae": base_mae, "zero_mae": zero_mae,
                      "improvement_vs_rank_uma": round(1 - mae / base_mae, 4)}))


if __name__ == "__main__":
    main()
