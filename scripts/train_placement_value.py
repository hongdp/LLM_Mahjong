"""Placement-value W for exp55-D hanchan credit assignment.

Small MLP on between-deal match states -> E[final uma | state] (per-deal
credit is the telescoping difference W(after) - W(before)). Trained on
human houou hanchan (extract_placement_states.py); holdout split by game
hash. Two heads share the trunk: placement 4-way CE (calibration
diagnostics) + uma regression (the reward signal).

Baselines the model must beat on holdout MAE:
  rank-uma: assign UMA by the CURRENT score ranking + own delta-from-25k
  mean:     predict 0 (uma is zero-mean by construction)
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UMA = [15000.0, 5000.0, -5000.0, -15000.0]


class PlacementValue(nn.Module):
    def __init__(self, d_in=12, width=128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, width), nn.ReLU(),
                                   nn.Linear(width, width), nn.ReLU())
        self.place = nn.Linear(width, 4)
        self.uma = nn.Linear(width, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.place(h), self.uma(h).squeeze(-1)


def rank_uma_baseline(X):
    """UMA by current score order (self = column 0; rel scores cols 0-3)
    + own points-25k, in uma units."""
    scores = X[:, :4] * 1e5
    self_rank = (scores[:, 1:] > scores[:, :1]).sum(1)  # ties favour self
    uma = np.array(UMA)[self_rank]
    return uma + (scores[:, 0] - 25000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="experiments/placement_value/states.npz")
    ap.add_argument("--out", default="experiments/placement_value/w_mlp.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--residual", action="store_true",
                    help="fit y - rank_uma(x): late game is a near-exact "
                         "piecewise formula rank-uma already encodes; the "
                         "net only learns the early/mid-game contextual "
                         "correction (stage diag 2026-08-30: plain MLP was "
                         "-1043 pts WORSE than rank-uma at S4)")
    args = ap.parse_args()

    d = np.load(args.data)
    X, y, g = d["X"], d["y"], d["g"]
    if args.residual:
        y = y - rank_uma_baseline(X).astype(np.float32)
    hold = (g % 100) < 10
    Xtr, ytr = X[~hold], y[~hold]
    Xho, yho = X[hold], y[hold]
    print(f"train {len(ytr)} holdout {len(yho)}")

    dev = torch.device(args.device)
    net = PlacementValue(d_in=X.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    Xt = torch.from_numpy(Xtr).to(dev)
    yt = torch.from_numpy(ytr).to(dev) / 1000.0          # k-points scale
    Xh = torch.from_numpy(Xho).to(dev)
    yh = torch.from_numpy(yho).to(dev) / 1000.0

    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(yt), device=dev)
        tot = 0.0
        for lo in range(0, len(yt), args.batch):
            sel = perm[lo:lo + args.batch]
            _, u = net(Xt[sel])
            loss = nn.functional.smooth_l1_loss(u, yt[sel])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(sel)
        net.eval()
        with torch.no_grad():
            _, uh = net(Xh)
            mae = float((uh - yh).abs().mean()) * 1000
        # in residual mode this MAE is already "W = base + net" vs true uma
        print(f"ep{ep}: train_huber={tot/len(yt):.4f} holdout_MAE={mae:.0f} pts",
              flush=True)

    base = rank_uma_baseline(Xho)
    print(f"baseline rank-uma MAE: {np.abs(base - yho).mean():.0f} pts")
    print(f"baseline zero     MAE: {np.abs(yho).mean():.0f} pts")
    torch.save({"state_dict": net.state_dict(), "d_in": X.shape[1],
                "scale": 1000.0}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
