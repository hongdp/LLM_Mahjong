"""exp112: host-side placement credit for Rust mirror hanchan episodes (`--hanchan_credit wpure`).

W(state) = rank-uma analytic potential + MLP residual, FITTED ON PURE SELF-PLAY (scripts/fit_pure_w.py) — the
same PlacementValue family as exp55-D's human-fitted W, but zero human data. Per deal the seat's reward on its
LAST step of that deal becomes W(after) - W(before); the final deal pays true_uma - W(before), so every seat's
rewards telescope exactly to uma - W(match start) (tests/test_wpure_credit.py). Deals in which a seat never
acted roll their credit into the seat's next paid step; any remainder lands on its final step.

Feature layout == src/tasks/mahjong/hanchan.py::PlacementCredit (12 dims): rel points/1e5 x4, round_idx/8,
honba/8, kyotaku/4000, dealer-relative one-hot x4, deals_left/8, with round_idx = rw*4 + dealer (the Rust mirror
match always starts with dealer 0) and deals_left = max(0, 8 - round_idx).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from src.tasks.mahjong.hanchan import _rank_uma_single

REWARD_SCALE = 0.001
W_DIM = 12


def deal_features(points, dealer: int, rw: int, honba: int, kyotaku: int, me: int) -> np.ndarray:
    rel = [points[(me + k) % 4] for k in range(4)]
    drel = (dealer - me) % 4
    round_idx = rw * 4 + (dealer % 4)
    x = np.zeros(W_DIM, dtype=np.float32)
    x[:4] = [v / 1e5 for v in rel]
    x[4] = round_idx / 8.0; x[5] = honba / 8.0; x[6] = kyotaku / 4000.0
    x[7 + drel] = 1.0
    x[11] = max(0, 8 - round_idx) / 8.0
    return x


def rank_uma(points, me: int) -> float:
    return float(_rank_uma_single([points[(me + k) % 4] for k in range(4)]))


class PureW:
    """W(state, seat) in POINTS. `path` -> PlacementValue checkpoint (d_in 12, uma head in k-points residual)."""

    def __init__(self, path: Optional[str], device: str = "cpu"):
        self.net = None
        self.device = device
        if path:
            from scripts.train_placement_value import PlacementValue
            blob = torch.load(path, map_location=device)
            net = PlacementValue(d_in=blob["d_in"]); net.load_state_dict(blob["state_dict"]); net.eval()
            self.net = net.to(device)

    @torch.no_grad()
    def batch(self, feats: np.ndarray, base: np.ndarray) -> np.ndarray:
        """feats [N,12], base [N] rank-uma points -> W points [N]. net=None: analytic rank-uma only (exp70's RankUmaCredit)."""
        if self.net is None or len(feats) == 0:
            return base.astype(np.float64)
        _, u = self.net(torch.from_numpy(feats).to(self.device))
        return base.astype(np.float64) + u.float().cpu().numpy().astype(np.float64) * 1000.0


def rewrite_wpure(games: List[dict], w: PureW, gamma: float) -> Dict[str, float]:
    """In place: replace every episode's rewards/returns of the given finished hanchan games by W-credit.
    Returns diagnostics (mean |credit| etc.)."""
    # 1) gather every (game, deal, seat) W query in one batch
    feats, base, keys = [], [], []
    for gi, g in enumerate(games):
        deals = g.get("deals") or []
        for di, d in enumerate(deals):
            for seat in range(4):
                feats.append(deal_features(d["start_points"], d["dealer"], d["rw"], d["honba"], d["kyotaku"], seat))
                base.append(rank_uma(d["start_points"], seat)); keys.append((gi, di, seat))
    W = w.batch(np.stack(feats) if feats else np.zeros((0, W_DIM), np.float32), np.asarray(base))
    Wmap = {k: v for k, v in zip(keys, W)}
    n_pay, tot_abs, n_rolled = 0, 0.0, 0
    for gi, g in enumerate(games):
        deals = g.get("deals") or []; nd = len(deals)
        uma = g["hanchan"]["uma_points"]; ends = g["deal_ends"]
        for e in g["episodes"]:
            seat = int(e["key"][1]); T = len(e["rewards"])
            r = np.zeros(T, dtype=np.float32)
            paid_at = {di: step for di, step in ends[seat]}
            pending = 0.0; last_step = None
            for di in range(nd):
                w_before = Wmap[(gi, di, seat)]
                w_after = Wmap[(gi, di + 1, seat)] if di + 1 < nd else float(uma[seat])
                pending += (w_after - w_before)
                step = paid_at.get(di)
                if step is not None:
                    r[step] += pending * REWARD_SCALE; tot_abs += abs(pending); n_pay += 1
                    pending = 0.0; last_step = step
                else:
                    n_rolled += 1
            if pending != 0.0:
                r[last_step if last_step is not None else T - 1] += pending * REWARD_SCALE
            e["rewards"] = r
            ret = np.zeros(T, dtype=np.float32); acc = 0.0
            for t in range(T - 1, -1, -1):
                acc = r[t] + gamma * acc; ret[t] = acc
            e["returns"] = ret
    return {"paid": n_pay, "mean_abs_credit_pts": tot_abs / max(n_pay, 1), "rolled": n_rolled}
