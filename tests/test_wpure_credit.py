"""exp112: host-side W credit on Rust mirror hanchan episodes.
(1) deal_ends export is consistent (one entry per deal the seat acted in, indices inside the episode);
(2) with the analytic W (no MLP) every seat's rewritten rewards telescope to uma - W(match start) and the
    recomputed returns are the gamma-discounted suffix sums; no reward lands outside the deal-end steps;
(3) the credit-none path still telescopes to the final margin + uma (regression guard for the Rust change)."""
import os

import numpy as np
import pytest
import torch

riichi_rs = pytest.importorskip("riichi_rs")
from src.agents.dnn.hanchan_credit import PureW, rank_uma, rewrite_wpure, REWARD_SCALE  # noqa: E402

CK = "/home/hongdp/Workspace/LLM_Mahjong/experiments/exp88_Q1x/games_final.pt"


def _games(credit):
    from src.agents.dnn.rust_rollout import collect_rust
    from scripts.run_elo_league import load_dnn
    net = load_dnn(CK, "cuda").eval()
    cfg = {"temperature": 1.0, "gamma": 0.995, "games_per_worker": 16, "encoder_variant": "v3r",
           "hanchan": True, "hanchan_pure": True, "hanchan_credit": credit, "hanchan_w_path": None, "shaping": False}
    eps, _ = collect_rust(net, 24, cfg, 1, seeds=[7_400_000 + i for i in range(24)], device="cuda")
    return eps, collect_rust.last_games


@pytest.mark.skipif(not (os.path.exists(CK) and torch.cuda.is_available()), reason="needs a cnn_m_v3r checkpoint + CUDA")
def test_deal_ends_and_credit_none_invariant():
    eps, games = _games("none")
    for g in games:
        nd = len(g["deals"]); assert nd == g["hanchan"]["n_deals"]
        for e in g["episodes"]:
            seat = int(e["key"][1]); T = len(e["rewards"]); ends = g["deal_ends"][seat]
            assert len(ends) <= nd and all(0 <= di < nd and 0 <= st < T for di, st in ends)
            assert [di for di, _ in ends] == sorted({di for di, _ in ends})      # one entry per deal, increasing
            assert ends[-1][1] == T - 1                                             # the seat's last step closes its last deal
            # credit none: sum of rewards == final margin + UMA bonus (= uma_points) in reward units
            assert abs(float(e["rewards"].sum()) - g["hanchan"]["uma_points"][seat] * REWARD_SCALE) < 1e-4
        # deal facts carry the match context and outcome
        d0 = g["deals"][0]; assert d0["dealer"] == 0 and d0["rw"] == 0 and d0["honba"] == 0 and d0["n"] == 1   # MatchState.begin_deal numbers deals from 1
        assert all(isinstance(d["winners"], list) for d in g["deals"])


@pytest.mark.skipif(not (os.path.exists(CK) and torch.cuda.is_available()), reason="needs a cnn_m_v3r checkpoint + CUDA")
def test_wpure_credit_telescopes_with_analytic_w():
    eps, games = _games("none")
    import copy
    games2 = copy.deepcopy(games)
    stats = rewrite_wpure(games2, PureW(None), gamma=0.995)
    assert stats["paid"] > 0
    for g in games2:
        uma = g["hanchan"]["uma_points"]; d0 = g["deals"][0]
        for e in g["episodes"]:
            seat = int(e["key"][1]); r = e["rewards"]; T = len(r)
            w0 = rank_uma(d0["start_points"], seat)
            assert abs(float(r.sum()) - (uma[seat] - w0) * REWARD_SCALE) < 1e-4, (seat, r.sum(), uma[seat], w0)
            paid_steps = {st for _, st in g["deal_ends"][seat]} | {T - 1}
            assert all(r[t] == 0 for t in range(T) if t not in paid_steps)
            ret = np.zeros(T, np.float32); acc = 0.0
            for t in range(T - 1, -1, -1):
                acc = r[t] + 0.995 * acc; ret[t] = acc
            assert np.allclose(ret, e["returns"], atol=1e-5)
