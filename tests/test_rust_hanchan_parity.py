"""Rust hanchan (VecEnv match chaining + MatchState) reproduces the Python driver (play_hanchan_gen /
--hanchan_pure, credit=None) deal for deal at T=0."""
import os
import random

import numpy as np
import pytest
import torch

riichi_rs = pytest.importorskip("riichi_rs")
from src.agents.dnn.action_space import space_of_arch
from src.agents.dnn.arch_zoo import ZOO
from src.agents.dnn.net import load_compatible
from src.agents.dnn.parallel_rollout import collect_parallel
from src.agents.dnn.rust_rollout import collect_rust
from src.tasks.mahjong.hanchan import HanchanTable

P3 = "/home/hongdp/Workspace/LLM_Mahjong/experiments/exp68r3_P/latest.pt"


def test_hanchan_table_context_deal_parity():
    """Table.hanchan(seed, dealer, rw, points, kyotaku, honba) == random.seed(seed); HanchanTable(...)."""
    rng = random.Random(11)
    for i in range(60):
        seed = 6_000_000 * 1_000_003 + i * 977   # > 2^32 like real match deal seeds
        dealer, rw = rng.randrange(4), rng.randrange(2)
        pts = [25000 + 1000 * rng.randrange(-8, 9) for _ in range(4)]
        pts[0] += 100000 - sum(pts)
        kyo = 1000 * rng.randrange(3)
        random.seed(seed)
        py = HanchanTable(dealer, rw, pts, kyo)
        rs = riichi_rs.Table.hanchan(seed, dealer, rw, pts, kyo, 0)
        assert rs.dealer == py.dealer and rs.round_wind_idx == py.round_wind_idx and rs.turn == py.turn, (seed, dealer, rw)
        assert rs.points == list(py.points) and rs.kyotaku == py.kyotaku
        for p in range(4):
            assert rs.hands[p] == list(py.hands[p]), (seed, dealer, p)
            assert rs.red[p] == [py.red[p][k] for k in "mps"], (seed, dealer, p)
        assert rs.last_drawn == [py.last_drawn[p] for p in range(4)]


def _cfg(arch="cnn_m_r"):
    return dict(channels=64, blocks=3, arch=arch, temperature=0.0, gamma=0.999, games_per_worker=4,
                rollout_temps=None, shaping=False, shaping_scale=1.0, seed=5, critic_feats="none",
                gpu_infer=True, gpu_infer_opponents=False, infer_max_batch=64, infer_wait_ms=0.0, infer_device="cpu",
                bf16_infer=False, no_episodes=False, league=[], league_frac=0.0, league_learner_seats=1,
                league_opp_temp=None, hanchan=True, hanchan_pure=True, hanchan_credit="none", hanchan_w_path=None,
                action_space=space_of_arch(arch), single_dev_p=0.0, single_dev_temp=1.0, all_seats_episodes=False, cf_p=0.0)


@pytest.mark.skipif(not os.path.exists(P3), reason="P3 checkpoint not on this machine")
def test_greedy_hanchan_episodes_identical():
    torch.manual_seed(0)
    net = ZOO["cnn_m_r"][0]()
    load_compatible(net, torch.load(P3, map_location="cpu", weights_only=False)["state_dict"])
    net.eval()
    seeds = [7_400_000 + i for i in range(8)]
    cfg = _cfg()
    ep_py, res_py = collect_parallel(net, len(seeds), cfg, 1, seeds)
    ep_rs, res_rs = collect_rust(net, len(seeds), cfg, 1, seeds, device="cpu")
    assert sorted(res_py) == sorted(res_rs)
    assert len(ep_rs) == len(ep_py) == 4 * len(seeds)
    by_key = {e["key"]: e for e in ep_py}
    for e in ep_rs:
        p = by_key[tuple(e["key"])]
        assert len(e["actions"]) > 60, e["key"]                      # a whole match, not one deal
        assert np.array_equal(p["actions"], e["actions"]), e["key"]
        assert np.array_equal(np.rint(p["planes"].astype(np.float32) * riichi_rs.PLANE_Q).astype(np.uint8), e["planes"]), e["key"]
        assert np.allclose(p["rewards"], e["rewards"], atol=1e-6), (e["key"], np.abs(p["rewards"] - e["rewards"]).max())
        assert np.allclose(p["returns"], e["returns"], atol=1e-4), e["key"]
    g_py = {int(g["episodes"][0]["key"][0]): g for g in collect_parallel.last_games}
    for g in collect_rust.last_games:
        q = g_py[int(g["seed"])]
        assert g["hanchan"]["placements"] == list(q["hanchan"]["placements"]), g["seed"]
        assert g["hanchan"]["uma_points"] == list(q["hanchan"]["uma_points"]), g["seed"]
        assert g["hanchan"]["n_deals"] == q["hanchan"]["n_deals"] and g["hanchan"]["busted"] == q["hanchan"]["busted"]
        assert list(g["points"]) == list(q["points"])
        # telescoping invariant per seat: sum(rewards) == final - 25000 + UMA
        for e in [x for x in ep_rs if int(x["key"][0]) == int(g["seed"])]:
            assert abs(float(e["rewards"].sum()) - g["hanchan"]["uma_points"][int(e["key"][1])] * 0.001) < 1e-4, e["key"]
    assert collect_rust.last_hanchan["pure_abs_uma"]["n"] == len(seeds)
