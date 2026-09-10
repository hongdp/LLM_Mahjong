"""P4b: collect_rust (Rust VecEnv + greedy net) reproduces collect_parallel's episodes exactly at T=0."""
import os

import numpy as np
import pytest
import torch

riichi_rs = pytest.importorskip("riichi_rs")
from src.agents.dnn.action_space import space_of_arch
from src.agents.dnn.arch_zoo import ZOO
from src.agents.dnn.net import load_compatible
from src.agents.dnn.parallel_rollout import collect_parallel
from src.agents.dnn.rust_rollout import collect_rust


def _same_planes(py_planes, rs_planes):
    """Python packer ships float16 of the encoder's f32 values; Rust ships uint8 on the
    PLANE_Q grid. Both are exact images of the same k/20 values."""
    assert rs_planes.dtype == np.uint8
    return np.array_equal(np.rint(py_planes.astype(np.float32) * riichi_rs.PLANE_Q).astype(np.uint8), rs_planes)

P3 = "/home/hongdp/Workspace/LLM_Mahjong/experiments/exp68r3_P/latest.pt"


def _cfg(temperature, shaping=False, arch="cnn_m_r"):
    return dict(channels=64, blocks=3, arch=arch, temperature=temperature, gamma=0.995, games_per_worker=8,
                rollout_temps=None, shaping=shaping, shaping_scale=1.0, seed=5, critic_feats="none",
                gpu_infer=False, gpu_infer_opponents=False, infer_max_batch=64, infer_wait_ms=0.0, infer_device="cpu",
                bf16_infer=False, no_episodes=False, league=[], league_frac=0.0, league_learner_seats=1,
                league_opp_temp=None, hanchan=False, hanchan_w_path=None, action_space=space_of_arch(arch),
                single_dev_p=0.0, single_dev_temp=1.0, all_seats_episodes=False, cf_p=0.0)


@pytest.mark.skipif(not os.path.exists(P3), reason="P3 checkpoint not on this machine")
@pytest.mark.parametrize("shaping", [False, True])
def test_greedy_rollout_episodes_identical(shaping):
    torch.manual_seed(0)
    net = ZOO["cnn_m_r"][0]()
    load_compatible(net, torch.load(P3, map_location="cpu", weights_only=False)["state_dict"])
    net.eval()
    seeds = [7_100_000 + i for i in range(24)]
    cfg = _cfg(0.0, shaping)
    ep_py, res_py = collect_parallel(net, len(seeds), cfg, 2, seeds)
    ep_rs, res_rs = collect_rust(net, len(seeds), cfg, 1, seeds, device="cpu")
    assert sorted(res_py) == sorted(res_rs)
    by_key = {e["key"]: e for e in ep_py}
    assert len(ep_rs) == len(ep_py)
    for e in ep_rs:
        p = by_key[tuple(e["key"])]
        assert np.array_equal(p["actions"], e["actions"]), e["key"]
        assert np.array_equal(p["mask"], e["mask"]), e["key"]
        assert _same_planes(p["planes"], e["planes"]), e["key"]
        assert np.array_equal(p["scalars"], e["scalars"]), e["key"]
        assert np.allclose(p["rewards"], e["rewards"], atol=1e-6), (e["key"], p["rewards"], e["rewards"])
        assert np.allclose(p["returns"], e["returns"], atol=1e-5), e["key"]
        assert np.allclose(p["old_logprobs"], e["old_logprobs"], atol=1e-5), e["key"]


def test_greedy_rollout_v3r_identical():
    """v3r encoder path: a random-init cnn_m_v3r net played greedily on both engines."""
    torch.manual_seed(1)
    net = ZOO["cnn_m_v3r"][0]()
    net.eval()
    seeds = [7_200_000 + i for i in range(16)]
    cfg = _cfg(0.0, False, arch="cnn_m_v3r")
    ep_py, res_py = collect_parallel(net, len(seeds), cfg, 2, seeds)
    ep_rs, res_rs = collect_rust(net, len(seeds), cfg, 1, seeds, device="cpu")
    assert sorted(res_py) == sorted(res_rs)
    by_key = {e["key"]: e for e in ep_py}
    assert len(ep_rs) == len(ep_py)
    for e in ep_rs:
        p = by_key[tuple(e["key"])]
        assert p["planes"].shape == e["planes"].shape == (len(e["actions"]), 56, 34), e["key"]
        assert np.array_equal(p["actions"], e["actions"]) and _same_planes(p["planes"], e["planes"]), e["key"]
        assert np.array_equal(p["scalars"], e["scalars"]) and np.allclose(p["returns"], e["returns"], atol=1e-5), e["key"]


@pytest.mark.skipif(not os.path.exists(P3), reason="P3 checkpoint not on this machine")
def test_greedy_league_rollout_identical():
    """League (frozen pool in non-learner seats, exp82): learner-seat episodes, learner_seats and
    the {seat: pool_idx} map match collect_parallel deal for deal at T=0."""
    torch.manual_seed(2)
    net = ZOO["cnn_m_r"][0]()
    load_compatible(net, torch.load(P3, map_location="cpu", weights_only=False)["state_dict"])
    net.eval()
    seeds = [7_300_000 + i for i in range(24)]
    cfg = _cfg(0.0, False)
    cfg.update(league=[{"name": "P3", "path": P3}], league_frac=0.6, league_learner_seats=0, league_opp_temp=0.0)
    ep_py, res_py = collect_parallel(net, len(seeds), cfg, 2, seeds)
    ep_rs, res_rs = collect_rust(net, len(seeds), cfg, 1, seeds, device="cpu")
    assert sorted(res_py) == sorted(res_rs)
    assert len(ep_rs) == len(ep_py) and len(ep_py) < 4 * len(seeds)      # some seats were opponents
    by_key = {e["key"]: e for e in ep_py}
    for e in ep_rs:
        p = by_key[tuple(e["key"])]
        assert np.array_equal(p["actions"], e["actions"]) and _same_planes(p["planes"], e["planes"]), e["key"]
        assert np.allclose(p["returns"], e["returns"], atol=1e-5), e["key"]
    g_py = {int(g["episodes"][0]["key"][0]): g for g in collect_parallel.last_games if g["episodes"]}
    for g in collect_rust.last_games:
        q = g_py[int(g["seed"])]
        assert sorted(g["learner_seats"]) == sorted(q["learner_seats"]), g["seed"]
        assert {int(k): int(v) for k, v in (g.get("league") or {}).items()} == {int(k): int(v) for k, v in (q.get("league") or {}).items()}, g["seed"]
    assert collect_rust.last_league.get("P3", {}).get("n", 0) > 0
