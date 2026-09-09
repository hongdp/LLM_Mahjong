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

P3 = "/home/hongdp/Workspace/LLM_Mahjong/experiments/exp68r3_P/latest.pt"


def _cfg(temperature, shaping=False):
    return dict(channels=64, blocks=3, arch="cnn_m_r", temperature=temperature, gamma=0.995, games_per_worker=8,
                rollout_temps=None, shaping=shaping, shaping_scale=1.0, seed=5, critic_feats="none",
                gpu_infer=False, gpu_infer_opponents=False, infer_max_batch=64, infer_wait_ms=0.0, infer_device="cpu",
                bf16_infer=False, no_episodes=False, league=[], league_frac=0.0, league_learner_seats=1,
                league_opp_temp=None, hanchan=False, hanchan_w_path=None, action_space=space_of_arch("cnn_m_r"),
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
        assert np.array_equal(p["planes"], e["planes"]), e["key"]
        assert np.array_equal(p["scalars"], e["scalars"]), e["key"]
        assert np.allclose(p["rewards"], e["rewards"], atol=1e-6), (e["key"], p["rewards"], e["rewards"])
        assert np.allclose(p["returns"], e["returns"], atol=1e-5), e["key"]
        assert np.allclose(p["old_logprobs"], e["old_logprobs"], atol=1e-5), e["key"]
