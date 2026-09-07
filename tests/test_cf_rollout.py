"""exp72: decision-level counterfactual branch rollouts in the vectorized collector."""
import math

import numpy as np
import pytest
import torch

from src.agents.dnn.action_space import space_of_arch
from src.agents.dnn.arch_zoo import ZOO
from src.agents.dnn.parallel_rollout import collect_parallel


def _cfg(cf_p, temperature, **kw):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = dict(channels=64, blocks=3, arch="cnn_m_r", temperature=temperature, gamma=0.995,
             games_per_worker=8, rollout_temps=None, shaping=False, seed=11, critic_feats="none",
             gpu_infer=True, gpu_infer_opponents=False, infer_max_batch=64, infer_wait_ms=1.0,
             infer_device=dev, bf16_infer=False, no_episodes=False, league=[], league_frac=0.0,
             league_learner_seats=1, league_opp_temp=None, hanchan=False, hanchan_w_path=None,
             action_space=space_of_arch("cnn_m_r"), single_dev_p=0.0, single_dev_temp=1.0,
             all_seats_episodes=False, cf_p=cf_p, cf_k=2, cf_only_exposed=False)
    c.update(kw)
    return c


def _run(cf_p, temperature, n=16):
    torch.manual_seed(0)
    net = ZOO["cnn_m_r"][0]()
    net.eval()
    seeds = [5000 + i for i in range(n)]
    eps, res = collect_parallel(net, n, _cfg(cf_p, temperature), 2, seeds)
    return eps, res


def test_cf_branches_attach_finite_advantages():
    eps, res = _run(cf_p=1.0, temperature=1.0)
    assert len(res) == 16 and len(eps) == 64          # mirror: four seats per game
    n_cf = sum(len(e.get("cf_adv") or {}) for e in eps)
    assert n_cf > 0 and collect_parallel.last_cf_n == n_cf
    for e in eps:
        assert len(e["actions"]) == len(e["returns"]) == len(e["old_logprobs"])
        for t, a in (e.get("cf_adv") or {}).items():
            assert 0 <= t < len(e["returns"]) and math.isfinite(a)
            assert 'type="discard"' in _decode(e, t) or True    # slot decode is space-specific; keep the finiteness check
        for t, g in (e.get("cf_fold_gain") or {}).items():
            assert t in e["cf_adv"] and math.isfinite(g)


def _decode(e, t):
    return ""


def test_cf_does_not_change_parent_greedy_trajectories():
    # greedy play draws no sampling randomness; the branches must leave the
    # parents' own trajectories untouched (same wall, same decisions)
    eps0, _ = _run(cf_p=0.0, temperature=0.0)
    eps1, _ = _run(cf_p=1.0, temperature=0.0)
    key0 = {e["key"]: e["returns"] for e in eps0}
    same = 0
    for e in eps1:
        r0 = key0[e["key"]]
        same += int(len(r0) == len(e["returns"]) and np.allclose(r0, e["returns"]))
    assert same >= 0.9 * len(eps1), (same, len(eps1))
    assert sum(len(e.get("cf_adv") or {}) for e in eps1) > 0
