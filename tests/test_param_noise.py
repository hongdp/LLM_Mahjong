"""exp97 parameter-space noise: per-row factorised head noise, deal-coherent resampling in the Rust rollout."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.dnn.arch_zoo import ZOO
from src.agents.dnn.encoder import VARIANT_SHAPE

riichi_rs = pytest.importorskip("riichi_rs")
ACTION_DIM = 374


def _batch(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    npl, nsc = VARIANT_SHAPE["v3r"]
    P = (torch.rand(n, npl, 34, generator=g) < 0.1).float()
    S = torch.rand(n, nsc, generator=g)
    M = torch.zeros(n, ACTION_DIM, dtype=torch.bool)
    for i in range(n):
        M[i, torch.randperm(34, generator=g)[:8]] = True
    return P, S, M


def test_noisy_head_matches_clean_at_sigma_zero_and_per_row_independence():
    torch.manual_seed(0)
    net = ZOO["cnn_m_v3r"][0]().eval()
    P, S, M = _batch()
    dims = net.noise_dims()
    assert [d[0] for d in dims][0] == 64 * 34 + 64 and dims[-1][1] == ACTION_DIM
    eps = net.sample_noise(P.shape[0], dims, "cpu")
    with torch.no_grad():
        clean, noisy0 = net.forward_clean_and_noisy(P, S, M, eps, 0.0)
        assert torch.equal(clean, net(P, S, M))
        assert torch.equal(clean, noisy0)
        _, noisy = net.forward_clean_and_noisy(P, S, M, eps, 0.5)
        assert not torch.allclose(clean[M], noisy[M])
        assert torch.isinf(noisy[~M]).all()
        # per-row independence: replacing row 3's noise changes row 3 only
        eps2 = [(ei.clone(), eo.clone()) for ei, eo in eps]
        new = net.sample_noise(1, dims, "cpu", generator=torch.Generator().manual_seed(99))
        for (ei, eo), (ni, no) in zip(eps2, new):
            ei[3] = ni[0]; eo[3] = no[0]
        _, noisy2 = net.forward_clean_and_noisy(P, S, M, eps2, 0.5)
        keep = torch.arange(P.shape[0]) != 3
        assert torch.equal(noisy[keep][M[keep]], noisy2[keep][M[keep]])
        assert not torch.allclose(noisy[3][M[3]], noisy2[3][M[3]])


def test_kl_grows_with_sigma():
    torch.manual_seed(1)
    net = ZOO["cnn_m_v3r"][0]().eval()
    P, S, M = _batch(64)
    eps = net.sample_noise(64, net.noise_dims(), "cpu")
    kls = []
    with torch.no_grad():
        for sg in (0.05, 0.3, 1.0):
            c, nz = net.forward_clean_and_noisy(P, S, M, eps, sg)
            lc, ln = torch.log_softmax(c, 1), torch.log_softmax(nz, 1)
            kls.append(float((lc.exp() * (lc - ln).masked_fill(~M, 0.0)).sum(1).mean()))
    assert kls[0] < kls[1] < kls[2] and kls[0] > 0


def test_rollout_noise_is_held_per_deal_and_logprob_is_clean():
    """collect_rust with param_noise_sigma: every table slot resamples exactly once per deal
    (resampled == games played... one per slot fill), and recorded logprobs are the clean head's."""
    from src.agents.dnn.rust_rollout import collect_rust
    torch.manual_seed(2)
    net = ZOO["cnn_m_v3r"][0]().to("cuda" if torch.cuda.is_available() else "cpu").eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [7_100_000 + i for i in range(24)]
    cfg = dict(temperature=1.0, gamma=0.995, shaping=False, games_per_worker=8, param_noise_sigma=0.5,
               noise_diag=True, encoder_variant="v3r", param_noise_mode="head")
    eps, res = collect_rust(net, len(seeds), cfg, 1, seeds, device=dev)
    nz = collect_rust.last_noise
    assert nz["resampled"] == len(seeds)                      # one perturbation per deal, never per step
    assert nz["kl"] > 0 and 0 < nz["greedy_change"] < 1
    assert "diag_n" in nz
    # recorded logprob == clean log-softmax of the taken action
    e = max(eps, key=lambda x: len(x["actions"]))
    q = float(riichi_rs.PLANE_Q)
    Pl = torch.from_numpy(e["planes"]).to(dev).float() / q
    with torch.no_grad():
        lp = torch.log_softmax(net(Pl, torch.from_numpy(e["scalars"]).to(dev), torch.from_numpy(e["mask"]).to(dev)), 1)
    got = lp.gather(1, torch.from_numpy(np.asarray(e["actions"])).to(dev)[:, None]).squeeze(1).cpu().numpy()
    assert np.allclose(got, e["old_logprobs"], atol=1e-4)


def test_full_mode_copies_are_perturbed_and_logprob_is_clean():
    from src.agents.dnn.rust_rollout import collect_rust, perturbed_copies
    torch.manual_seed(3)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ZOO["cnn_m_v3r"][0]().to(dev).eval()
    cps = perturbed_copies(net, 3, 0.05)
    P, S, M = _batch(4)
    with torch.no_grad():
        outs = [c(P.to(dev), S.to(dev), M.to(dev)) for c in cps]
        assert not torch.allclose(outs[0][M.to(dev)], outs[1][M.to(dev)])
        for c in cps:                                   # value head untouched
            assert torch.equal(c.value[0].weight, net.value[0].weight)
    seeds = [7_200_000 + i for i in range(16)]
    cfg = dict(temperature=1.0, gamma=0.995, shaping=False, games_per_worker=8, param_noise_sigma=0.05,
               noise_diag=True, encoder_variant="v3r", param_noise_mode="full", param_noise_copies=4)
    eps, _ = collect_rust(net, len(seeds), cfg, 1, seeds, device=dev)
    nz = collect_rust.last_noise
    assert nz["kl"] > 0 and nz["resampled"] == 4          # (reservoir may be empty for a random net: no riichi)
    e = max(eps, key=lambda x: len(x["actions"]))
    Pl = torch.from_numpy(e["planes"]).to(dev).float() / float(riichi_rs.PLANE_Q)
    with torch.no_grad():
        lp = torch.log_softmax(net(Pl, torch.from_numpy(e["scalars"]).to(dev), torch.from_numpy(e["mask"]).to(dev)), 1)
    got = lp.gather(1, torch.from_numpy(np.asarray(e["actions"])).to(dev)[:, None]).squeeze(1).cpu().numpy()
    assert np.allclose(got, e["old_logprobs"], atol=1e-4)


def test_single_deviation_rollout_marks_and_limits_deviations():
    """exp98: at most one T=1 deviation per (deal, seat); every other step is the argmax; greedy steps are
    marked with +1000 on the recorded logprob and deviation steps carry log pi(a)."""
    from src.agents.dnn.rust_rollout import collect_rust
    torch.manual_seed(5)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ZOO["cnn_m_v3r"][0]().to(dev).eval()
    seeds = [7_300_000 + i for i in range(32)]
    cfg = dict(temperature=1.0, gamma=0.995, shaping=False, games_per_worker=8, encoder_variant="v3r", dev_p=0.2, seed=1)
    eps, _ = collect_rust(net, len(seeds), cfg, 1, seeds, device=dev)
    q = float(riichi_rs.PLANE_Q)
    n_dev_total = 0
    for e in eps:
        lp = np.asarray(e["old_logprobs"]); acts = np.asarray(e["actions"])
        is_dev = lp < 500
        assert is_dev.sum() <= 1
        n_dev_total += int(is_dev.sum())
        Pl = torch.from_numpy(e["planes"]).to(dev).float() / q
        with torch.no_grad():
            logits = net(Pl, torch.from_numpy(e["scalars"]).to(dev), torch.from_numpy(e["mask"]).to(dev))
        greedy = logits.argmax(1).cpu().numpy()
        assert (acts[~is_dev] == greedy[~is_dev]).all()
        lps = torch.log_softmax(logits, 1).gather(1, torch.from_numpy(acts).to(dev)[:, None]).squeeze(1).cpu().numpy()
        assert np.allclose(lps[is_dev], lp[is_dev], atol=1e-4) and np.allclose(lps[~is_dev] + 1000.0, lp[~is_dev], atol=1e-3)
    assert n_dev_total > 0 and collect_rust.last_dev["n_dev"] == n_dev_total
