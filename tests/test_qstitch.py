"""exp96 prior-anchored Double DQN: network wrapper, replay targets, fold mask, VecEnv.safe_info."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_dnn_qstitch import RED0, SKIP0, RingReplay, draw_roles, fold_mask
from src.agents.dnn.arch_zoo import ZOO
from src.agents.dnn.encoder import VARIANT_SHAPE

riichi_rs = pytest.importorskip("riichi_rs")
ACTION_DIM = 374


def _batch(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    npl, nsc = VARIANT_SHAPE["v3r"]
    P = (torch.rand(n, npl, 34, generator=g) < 0.1).float()
    S = torch.rand(n, nsc, generator=g)
    M = torch.zeros(n, ACTION_DIM, dtype=torch.bool)
    for i in range(n):
        M[i, torch.randperm(34, generator=g)[:6]] = True
    return P, S, M


def test_zero_advantage_is_the_prior():
    torch.manual_seed(1)
    net = ZOO["aq_cnn_m_v3r"][0]().eval()
    net.q.load_state_dict(net.prior.state_dict())
    torch.nn.init.zeros_(net.q.head[-1].weight); torch.nn.init.zeros_(net.q.head[-1].bias)
    P, S, M = _batch()
    with torch.no_grad():
        lg = net(P, S, M)
        p0 = torch.softmax(net.prior(P, S, M), 1)
    assert torch.equal(lg.argmax(1), p0.argmax(1))
    assert torch.allclose(lg[M], torch.log(p0 + net.eps)[M], atol=1e-6)
    assert torch.isinf(lg[~M]).all()
    assert not any(p.requires_grad for p in net.prior.parameters())


def test_large_advantage_overrules_prior_small_does_not():
    torch.manual_seed(2)
    net = ZOO["aq_cnn_m_v3r"][0]().eval()
    P, S, M = _batch(4)
    with torch.no_grad():
        p0 = torch.softmax(net.prior(P, S, M), 1)
        order = p0.masked_fill(~M, -1).argsort(1, descending=True)
        top, second = order[:, 0], order[:, 1]
        gap_nats = (torch.log(p0 + net.eps).gather(1, top[:, None]) - torch.log(p0 + net.eps).gather(1, second[:, None])).squeeze(1)

        def logits_with(adv):
            a = torch.zeros(4, ACTION_DIM)
            a[torch.arange(4), second] = adv
            a = a - (p0 * a).sum(1, keepdim=True)
            return torch.log(p0 + net.eps).masked_fill(~M, float("-inf")) + a.masked_fill(~M, 0) / net.tau
        small = logits_with((gap_nats * net.tau * 0.5))
        large = logits_with((gap_nats * net.tau * 1.5 + 1e-3))
    assert torch.equal(small.argmax(1), top)
    assert torch.equal(large.argmax(1), second)


def _episode(T, rew_last, tag):
    return {"planes": np.full((T, 2, 34), tag, dtype=np.uint8), "scalars": np.zeros((T, 3), dtype=np.float32),
            "mask": np.ones((T, 5), dtype=bool), "actions": np.arange(T) % 5,
            "rewards": np.array([0.0] * (T - 1) + [rew_last], dtype=np.float32)}


def test_replay_nstep_targets_and_bootstrap_slots():
    rp = RingReplay(64, 2, 3, 5)
    n = rp.add_block([_episode(5, 4.0, 1), _episode(2, -2.0, 2)], reward_scale=0.5, nstep=3)
    assert n == 7
    # episode 1 (slots 0..4): rows 0,1 bootstrap at +3 with zero reward; rows 2..4 reach the terminal reward 2.0
    assert rp.nidx[:5].tolist() == [3, 4, -1, -1, -1]
    assert rp.nret[:5].tolist() == [0.0, 0.0, 2.0, 2.0, 2.0]
    assert rp.mcret[:5].tolist() == [2.0] * 5
    # episode 2 (slots 5,6): never bootstraps into another episode
    assert rp.nidx[5:7].tolist() == [-1, -1]
    assert rp.nret[5:7].tolist() == [-1.0, -1.0]


def test_replay_wraparound_invalidates_overwritten_bootstrap():
    rp = RingReplay(10, 2, 3, 5)
    rp.add_block([_episode(8, 1.0, 1)], 1.0, 3)
    rp.add_block([_episode(4, 1.0, 2)], 1.0, 3)           # slots 8, 9, 0, 1: overwrites the first block's head
    assert rp.planes[0, 0, 0] == 2 and rp.planes[8, 0, 0] == 2
    assert rp.nidx[8] == 1                                  # wraps to slot 1 of the same block
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        seen |= set(rp.sample(10, rng).tolist())
    # slots 5..7 of block 1 are terminal-reaching (valid); slots 2..4 bootstrap to 5..7 (still block 1, valid)
    assert {2, 3, 4, 5, 6, 7, 8, 9, 0, 1} >= seen and {8, 9, 0, 1} <= seen
    rp.add_block([_episode(4, 1.0, 3)], 1.0, 3)           # slots 2..5: slot 4's old bootstrap (7) survives, but slot 4 is new now
    for i in rp.sample(10, rng):
        b = rp.nidx[i]
        assert b < 0 or rp.block[b] == rp.block[i]


def test_fold_mask_restricts_to_genbutsu_only_when_exposed_and_not_tenpai():
    mask = np.zeros((4, ACTION_DIM), dtype=bool)
    mask[:, [0, 5, 9]] = True; mask[0, RED0 + 4] = True
    genb = np.zeros((4, 34), dtype=bool); genb[:, 5] = True; genb[0, 4] = True
    info = np.array([[1, 2], [0, -9], [1, 0], [2, 1]], dtype=np.int32)
    out = fold_mask(mask, genb, info)
    assert sorted(np.nonzero(out[0])[0].tolist()) == [5, RED0 + 4]
    assert (out[1] == mask[1]).all()                      # nobody in riichi
    assert (out[2] == mask[2]).all()                      # tenpai: free to push
    assert np.nonzero(out[3])[0].tolist() == [5]
    call = np.zeros((1, ACTION_DIM), dtype=bool); call[0, [3 * 34 + 2, SKIP0]] = True
    assert np.nonzero(fold_mask(call, np.zeros((1, 34), bool), np.array([[1, 2]], dtype=np.int32))[0])[0].tolist() == [SKIP0]


def test_draw_roles_deterministic():
    p = np.array([0.5, 0.25, 0.25])
    assert draw_roles(123, p).tolist() == draw_roles(123, p).tolist()
    r = np.stack([draw_roles(s, p) for s in range(4000)])
    assert abs((r == 0).mean() - 0.5) < 0.02


def _tile_name(i):
    return f"{i % 9 + 1}{'mps'[i // 9]}" if i < 27 else f"{i - 26}z"


def _greedy_shanten_action(pl, sc, m, rng):
    """Shanten-greedy driver so that riichi actually happens: win > riichi > best-shanten discard > skip."""
    for t0 in (6 * 34, 5 * 34, 1 * 34, 9 * 34):
        hit = np.nonzero(m[t0:t0 + 34])[0]
        if len(hit):
            return int(t0 + hit[0])
    disc = np.nonzero(m[:34])[0]
    if len(disc) == 0:
        return int(SKIP0) if m[SKIP0] else int(rng.choice(np.nonzero(m)[0]))
    counts = (pl[0:4] > 0).sum(0)
    n_melds = int(round(float(sc[11]) * 4))
    best, best_sh = int(disc[0]), 99
    for t in disc:
        c = counts.copy(); c[t] -= 1
        hand = [_tile_name(i) for i in range(34) for _ in range(int(c[i]))]
        sh = riichi_rs.shanten(hand, n_melds)
        if sh < best_sh:
            best, best_sh = int(t), sh
    return best


def test_safe_info_bounds_against_observation_planes():
    """genbutsu must contain every riichi opponent's own river and lie inside the union of all rivers;
    the riichi-opponent count must match the observation's riichi flags."""
    env = riichi_rs.VecEnv(list(range(7000, 7064)), 32, 1.0, False, 1.0, True, "v3r")
    rng = np.random.default_rng(0)
    exposed = 0
    while not env.done():
        planes, scalars, mask, seats, gids = env.observe()
        n = planes.shape[0]
        if n == 0:
            env.step([], []); env.drain_finished(); continue
        genb, info = env.safe_info()
        assert genb.shape == (n, 34) and info.shape == (n, 2)
        pl = planes.reshape(n, -1, 34)
        for i in range(n):
            flags = scalars[i, 5:8] > 0.5
            assert info[i, 0] == flags.sum()
            if info[i, 0] == 0:
                assert not genb[i].any() and info[i, 1] == -9
                continue
            exposed += 1
            rivers = pl[i, 8:12] > 0
            own = np.logical_and.reduce([rivers[1 + k] for k in range(3) if flags[k]])
            assert (genb[i] >= own).all()
            assert (genb[i] <= rivers.any(0)).all()
            assert -1 <= info[i, 1] <= 6
        acts = [_greedy_shanten_action(pl[i], scalars[i], mask[i], rng) for i in range(n)]
        env.step(acts, [0.0] * n); env.drain_finished()
    assert exposed > 50
