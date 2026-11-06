import torch

from src.agents.dnn.selfplay import DnnStep, apply_shaping, returns_to_go


def _steps(phis, rewards):
    z = torch.zeros(1)
    return [DnnStep(planes=z, scalars=z, mask=torch.zeros(1, dtype=torch.bool), action_idx=0,
                    logprob=0.0, reward=r, phi=p) for p, r in zip(phis, rewards)]


def test_shaping_telescopes_and_scales():
    phis = [-7.0, -5.0, -5.0, -2.0, 0.5]
    for scale in (1.0, 0.3):
        st = _steps(phis, [0.0, 0.0, 0.0, 0.0, 8.0])
        apply_shaping(st, 1.0, scale)
        # gamma=1: total shaping = scale * (0 - Phi_0); settlement untouched
        assert abs(sum(s.reward for s in st) - (8.0 + scale * 7.0)) < 1e-9
        # a shanten regression is punished at its own step
        st2 = _steps([-5.0, -7.0, 0.0], [0.0, 0.0, 0.0])
        apply_shaping(st2, 0.995, scale)
        assert st2[0].reward < 0 < st2[1].reward
    r = returns_to_go(_steps([0, 0], [1.0, 2.0]), 0.5)
    assert abs(r[0] - 2.0) < 1e-9 and abs(r[1] - 2.0) < 1e-9
