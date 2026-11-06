"""exp116: the posture prior is exactly + beta[shanten bucket] on the genbutsu discard slots (discard, discard0) and
nothing else; a v3r checkpoint widened into cnn_l_v3rg with beta = 0 is the same function as the v3r net."""
import torch

from src.agents.dnn.arch_zoo import ZOO
from src.agents.dnn.encoder import ACTION_DIM, ACTION_TYPES, TILE_TYPES
from src.agents.dnn.net import load_compatible


def _inputs(net, B=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    P = torch.rand(B, net.in_planes, 34, generator=g); S = torch.rand(B, net.scalar_fc[0].in_features, generator=g)
    M = torch.ones(B, ACTION_DIM, dtype=torch.bool)
    return P, S, M


def test_prior_only_touches_genbutsu_discards():
    torch.manual_seed(0)
    net = ZOO["cnn_l_v3rg"][0]().eval()
    P, S, M = _inputs(net)
    P[:, -1, :] = (torch.rand(P.shape[0], 34) > 0.6).float()                   # genbutsu plane
    S[:, -3:] = torch.nn.functional.one_hot(torch.randint(0, 3, (P.shape[0],)), 3).float()
    with torch.no_grad():
        net.posture_beta.zero_(); a = net(P, S, M)
        net.posture_beta.copy_(torch.tensor([1.0, 8.0, 16.0])); b = net(P, S, M)
    d = b - a
    exp = (S[:, -3:] @ torch.tensor([1.0, 8.0, 16.0]))[:, None] * P[:, -1, :]
    D, D0 = ACTION_TYPES.index("discard") * TILE_TYPES, ACTION_TYPES.index("discard0") * TILE_TYPES
    assert torch.allclose(d[:, D:D + 34], exp, atol=1e-4) and torch.allclose(d[:, D0:D0 + 34], exp, atol=1e-4)
    rest = torch.ones(ACTION_DIM, dtype=torch.bool); rest[D:D + 34] = False; rest[D0:D0 + 34] = False
    assert d[:, rest].abs().max() < 1e-5
    lo, v = net.forward_with_value(P, S, M)
    assert torch.allclose(lo, b, atol=1e-5)


def test_v3r_checkpoint_widens_function_preserving():
    torch.manual_seed(1)
    src = ZOO["cnn_l_v3r"][0]().eval()
    dst = ZOO["cnn_l_v3rg"][0]().eval()
    load_compatible(dst, src.state_dict())
    assert dst.posture_beta.abs().sum() == 0
    P, S, M = _inputs(dst, seed=3)
    with torch.no_grad():
        ref = src(P[:, :-1], S[:, :-3], M)
        out = dst(P, S, M)                     # extra plane / scalars present but their weights are zero
    assert torch.allclose(ref, out, atol=1e-5)


def test_prior_gradient_reaches_beta():
    net = ZOO["cnn_l_v3rg"][0]().train()
    P, S, M = _inputs(net, seed=5)
    P[:, -1, :] = 1.0; S[:, -3:] = torch.tensor([0.0, 1.0, 0.0])
    lo = net(P, S, M)
    torch.log_softmax(lo, 1)[:, 0].sum().backward()          # push probability toward discard of tile 0
    assert net.posture_beta.grad is not None and net.posture_beta.grad[1].abs() > 0
