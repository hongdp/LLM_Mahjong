"""exp113: net2net widen + deepen must preserve the function exactly (eval and train-mode BN), and every new
parameter group must receive gradient (no dead capacity)."""
import torch

from scripts.net2net_widen import widen_state
from src.agents.dnn.arch_zoo import CnnPolicy, N_PLANES_V3R, N_SCALARS_V3


def _nets():
    torch.manual_seed(1)
    small = CnnPolicy(32, 2, in_planes=N_PLANES_V3R, in_scalars=N_SCALARS_V3, encoder_variant="v3r").double()
    with torch.no_grad():                    # non-trivial BN stats / affine so the copy is really tested
        for m in small.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.running_mean.uniform_(-0.5, 0.5); m.running_var.uniform_(0.5, 2.0)
                m.weight.uniform_(0.5, 1.5); m.bias.uniform_(-0.2, 0.2)
    big = CnnPolicy(96, 6, in_planes=N_PLANES_V3R, in_scalars=N_SCALARS_V3, encoder_variant="v3r").double()
    big.load_state_dict(widen_state(small.state_dict(), big, 32, 2))
    P = torch.randn(48, N_PLANES_V3R, 34, dtype=torch.float64); S = torch.randn(48, N_SCALARS_V3, dtype=torch.float64)
    M = torch.ones(48, small.head[-1].out_features, dtype=torch.bool)
    return small, big, P, S, M


def test_function_preserved_eval_and_train():
    small, big, P, S, M = _nets()
    for mode in ("eval", "train"):
        getattr(small, mode)(); getattr(big, mode)()
        with torch.no_grad():
            ls, vs = small.forward_with_value(P, S, M); lb, vb = big.forward_with_value(P, S, M)
        assert torch.allclose(ls, lb, atol=1e-9), mode
        assert torch.allclose(vs, vb, atol=1e-9), mode


def test_new_capacity_gets_gradient():
    _, big, P, S, M = _nets()
    big.train()
    lo, v = big.forward_with_value(P, S, M)
    (lo.logsumexp(1).mean() + v.pow(2).mean()).backward()
    # an inserted identity block (index 0 is new: old blocks sit at 2 and 5) must get gradient on its zeroed gamma
    assert big.blocks[0].n2.weight.grad.abs().sum() > 0
    # head columns reading new channels start at zero but receive gradient
    assert big.head[0].weight.grad[:, 32 * 34:96 * 34].abs().sum() > 0
    # new stem rows (incoming) receive gradient only through later layers -> after one step they will; check the
    # path exists: new-channel activations are non-zero
    with torch.no_grad():
        h = torch.relu(big.stem(P))
    assert h[:, 32:].abs().sum() > 0
