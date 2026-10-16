"""exp113: the channels-last GEMM trunk (net.fast_trunk) must be a pure re-implementation of the cudnn trunk:
float64-identical outputs in eval and train mode, identical BatchNorm running-stat updates, identical gradients."""
import copy

import pytest
import torch

from src.agents.dnn.arch_zoo import CnnPolicy, N_PLANES_V3R, N_SCALARS_V3


def _pair(ch=64, bl=3):
    torch.manual_seed(0)
    a = CnnPolicy(ch, bl, in_planes=N_PLANES_V3R, in_scalars=N_SCALARS_V3, encoder_variant="v3r").double()
    b = copy.deepcopy(a); b.fast_trunk = True
    P = torch.randn(64, N_PLANES_V3R, 34, dtype=torch.float64); S = torch.randn(64, N_SCALARS_V3, dtype=torch.float64)
    return a, b, P, S


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_outputs_identical(mode):
    a, b, P, S = _pair()
    getattr(a, mode)(); getattr(b, mode)()
    with torch.no_grad():
        ya, yb = a.trunk(P, S), b.trunk(P, S)
    assert ya.shape == yb.shape
    assert torch.allclose(ya, yb, atol=1e-10, rtol=0)


def test_running_stats_and_grads_identical():
    a, b, P, S = _pair()
    a.train(); b.train()
    mask = torch.ones(64, a.head[-1].out_features, dtype=torch.bool)
    for net in (a, b):
        logits, v = net.forward_with_value(P, S, mask)
        (logits.logsumexp(1).mean() + v.pow(2).mean()).backward()
    sa, sb = a.state_dict(), b.state_dict()
    for k in sa:
        assert torch.allclose(sa[k].double(), sb[k].double(), atol=1e-10, rtol=0), k
    for (na, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-9, rtol=0), na
