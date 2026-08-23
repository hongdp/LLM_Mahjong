"""exp28: the recorded logprob is the BEHAVIOUR policy's (softmax(logits/T)),
so PPO's ratio pi/b is a proper importance weight at T != 1; T = 1 unchanged."""
import torch

from src.agents.dnn.arch_zoo import ZOO
from src.agents.dnn.encoder import ACTION_DIM, N_PLANES_V1R, TILE_TYPES


def _inputs(n=64):
    torch.manual_seed(0)
    p = torch.rand(n, N_PLANES_V1R, TILE_TYPES).round()
    s = torch.rand(n, 20)
    m = torch.zeros(n, ACTION_DIM, dtype=torch.bool)
    m[:, :6] = True
    return p, s, m


def test_logprob_matches_tempered_distribution():
    for name in ("cnn_m_r", "convformer_m_r"):
        net = ZOO[name][0]().eval()
        p, s, m = _inputs()
        for T in (0.5, 1.0, 1.5):
            torch.manual_seed(1)
            idx, lp = net.act(p, s, m, temperature=T)
            logits = net.forward(p, s, m)
            ref = torch.log_softmax(logits / T, 1).gather(1, idx[:, None]).squeeze(1)
            assert torch.allclose(lp, ref, atol=1e-5), (name, T)
        idx0, lp0 = net.act(p, s, m, temperature=0)
        assert bool((idx0 == net.forward(p, s, m).argmax(1)).all())


def test_mixed_temperature_rollout_runs():
    from src.agents.dnn.selfplay import play_game
    net = ZOO["cnn_m_r"][0]().eval()
    g = play_game(net, temperature={0: 0.7, 1: 1.0, 2: 1.3, 3: 1.0}, deal_seed=5)
    assert g.result and sum(len(t) for t in g.trajectories.values()) > 40
