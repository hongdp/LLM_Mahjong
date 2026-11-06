import random

import torch

from src.agents.dnn import encoder as enc
from src.agents.dnn.arch_zoo import ZOO
from src.agents.dnn.encoder import encode_state, variant_of_arch, variant_shape
from src.tasks.mahjong.hanchan import UMA, RankUmaCredit, _rank_uma_single
from src.tasks.mahjong.table import PyMahjongTable


def test_rank_uma_credit_is_analytic_and_telescopes():
    c = RankUmaCredit()
    pts = [25000, 25000, 25000, 25000]
    assert all(c.w(p, pts, 0, 0, 0, 0, 8) == UMA[0] for p in range(4))      # ties favour self -> rank 1 uma
    # a sequence of deal outcomes: credits W(after)-W(before) plus the final
    # true_uma - W(before) must sum exactly to points-25000+uma of the final rank
    seq = [[33000, 21000, 25000, 21000], [33000, 13000, 33000, 21000], [41000, 13000, 25000, 21000]]
    for me in range(4):
        total = 0.0
        before = c.w(me, pts, 0, 0, 0, 0, 8)
        for k, after_pts in enumerate(seq):
            if k < len(seq) - 1:
                after = c.w(me, after_pts, 0, 0, 0, 0, 8 - k - 1)
                total += after - before
                before = after
        final = seq[-1]
        rank = sorted(range(4), key=lambda s: (-final[s], s)).index(me)
        true_uma = final[me] - 25000 + UMA[rank]
        total += true_uma - before
        # credits telescope to true_uma minus the start potential (a constant
        # shared by all seats: equal scores -> everyone "ranks first" = UMA[0])
        assert abs(total - (true_uma - c.w(me, pts, 0, 0, 0, 0, 8))) < 1e-6
    assert _rank_uma_single([41000, 13000, 25000, 21000]) == UMA[0] + 16000


def test_v1rh_variant_and_arch():
    assert variant_of_arch("cnn_m_rh") == "v1rh" and variant_of_arch("cnn_m_r") == "v1r"
    assert variant_shape("v1rh") == (enc.N_PLANES_V1R, enc.N_SCALARS + 3)
    random.seed(3)
    t = PyMahjongTable(randomize_round=True)
    t.text_obs = False
    t.honba = 2
    P, s = encode_state(t, t.turn, variant="v1rh")
    P1, s1 = encode_state(t, t.turn, variant="v1r")
    assert torch.equal(P, P1) and s.shape == (enc.N_SCALARS + 3,) and torch.equal(s[:enc.N_SCALARS], s1)
    assert abs(float(s[enc.N_SCALARS]) - t.round_number / 4.0) < 1e-6 and abs(float(s[enc.N_SCALARS + 1]) - 0.25) < 1e-6
    net = ZOO["cnn_m_rh"][0]()
    mask = torch.ones(1, enc.ACTION_DIM, dtype=torch.bool)
    with torch.no_grad():
        lg = net(P[None], s[None], mask)
    assert lg.shape == (1, enc.ACTION_DIM) and torch.isfinite(lg).all()
