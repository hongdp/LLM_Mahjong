import random

import torch

from src.agents.dnn import encoder as enc
from src.agents.dnn.arch_zoo import ZOO
from src.agents.dnn.encoder import encode_state, variant_of_arch
from src.agents.dnn.oracle_features import oracle_features
from src.tasks.mahjong.table import PyMahjongTable


def test_aux_waits_head_and_targets():
    assert variant_of_arch("cnn_m_r_aux") == "v1r"
    net = ZOO["cnn_m_r_aux"][0]()
    assert net.aux_waits and net.encoder_variant == "v1r" and net.critic_feat_dim == 0
    base = ZOO["cnn_m_r"][0]()
    assert base.aux_waits_head is None
    random.seed(11)
    t = PyMahjongTable(randomize_round=True)
    t.text_obs = False
    pid = t.turn
    P, s = encode_state(t, pid, variant="v1r")
    o = oracle_features(t, pid)
    tgt = o[111:213].view(3, 34)
    for off in range(1, 4):                      # target = opponents' waits, 13-tile seats only
        opp = (pid + off) % 4
        if len(t.hands[opp]) % 3 == 1:
            assert float(tgt[off - 1].sum()) == float(len(set(t._waits(opp))))
    mask = torch.ones(1, enc.ACTION_DIM, dtype=torch.bool)
    with torch.no_grad():
        logits = net(P[None], s[None], mask)
        aux = net.aux_waits_logits(P[None], s[None])
        lg2, v = net.forward_with_value(P[None], s[None], mask)
    assert aux.shape == (1, 102) and logits.shape == (1, enc.ACTION_DIM) and v.shape == (1,)
    assert torch.equal(logits, lg2)               # aux head does not touch the policy path
