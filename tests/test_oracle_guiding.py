import random

import torch

from src.agents.dnn import encoder as enc
from src.agents.dnn.arch_zoo import ZOO
from src.agents.dnn.encoder import encode_state, variant_of_arch, variant_shape
from src.tasks.mahjong.table import PyMahjongTable


def _table(seed=5):
    random.seed(seed)
    t = PyMahjongTable(randomize_round=True)
    t.text_obs = False
    return t


def test_v1ro_blind_by_default_and_visible_when_flagged():
    t = _table()
    pid = t.turn
    P0, s0 = encode_state(t, pid, variant="v1ro")
    P1, s1 = encode_state(t, pid, variant="v1r")
    assert P0.shape == (enc.N_PLANES_V1RO, 34) and variant_shape("v1ro") == (29, enc.N_SCALARS)
    assert torch.equal(P0[:enc.N_PLANES_V1R], P1)             # public part identical to v1r
    assert float(P0[enc.N_PLANES_V1R:].abs().sum()) == 0.0    # oracle planes zero unless flagged
    assert torch.equal(s0, s1)
    t.oracle_visible = True
    P2, _ = encode_state(t, pid, variant="v1ro")
    O = P2[enc.N_PLANES_V1R:]
    for off in range(1, 4):                                   # opponents' concealed hands exposed
        opp = (pid + off) % 4
        assert abs(float(O[off - 1].sum()) * 4 - len(t.hands[opp])) < 1e-6
    assert abs(float(O[6].sum()) * 4 - len(t.wall)) < 1e-6    # live wall composition
    assert float(O[7].sum()) == 1.0                           # ura indicator one-hot
    assert float(O[:3].sum()) > 0 and torch.equal(P2[:enc.N_PLANES_V1R], P1)


def test_cnn_m_ro_arch_and_variant_resolution():
    assert variant_of_arch("cnn_m_ro") == "v1ro" and variant_of_arch("cnn_m_r") == "v1r"
    net = ZOO["cnn_m_ro"][0]()
    assert net.encoder_variant == "v1ro"
    t = _table(7)
    pid = t.turn
    P, s = encode_state(t, pid, variant="v1ro")
    mask = torch.ones(1, enc.ACTION_DIM, dtype=torch.bool)
    with torch.no_grad():
        logits = net(P[None], s[None], mask)
    assert logits.shape == (1, enc.ACTION_DIM) and torch.isfinite(logits).all()
