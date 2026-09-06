import random

import torch

from src.agents.dnn import encoder as enc
from src.agents.dnn.action_space import space_of_arch
from src.agents.dnn.arch_zoo import ZOO
from src.agents.dnn.net import load_compatible
from src.agents.dnn.oracle_features import ORACLE_DIM, oracle_features
from src.agents.dnn.selfplay import CFEAT_DIM, critic_features
from src.tasks.mahjong.table import PyMahjongTable


def _table(seed=3):
    random.seed(seed)
    t = PyMahjongTable(randomize_round=True)
    t.text_obs = False
    return t


def test_oracle_features_layout_and_hidden_content():
    t = _table()
    pid = t.turn
    f = oracle_features(t, pid)
    assert f.shape == (ORACLE_DIM,) and CFEAT_DIM["oracle"] == ORACLE_DIM
    # opponents' concealed hands are fully exposed (counts / 4), own hand is not
    for off in range(1, 4):
        opp = (pid + off) % 4
        seg = f[(off - 1) * 34:(off - 1) * 34 + 34]
        assert abs(float(seg.sum()) * 4 - len(t.hands[opp])) < 1e-6
        assert sum(t.red[opp].values()) == float(f[102 + (off - 1) * 3:102 + (off - 1) * 3 + 3].sum())
    assert abs(float(f[216:250].sum()) * 4 - len(t.wall)) < 1e-6
    assert abs(float(f[250]) - len(t.wall) / 70.0) < 1e-6
    assert float(f[251:285].sum()) == 1.0 and float(f[285:319].sum()) == 1.0
    assert torch.equal(critic_features(t, pid, "oracle"), f)
    assert critic_features(t, pid, "none") is None


def test_oracle_critic_net_loads_policy_from_bc_ckpt_shape():
    net = ZOO["convformer_m_v3r_m46_oc"][0]()
    base = ZOO["convformer_m_v3r_m46"][0]()
    assert net.critic_feat_dim == ORACLE_DIM and base.critic_feat_dim == 0
    assert enc.variant_of_arch("convformer_m_v3r_m46_oc") == "v3r"
    assert space_of_arch("convformer_m_v3r_m46_oc") == "mortal46"
    # policy keys transfer; only value_head shapes differ and are skipped
    skipped = load_compatible(net, base.state_dict())
    assert skipped and all(k.startswith("value") for k in skipped)     # value_head + value_cfeat_proj only
    B = 2
    planes = torch.rand(B, enc.N_PLANES_V3R, 34)
    scalars = torch.randn(B, enc.N_SCALARS_V3)
    mask = torch.ones(B, 46, dtype=torch.bool)
    cf = torch.rand(B, ORACLE_DIM)
    with torch.no_grad():
        l1, v1 = net.forward_with_value(planes, scalars, mask, cf)
        l2, v2 = net.forward_with_value(planes, scalars, mask, None)
        lb, _ = base.forward_with_value(planes, scalars, mask)
    assert torch.allclose(l1, lb, atol=1e-5) and torch.allclose(l2, lb, atol=1e-5)   # policy identical
    assert v1.shape == (B,) and not torch.allclose(v1, v2)                             # value uses cfeats
