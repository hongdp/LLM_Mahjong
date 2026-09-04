import numpy as np
import torch

from src.agents.dnn import encoder as enc
from src.agents.dnn import mortal_action as ma
from src.agents.dnn.symmetry import (SUIT_PERMS, SuitSymmetrized, apply_perm, green_count,
                                     rename_action, slot_perm, tile_perm)


def test_tile_and_slot_perms_are_bijections():
    for sp in SUIT_PERMS:
        tp = tile_perm(sp)
        assert sorted(tp.tolist()) == list(range(34))
        assert (tp[27:] == np.arange(27, 34)).all()          # honors fixed
        for space in ("mortal46", "native"):
            p = slot_perm(space, sp)
            assert sorted(p.tolist()) == list(range(len(p)))
    assert (tile_perm(SUIT_PERMS[0]) == np.arange(34)).all()


def test_apply_perm_roundtrip():
    x = torch.randn(2, 5, 34)
    for sp in SUIT_PERMS:
        p = tile_perm(sp)
        y = apply_perm(x, p)
        assert torch.equal(apply_perm(y, np.argsort(p)), x)
        # semantics: out[..., p[i]] == x[..., i]
        assert torch.equal(y[..., p[3]], x[..., 3])


def _actions():
    return ['<action type="discard" tile="5m" />', '<action type="discard0" tile="0p" />',
            '<action type="riichi" tile="3s" />', '<action type="chi" tile="7s" with="5s 6s" />',
            '<action type="chi" tile="4m" with="3m 5m" />', '<action type="pon" tile="1z" />',
            '<action type="kan" tile="9p" />', '<action type="skip" />', '<action type="ron" tile="2s" />']


def test_slot_perm_matches_renamed_actions_mortal46():
    acts = _actions()
    m0, _ = ma.legal_mask_46(acts)
    for sp in SUIT_PERMS:
        m1, _ = ma.legal_mask_46([rename_action(a, sp) for a in acts])
        got = apply_perm(torch.tensor(m0), slot_perm("mortal46", sp))
        assert torch.equal(got, torch.tensor(m1)), sp


def test_slot_perm_matches_renamed_actions_native():
    acts = _actions()
    m0, _ = enc.legal_mask(acts)
    for sp in SUIT_PERMS:
        m1, _ = enc.legal_mask([rename_action(a, sp) for a in acts])
        got = apply_perm(m0, slot_perm("native", sp))
        assert torch.equal(got, m1), sp


def test_symmetrized_net_is_invariant():
    from src.agents.dnn.arch_zoo import ZOO
    torch.manual_seed(0)
    net = ZOO["convformer_m_v3r_m46"][0]().eval()
    sym = SuitSymmetrized(net)
    B = 3
    planes = (torch.rand(B, enc.N_PLANES_V3R, 34) > 0.7).float()
    scalars = torch.randn(B, enc.N_SCALARS_V3)
    mask = torch.rand(B, ma.MORTAL_ACTION_DIM) > 0.5
    mask[:, 45] = True
    with torch.no_grad():
        base = sym(planes, scalars, mask)
        for sp in SUIT_PERMS[1:]:
            tp, spp = tile_perm(sp), slot_perm("mortal46", sp)
            out = sym(apply_perm(planes, tp), scalars, apply_perm(mask, spp))
            back = apply_perm(out, np.argsort(spp))
            fin = torch.isfinite(base)
            assert torch.equal(fin, torch.isfinite(back))
            assert torch.allclose(base[fin], back[fin], atol=1e-4), sp
    assert sym.encoder_variant == "v3r" and sym.action_space == "mortal46"


def test_batch_augmenter_consistency():
    from src.agents.dnn.symmetry import make_batch_augmenter
    torch.manual_seed(1)
    aug = make_batch_augmenter("v3r", "mortal46", "cpu", green_max=7)
    B = 60
    planes = (torch.rand(B, enc.N_PLANES_V3R, 34) > 0.8).float()
    planes[:, 0:4] = 0.0
    # give every sample a 13-tile hand: 4 copies of tiles 0..2 and one of tile 30
    planes[:, 0:4, 0:3] = 1.0
    planes[:, 0, 30] = 1.0
    # sample 59: all-green hand (9 green tiles) -> must be left untouched
    planes[59, 0:4] = 0.0
    planes[59, 0:2, [19, 20, 21, 23]] = 1.0
    planes[59, 0, 32] = 1.0
    mask = torch.rand(B, ma.MORTAL_ACTION_DIM) > 0.6
    label = torch.tensor([int(torch.nonzero(mask[i] | (torch.arange(46) == 45))[0]) for i in range(B)])
    mask[torch.arange(B), label] = True
    p2, m2, y2 = aug(planes, mask, label)
    assert torch.equal(p2[0:10], planes[0:10])                     # chunk 0 = identity
    assert torch.equal(p2[59], planes[59]) and int(y2[59]) == int(label[59])   # green guard
    for i in range(B):
        assert bool(m2[i, y2[i]])                                  # permuted label stays legal
        assert int(m2[i].sum()) == int(mask[i].sum())
        assert torch.equal(p2[i].sum(-1), planes[i].sum(-1))       # per-plane counts preserved
    # chunk 3 uses SUIT_PERMS[3]; verify the exact permutation on one row
    from src.agents.dnn.symmetry import SUIT_PERMS
    k = 3
    lo = int(torch.linspace(0, B, 7).long()[k])
    assert torch.equal(p2[lo], apply_perm(planes[lo], tile_perm(SUIT_PERMS[k])))
    assert int(y2[lo]) == int(slot_perm("mortal46", SUIT_PERMS[k])[label[lo]])


def test_green_count():
    assert green_count(["2s", "3s", "4s", "6s", "8s", "6z", "1m", "5s"], [{"tiles": ["6z", "6z", "6z"]}]) == 9
    assert green_count(["0s", "7s", "5z"], []) == 0
