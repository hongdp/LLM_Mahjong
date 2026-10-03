"""exp103 fold option (commitment macro) tests.

Load-bearing: (1) without table context the fold space is byte-identical to
native plus a never-legal slot, so nothing context-free changes; (2) a v3r
checkpoint loaded into a v3rf net reproduces its logits exactly on the old 374
slots; (3) in play, after an opponent riichi a policy that always takes
ENTER_FOLD discards only genbutsu while any is legal and never riichis/calls;
(4) the Rust rollout rewrites episodes so masks (375), scalars (+1) and the
inserted ENTER_FOLD rows are self-consistent with the recorded logprobs.
"""
import os
import random
import re
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks.mahjong.table import PyMahjongTable                    # noqa: E402
from src.agents.dnn import fold_option as fo                          # noqa: E402
from src.agents.dnn.encoder import legal_mask, ACTION_DIM, TYPE_TO_ID, TILE_TYPES, tile_to_34  # noqa: E402
from src.agents.dnn.action_space import REGISTRY, get_space, space_of_arch  # noqa: E402
from src.agents.dnn.arch_zoo import ZOO                               # noqa: E402
from src.agents.dnn.net import load_compatible, FOLD_SLOT_INIT_BIAS   # noqa: E402

TILE = re.compile(r'tile="([^"]+)"')


def _rand_table(seed, steps=40):
    random.seed(seed)
    t = PyMahjongTable(randomize_round=True)
    t.text_obs = False
    for _ in range(steps):
        if t.finished:
            break
        pid = t.turn
        acts = t.get_legal_actions(pid)
        if not acts:
            break
        _, _, done, info = t.step(pid, random.choice(acts))
        if done:
            break
    return t


def test_fold_space_without_context_equals_native():
    sp = REGISTRY["native_fold"]
    assert sp.dim == ACTION_DIM + 1 and space_of_arch("cnn_l_v3rf") == "native_fold"
    for seed in range(20):
        t = _rand_table(seed)
        if t.finished:
            continue
        acts = t.get_legal_actions(t.turn)
        if not acts:
            continue
        m0, lk0 = legal_mask(acts)
        m1, lk1 = sp.mask(acts)
        assert m1.shape[0] == ACTION_DIM + 1
        assert torch.equal(m1[:ACTION_DIM], m0) and not bool(m1[ACTION_DIM]) and lk0 == lk1
        assert sp.follow_up(int(m0.nonzero()[0]), acts) is None


def test_restrict_mask_rules():
    m = np.zeros(ACTION_DIM, dtype=np.bool_)
    d, r, p, ron = TYPE_TO_ID["discard"], TYPE_TO_ID["riichi"], TYPE_TO_ID["pon"], TYPE_TO_ID["ron"]
    for tile in (0, 5, 9, 30):
        m[d * TILE_TYPES + tile] = True
    m[r * TILE_TYPES + 5] = True
    m[p * TILE_TYPES + 9] = True
    genb = np.zeros(TILE_TYPES, dtype=np.bool_); genb[9] = True; genb[30] = True
    out = fo.restrict_mask(m, genb)
    kept = set(np.nonzero(out)[0].tolist())
    assert kept == {d * TILE_TYPES + 9, d * TILE_TYPES + 30}, "only genbutsu discards survive; riichi/pon dropped"
    # no safe tile among the legal discards -> all legal discards stay, riichi/pon still dropped
    genb2 = np.zeros(TILE_TYPES, dtype=np.bool_); genb2[20] = True
    out2 = fo.restrict_mask(m, genb2)
    assert set(np.nonzero(out2)[0].tolist()) == {d * TILE_TYPES + x for x in (0, 5, 9, 30)}
    # interrupt row offering only a call + skip: skip survives
    m3 = np.zeros(ACTION_DIM, dtype=np.bool_); m3[p * TILE_TYPES + 9] = True; m3[TYPE_TO_ID["skip"] * TILE_TYPES] = True
    assert set(np.nonzero(fo.restrict_mask(m3, genb))[0].tolist()) == {TYPE_TO_ID["skip"] * TILE_TYPES}
    # ron is never removed
    m4 = m3.copy(); m4[ron * TILE_TYPES + 3] = True
    assert bool(fo.restrict_mask(m4, genb)[ron * TILE_TYPES + 3])
    # enter legality: needs a turn row with >= 2 discards and an opponent riichi
    assert fo.enter_legal(m, 1, False) and not fo.enter_legal(m, 0, False) and not fo.enter_legal(m, 1, True)
    m5 = np.zeros(ACTION_DIM, dtype=np.bool_); m5[d * TILE_TYPES + 7] = True
    assert not fo.enter_legal(m5, 2, False), "a seat in riichi (one legal discard) cannot enter fold"


def test_v3r_checkpoint_loads_into_v3rf_with_identical_logits():
    torch.manual_seed(0)
    base = ZOO["cnn_m_v3r"][0]()
    wide = ZOO["cnn_m_v3rf"][0]()
    skipped = load_compatible(wide, base.state_dict())
    assert not [k for k in skipped if not k.startswith("value")]
    from src.agents.dnn.encoder import N_PLANES_V3R, N_SCALARS_V3
    planes = torch.randn(3, N_PLANES_V3R, 34)
    sc = torch.randn(3, N_SCALARS_V3)
    sc_f = torch.cat([sc, torch.zeros(3, 1)], 1)
    mask = torch.ones(3, ACTION_DIM, dtype=torch.bool)
    mask_f = torch.cat([mask, torch.ones(3, 1, dtype=torch.bool)], 1)
    a = base(planes, sc, mask)
    b = wide(planes, sc_f, mask_f)
    assert torch.allclose(a, b[:, :ACTION_DIM], atol=1e-5)
    assert torch.allclose(b[:, ACTION_DIM], torch.full((3,), FOLD_SLOT_INIT_BIAS), atol=1e-5)
    assert getattr(wide, "action_space") == "native_fold" and wide.encoder_variant == "v3rf"


class _AlwaysFold(torch.nn.Module):
    """Takes ENTER_FOLD whenever legal; otherwise the lowest legal slot (tsumogiri-ish)."""
    encoder_variant = "v3rf"
    action_space = "native_fold"

    def act(self, planes, scalars, mask, temperature=1.0, check=True):
        m = mask[0]
        if bool(m[fo.FOLD_SLOT]):
            return torch.tensor(fo.FOLD_SLOT), torch.tensor(-0.5)
        legal = m.nonzero().flatten()
        # prefer a discard over a win so deals run long enough to see a riichi
        d = legal[legal < 2 * TILE_TYPES]
        pick = d[0] if len(d) else legal[0]
        return pick, torch.tensor(-0.5)


def _genbutsu_set(t, pid):
    g = fo.genbutsu_34(t, pid)
    return None if g is None else set(np.nonzero(g)[0].tolist())


class _PreferRiichi(torch.nn.Module):
    """Opponent stub: declares riichi whenever legal, else a random legal slot."""
    encoder_variant = "v3r"
    action_space = "native"

    def act(self, planes, scalars, mask, temperature=1.0, check=True):
        m = mask[0]
        r0, r1 = TYPE_TO_ID["riichi"] * TILE_TYPES, (TYPE_TO_ID["riichi"] + 1) * TILE_TYPES
        rr = m[r0:r1].nonzero().flatten()
        if len(rr):
            return rr[0] + r0, torch.tensor(-0.5)
        legal = m.nonzero().flatten()
        return legal[random.randrange(len(legal))], torch.tensor(-0.5)


def test_python_engine_fold_mode_discards_only_genbutsu():
    """Drive real deals through play_game_gen with the real `_choose`; whenever seat 0 is in
    fold mode and some legal discard is genbutsu, the chosen discard must be genbutsu, and
    riichi/chi/pon/kan never appear once folding."""
    from src.agents.dnn.selfplay import play_game_gen, _choose
    net, opp = _AlwaysFold(), _PreferRiichi()
    entered = checked = 0
    for seed in range(400):
        random.seed(seed)
        gen = play_game_gen(deal_seed=1000 + seed)
        try:
            table, reqs = next(gen)
            while True:
                replies = []
                for pid, actions in reqs:
                    if pid == 0:
                        steps, a = _choose(net, table, 0, actions, 0.0, "cpu")
                        flags = fo.fold_flags(table)
                        if len(steps) == 2:
                            entered += 1
                            assert steps[0].action_idx == fo.FOLD_SLOT and steps[0].mask.shape[0] == ACTION_DIM + 1
                            assert steps[1].mask.shape[0] == ACTION_DIM + 1 and not bool(steps[1].mask[fo.FOLD_SLOT])
                            assert flags[0]
                        if flags[0]:
                            kind = re.search(r'type="(\w+)"', a).group(1)
                            assert kind in ("discard", "discard0", "ron", "tsumo", "skip"), a
                            g = _genbutsu_set(table, 0)
                            if kind.startswith("discard") and g is not None:
                                legal_g = [x for x in actions if 'type="discard' in x
                                           and tile_to_34(TILE.search(x).group(1)) in g]
                                if legal_g:
                                    checked += 1
                                    assert tile_to_34(TILE.search(a).group(1)) in g, (a, g)
                    else:
                        steps, a = _choose(opp, table, pid, actions, 1.0, "cpu")
                    replies.append((steps[-1], a))
                table, reqs = gen.send(replies)
        except StopIteration:
            pass
    assert entered >= 3 and checked >= 3, (entered, checked)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VecEnv rollout path uses the GPU")
def test_rust_rollout_rewrites_episodes_consistently():
    import riichi_rs  # noqa: F401
    from src.agents.dnn.rust_rollout import collect_rust
    ck = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "exp88_Q1x", "games_final.pt")
    ck = ck if os.path.exists(ck) else "/home/hongdp/Workspace/LLM_Mahjong/experiments/exp88_Q1x/games_final.pt"
    if not os.path.exists(ck):
        pytest.skip("needs a trained cnn_m_v3r checkpoint (riichi must actually happen)")
    net = ZOO["cnn_m_v3rf"][0]()
    load_compatible(net, torch.load(ck, map_location="cpu")["state_dict"])
    net = net.cuda().eval()
    with torch.no_grad():
        net.head[-1].bias[fo.FOLD_SLOT].fill_(25.0)     # ENTER_FOLD wins whenever legal
    cfg = {"temperature": 1.0, "gamma": 0.995, "games_per_worker": 8, "encoder_variant": "v3rf"}
    episodes, results = collect_rust(net, 32, cfg, 1, seeds=[7_000_000 + i for i in range(32)], device="cuda")
    st = collect_rust.last_fold
    assert st["unconsumed_keys"] == 0 and st["n_enter"] > 0 and st["enter_rate"] > 0.95, st
    n_macro = 0
    for e in episodes:
        n = len(e["actions"])
        assert e["mask"].shape == (n, ACTION_DIM + 1) and e["scalars"].shape[1] == 30 and e["planes"].shape[0] == n
        assert len(e["old_logprobs"]) == n and len(e["returns"]) == n and len(e["rewards"]) == n
        for i in range(n):
            a = int(e["actions"][i])
            assert bool(e["mask"][i, a]), "recorded action must be legal under the recorded (possibly restricted) mask"
            if a == fo.FOLD_SLOT:
                n_macro += 1
                assert i + 1 < n and e["scalars"][i, -1] == 0.0 and e["scalars"][i + 1, -1] == 1.0
                assert e["returns"][i] == e["returns"][i + 1] and e["rewards"][i] == 0.0
                assert np.array_equal(e["planes"][i], e["planes"][i + 1])
                assert int(e["actions"][i + 1]) < 2 * TILE_TYPES, "the committed step is a discard"
                assert not bool(e["mask"][i + 1, fo.FOLD_SLOT])
        # once in fold mode the flag never clears within the deal
        f = e["scalars"][:, -1]
        if f.any():
            first = int(np.argmax(f > 0))
            assert f[first:].all()
    assert n_macro == st["n_enter"]
    # recorded logprobs must be reproducible from the recorded (mask, scalars) under the same net
    e = max(episodes, key=lambda x: len(x["actions"]))
    with torch.no_grad():
        P = torch.from_numpy(e["planes"]).cuda().float().div_(float(riichi_rs.PLANE_Q))
        S = torch.from_numpy(e["scalars"]).cuda()
        M = torch.from_numpy(e["mask"]).cuda()
        lp = torch.log_softmax(net(P, S, M), 1).gather(1, torch.from_numpy(e["actions"]).cuda().long()[:, None]).squeeze(1)
    assert torch.allclose(lp.cpu(), torch.from_numpy(e["old_logprobs"]).float(), atol=2e-3), (lp[:5], e["old_logprobs"][:5])


def test_free_space_only_adds_the_flag():
    """Arm M2 (native_fold_free / cnn_l_v3rfl): inside fold mode the mask stays the full legal
    mask (riichi/calls included), the follow-up step still excludes wins/skip, and the only
    imposed structure is the ENTER_FOLD slot + the in_fold_mode scalar."""
    sp = REGISTRY["native_fold_free"]
    assert space_of_arch("cnn_l_v3rfl") == "native_fold_free" and sp.dim == ACTION_DIM + 1
    from src.agents.dnn.encoder import variant_of_arch
    assert variant_of_arch("cnn_l_v3rfl") == "v3rf"
    m = np.zeros(ACTION_DIM, dtype=np.bool_)
    d, r, p = TYPE_TO_ID["discard"], TYPE_TO_ID["riichi"], TYPE_TO_ID["pon"]
    for tile in (0, 5, 9):
        m[d * TILE_TYPES + tile] = True
    m[r * TILE_TYPES + 5] = True; m[p * TILE_TYPES + 9] = True
    genb = np.zeros(TILE_TYPES, dtype=np.bool_); genb[9] = True
    assert np.array_equal(fo.restrict_mask(m, genb, restrict=False), m)
    fd = fo.fold_discard_mask(m, genb, restrict=False)
    assert set(np.nonzero(fd)[0].tolist()) == {d * TILE_TYPES + x for x in (0, 5, 9)}
    wide, enter = fo.batch_masks(m[None], genb[None], np.array([1]), np.array([True]), restrict=False)
    assert np.array_equal(wide[0, :ACTION_DIM], m) and not wide[0, fo.FOLD_SLOT] and not enter[0]


def test_batch_masks_matches_per_row():
    """The vectorised rollout path must reproduce restrict_mask/enter_legal row by row (random legal masks)."""
    rng = np.random.default_rng(3)
    n = 400
    m = np.zeros((n, ACTION_DIM), dtype=np.bool_)
    for i in range(n):
        k = rng.integers(1, 14)
        m[i, TYPE_TO_ID["discard"] * TILE_TYPES + rng.choice(34, k, replace=False)] = True
        if rng.random() < 0.3: m[i, TYPE_TO_ID["riichi"] * TILE_TYPES + rng.integers(34)] = True
        if rng.random() < 0.3: m[i, TYPE_TO_ID["pon"] * TILE_TYPES + rng.integers(34)] = True
        if rng.random() < 0.1: m[i] = False; m[i, TYPE_TO_ID["pon"] * TILE_TYPES + rng.integers(34)] = True; m[i, TYPE_TO_ID["skip"] * TILE_TYPES] = True
        if rng.random() < 0.05: m[i, TYPE_TO_ID["ron"] * TILE_TYPES + rng.integers(34)] = True
    genb = rng.random((n, 34)) < 0.25
    n_opp = rng.integers(0, 3, n)
    in_fold = rng.random(n) < 0.5
    for restrict in (True, False):
        out, enter = fo.batch_masks(m, genb, n_opp, in_fold, restrict)
        for i in range(n):
            if in_fold[i]:
                ref = fo.widen(fo.restrict_mask(m[i], genb[i] if n_opp[i] > 0 else None, restrict), False)
            else:
                ref = fo.widen(m[i], fo.enter_legal(m[i], int(n_opp[i]), False))
            assert np.array_equal(out[i], ref), (i, restrict, in_fold[i], n_opp[i])
            assert enter[i] == bool(ref[fo.FOLD_SLOT])
