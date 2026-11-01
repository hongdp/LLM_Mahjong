"""exp124: m46_native (46-slot masks derived from native 374 masks, used by the Rust
rollout) must agree with MortalActionSpace (built from Python-engine action XML,
the space bc49 was trained in) at every decision of real games.

The Python game loop is driven by a checking chooser: at each decision it builds
both masks, asserts equality for the first step, the follow-up mode and the
second step, then maps the chosen slots back to a native index with
m46_native.to_native and asserts it is the index of the engine action that
MortalActionSpace resolves to. The policy is random but biased toward the
declare / call slots so riichi, kan (incl. multi-tile kan select), chi
low/mid/high, pon and agari all get exercised."""
import random
from collections import Counter

import numpy as np
import torch

from src.agents.dnn import m46_native as mn
from src.agents.dnn import mortal_action as ma
from src.agents.dnn import selfplay
from src.agents.dnn.action_space import REGISTRY
from src.agents.dnn.encoder import action_to_index, encode_state, legal_mask

SPACE = REGISTRY["mortal46"]
_PREFER = {ma.IDX_RIICHI: 6.0, ma.IDX_KAN: 6.0, ma.IDX_PON: 3.0, ma.IDX_CHI_LOW: 3.0,
           ma.IDX_CHI_MID: 3.0, ma.IDX_CHI_HIGH: 3.0, ma.IDX_AGARI: 50.0, ma.IDX_PASS: 0.5}


def _pick(mask46, rng):
    legal = np.nonzero(mask46)[0]
    w = np.array([_PREFER.get(int(s), 1.0) for s in legal])
    return int(rng.choice(legal, p=w / w.sum()))


def _closed_pick(m1, lookup1, table, pid, rng):
    """'closed' policy: agari > riichi > kan > shanten-minimising discard, never call --
    reaches riichi / ron / tsumo / riichi-select (incl. red fives) far more often than random."""
    for s in (ma.IDX_AGARI, ma.IDX_RIICHI, ma.IDX_KAN):
        if m1[s]:
            return s
    disc = [s for s in np.nonzero(m1[:37])[0]]
    if not disc:
        return ma.IDX_PASS if m1[ma.IDX_PASS] else int(rng.choice(np.nonzero(m1)[0]))
    hand = list(table.hands[pid])
    best, cands = 99, []
    for s in disc:
        t = ma.TILE_RE.search(lookup1[int(s)]).group(1)
        rest = list(hand)
        for i, h in enumerate(rest):
            if h == t:
                rest.pop(i)
                break
        sh = table._shanten(rest, len(table.melds[pid]))
        if sh < best:
            best, cands = sh, [int(s)]
        elif sh == best:
            cands.append(int(s))
    return int(rng.choice(cands))


def _make_checker(rng, stats, policy="random"):
    def choose(net, table, pid, actions, temperature, device, cmode="none"):
        native, _ = legal_mask(actions)
        M = native.numpy()[None]
        planes, scalars = encode_state(table, pid, variant="v3r")
        called = mn.called_from_planes(planes.numpy()[None])
        m1_ref, lookup1 = SPACE.mask(actions)
        m1 = mn.first_mask(M, called)[0]
        assert np.array_equal(m1, m1_ref.numpy()), (actions, np.nonzero(m1)[0], np.nonzero(m1_ref.numpy())[0])
        s1 = _closed_pick(m1, lookup1, table, pid, rng) if policy == "closed" else _pick(m1, rng)
        stats[("s1", s1)] += 1
        fu = SPACE.follow_up(s1, actions)
        mode = mn.follow_up_mode(np.array([s1]), M)[0]
        assert mode == {None: 0, "riichi": mn.RIICHI_MODE, "kan": mn.KAN_MODE}[fu], (s1, actions)
        s2, lookup = -1, lookup1
        if fu:
            m2_ref, lookup = SPACE.mask(actions, mode=fu)
            m2 = mn.second_mask(M, np.array([mode]))[0]
            assert np.array_equal(m2, m2_ref.numpy()), (fu, actions)
            s2 = _pick(m2, rng)
            stats[("s2", fu)] += 1
            stats[("s2", fu, "red" if s2 >= 34 else "plain")] += 1
            a = SPACE.resolve(s2, lookup)
        else:
            a = SPACE.resolve(s1, lookup1)
        nat = mn.to_native(np.array([s1]), np.array([s2]), M, called)[0]
        assert nat == action_to_index(a), (s1, s2, a, nat, action_to_index(a))
        stats["decisions"] += 1
        st = selfplay.DnnStep(planes=planes, scalars=scalars, mask=torch.from_numpy(m1),
                              action_idx=s1, logprob=0.0)
        return [st], a
    return choose


def run(n_games=200, seed=0, policy="random"):
    rng = np.random.default_rng(seed)
    stats = Counter()
    orig = selfplay._choose
    selfplay._choose = _make_checker(rng, stats, policy)
    try:
        for g in range(n_games):
            random.seed(seed * 100003 + g)
            selfplay.play_game(None, deal_seed=seed * 100003 + g)
    finally:
        selfplay._choose = orig
    return stats


def test_m46_native_matches_mortal_space():
    stats = run(120, seed=1)
    assert stats["decisions"] > 5000
    for s in (ma.IDX_PON, ma.IDX_KAN, ma.IDX_CHI_LOW,          # riichi / agari: closed-policy test
              ma.IDX_CHI_MID, ma.IDX_CHI_HIGH, ma.IDX_PASS):
        assert stats[("s1", s) ] > 0, f"slot {s} never exercised"



def _check_static(actions, called=-1):
    """first/second masks and native round trip for a hand-written action list."""
    M = legal_mask(actions)[0].numpy()[None]
    c = np.array([called])
    m1_ref, lk1 = SPACE.mask(actions)
    assert np.array_equal(mn.first_mask(M, c)[0], m1_ref.numpy())
    for s1 in np.nonzero(m1_ref.numpy())[0]:
        fu = SPACE.follow_up(int(s1), actions)
        mode = mn.follow_up_mode(np.array([s1]), M)
        assert mode[0] == {None: 0, "riichi": mn.RIICHI_MODE, "kan": mn.KAN_MODE}[fu]
        if not fu:
            assert mn.to_native(np.array([s1]), np.array([-1]), M, c)[0] == action_to_index(SPACE.resolve(int(s1), lk1))
            continue
        m2_ref, lk2 = SPACE.mask(actions, mode=fu)
        assert np.array_equal(mn.second_mask(M, mode)[0], m2_ref.numpy())
        for s2 in np.nonzero(m2_ref.numpy())[0]:
            assert mn.to_native(np.array([s1]), np.array([s2]), M, c)[0] == action_to_index(SPACE.resolve(int(s2), lk2))


def test_m46_native_static_rare_cases():
    # kyuushu on the first draw, alongside discards and a tsumo
    _check_static(['<action type="discard" tile="1m" />', '<action type="discard" tile="9p" />',
                   '<action type="kyuushu" />', '<action type="tsumo" />'])
    # two kan tiles (ankan + shouminkan) -> kan-select second step; red/plain riichi discards
    _check_static(['<action type="discard" tile="3s" />', '<action type="discard" tile="0p" />',
                   '<action type="discard" tile="5p" />', '<action type="riichi" tile="3s" />',
                   '<action type="riichi" tile="0p" />', '<action type="riichi" tile="5p" />',
                   '<action type="kan" tile="2m" />', '<action type="kan" tile="7z" />'])
    # pon + daiminkan + ron + three chi shapes on a called 4p (key tiles 5p / 3p / 2p)
    _check_static(['<action type="skip" />', '<action type="pon" tile="4p" />', '<action type="kan" tile="4p" />',
                   '<action type="ron" tile="4p" />', '<action type="chi" tile="4p" with="5p 6p" />',
                   '<action type="chi" tile="4p" with="3p 5p" />', '<action type="chi" tile="4p" with="2p 3p" />'],
                  called=12)
    # edge-of-suit chi (called 9s: only the high shape exists; called 1m: only low)
    _check_static(['<action type="skip" />', '<action type="chi" tile="9s" with="7s 8s" />'], called=26)
    _check_static(['<action type="skip" />', '<action type="chi" tile="1m" with="2m 3m" />'], called=0)


def test_m46_native_closed_policy_riichi_agari():
    stats = run(80, seed=2, policy="closed")
    assert stats[("s2", "riichi")] > 20
    assert stats[("s1", ma.IDX_AGARI)] > 10


def test_rust_rollout_m46_repeated_seed_same_slot():
    """exp124 arm B crash: with dup_k a VecEnv slot can host the SAME seed twice in one rollout; the
    row key must tell those games apart (per-slot game counter, not the seed). 4 slots x 6 copies of
    one seed forces every slot to replay it; every rewritten episode must stay inside its 46-slot mask."""
    import pytest
    pytest.importorskip("riichi_rs")
    from src.agents.dnn.arch_zoo import ZOO
    from src.agents.dnn.rust_rollout import collect_rust
    torch.manual_seed(0)
    net = ZOO["convformer_m_v3r_m46"][0]().eval()
    cfg = {"temperature": 1.0, "gamma": 0.995, "shaping": False, "games_per_worker": 4,
           "rollout_values": True}
    episodes, results = collect_rust(net, 24, cfg, 1, [11] * 24, device="cpu")
    assert len(results) == 24
    for e in episodes:
        m = np.asarray(e["mask"]); a = np.asarray(e["actions"])
        assert m.shape[1] == 46 and m[np.arange(len(a)), a].all()
        assert len(e["values"]) == len(a) == len(e["old_logprobs"])


if __name__ == "__main__":
    import sys
    st = run(int(sys.argv[1]) if len(sys.argv) > 1 else 300, seed=int(sys.argv[2]) if len(sys.argv) > 2 else 0,
             policy=sys.argv[3] if len(sys.argv) > 3 else "random")
    print(dict(sorted(((str(k), v) for k, v in st.items()))))

