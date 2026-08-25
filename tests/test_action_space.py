"""Action-space adapter tests.

The load-bearing test here is `test_native_space_is_byte_identical`: it is the
guarantee that introducing the adapter did not perturb any existing model or
checkpoint. The rest cover the multi-step protocol containment.
"""

import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks.mahjong.table import PyMahjongTable                    # noqa: E402
from src.agents.dnn.encoder import legal_mask, ACTION_DIM             # noqa: E402
from src.agents.dnn.action_space import (                             # noqa: E402
    REGISTRY, get_space, space_of_arch, NativeActionSpace, MortalActionSpace)
from src.agents.dnn.mortal_action import (                            # noqa: E402
    IDX_RIICHI, IDX_KAN, MORTAL_ACTION_DIM)


def _decisions(seed, steps=50):
    random.seed(seed)
    t = PyMahjongTable(randomize_round=True)
    t.text_obs = False
    guard = 0
    while not t.finished and guard < steps:
        guard += 1
        pid = t.turn
        acts = t.get_legal_actions(pid)
        if not acts:
            break
        yield acts
        _, _, done, info = t.step(pid, random.choice(acts))
        if done:
            break
        if not (info.get("discarded") or info.get("chankan")):
            continue
        for off in range(1, 4):
            io = t.get_interrupt_actions((pid + off) % 4)
            if len(io) > 1:
                yield io
        try:
            if t.pending_kan:
                t.resolve_pending_kan()
            else:
                _, rd = t.advance_turn()
                if rd:
                    break
        except Exception:
            break


def test_native_space_is_byte_identical():
    """The adapter must be a pure pass-through for the incumbent space, or
    every pre-existing checkpoint silently changes behaviour."""
    space = REGISTRY["native"]
    n = 0
    for acts in _decisions(4242, steps=80):
        want_mask, want_lookup = legal_mask(acts)
        got_mask, got_lookup = space.mask(acts)
        assert np.array_equal(np.asarray(want_mask), np.asarray(got_mask))
        assert want_lookup == got_lookup
        assert space.follow_up(int(np.asarray(got_mask).nonzero()[0][0]), acts) is None
        n += 1
    assert n > 50, "test did not exercise enough decisions"


def test_native_space_never_requests_follow_up():
    space = REGISTRY["native"]
    for acts in _decisions(4243, steps=60):
        mask, _ = space.mask(acts)
        for slot in np.asarray(mask).nonzero()[0]:
            assert space.follow_up(int(slot), acts) is None


def test_dims_are_declared_correctly():
    assert REGISTRY["native"].dim == ACTION_DIM == 374
    assert REGISTRY["mortal46"].dim == MORTAL_ACTION_DIM == 46


def test_space_resolution_defaults_to_native():
    # every existing arch name stays native
    for arch in ("cnn_m_r", "cnn_xl_r", "hrf_xl_v4", "handset_xl_cnn_m_r",
                 "mortal_bb_xl_r", "", None):
        assert space_of_arch(arch) == "native"
    # only an explicit marker opts in
    assert space_of_arch("mortal_bb_xl_m46") == "mortal46"
    # unknown names fall back rather than raising, so no checkpoint is locked out
    assert get_space("no_such_space").name == "native"


def test_mortal_riichi_two_step():
    space = REGISTRY["mortal46"]
    acts = ['<action type="discard" tile="1m" />',
            '<action type="riichi" tile="3m" />',
            '<action type="riichi" tile="6m" />']
    mask, lookup = space.mask(acts)
    assert mask[IDX_RIICHI], "riichi must be offered as a single declaration slot"
    # picking the declaration slot demands a second query...
    mode = space.follow_up(IDX_RIICHI, acts)
    assert mode == "riichi"
    # ...whose mask covers exactly the riichi discards, each on its own slot
    m2, lk2 = space.mask(acts, mode=mode)
    chosen = sorted(lk2.values())
    assert chosen == ['<action type="riichi" tile="3m" />',
                      '<action type="riichi" tile="6m" />']
    assert int(np.asarray(m2).sum()) == 2, "two riichi tiles must occupy two slots"


def test_mortal_kan_follow_up_only_when_ambiguous():
    space = REGISTRY["mortal46"]
    one = ['<action type="skip" />', '<action type="kan" tile="3m" />']
    assert space.follow_up(IDX_KAN, one) is None, "single kan needs no second step"
    two = ['<action type="skip" />',
           '<action type="kan" tile="3m" />',
           '<action type="kan" tile="6m" />']
    assert space.follow_up(IDX_KAN, two) == "kan"
    m2, lk2 = space.mask(two, mode="kan")
    assert sorted(lk2.values()) == ['<action type="kan" tile="3m" />',
                                    '<action type="kan" tile="6m" />']


def test_mortal_second_step_never_recurses():
    space = REGISTRY["mortal46"]
    acts = ['<action type="riichi" tile="3m" />']
    m2, lk2 = space.mask(acts, mode="riichi")
    slot = int(np.asarray(m2).nonzero()[0][0])
    assert space.follow_up(slot, acts, mode="riichi") is None


def test_resolve_returns_engine_action():
    for name in ("native", "mortal46"):
        space = REGISTRY[name]
        for acts in _decisions(4244, steps=30):
            mask, lookup = space.mask(acts)
            for slot in np.asarray(mask).nonzero()[0]:
                slot = int(slot)
                if space.follow_up(slot, acts) is not None:
                    continue
                assert space.resolve(slot, lookup) in acts
