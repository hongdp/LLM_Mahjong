"""Tests for the Mortal-aligned observation encoder and action space (exp41).

These guard the two things a hand-port can silently get wrong: the plane
layout drifting from the Rust source, and two of our bundled actions collapsing
onto one Mortal slot.
"""

import random
import sys, os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks.mahjong.table import PyMahjongTable                  # noqa: E402
from src.agents.dnn.mortal_obs import (encode_mortal_obs,           # noqa: E402
                                       MORTAL_V3_PLANES)
from src.agents.dnn.mortal_action import (                          # noqa: E402
    action_to_slot, legal_mask_46, collisions, MORTAL_ACTION_DIM,
    IDX_CHI_LOW, IDX_CHI_MID, IDX_CHI_HIGH, IDX_PON, IDX_KAN,
    IDX_AGARI, IDX_RYUKYOKU, IDX_PASS, IDX_RIICHI, tile_to_slot)


def _play(seed, steps=40):
    """Drive a game with random legal actions and yield decision points."""
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
        yield t, pid, acts
        _, _, done, info = t.step(pid, random.choice(acts))
        if done or not (info.get("discarded") or info.get("chankan")):
            if done:
                break
            continue
        for off in range(1, 4):
            io = t.get_interrupt_actions((pid + off) % 4)
            if len(io) > 1:
                yield t, (pid + off) % 4, io
        try:
            if t.pending_kan:
                t.resolve_pending_kan()
            else:
                _, rd = t.advance_turn()
                if rd:
                    break
        except Exception:
            break


# --------------------------------------------------------------- observation

def test_obs_shape_and_layout_cursor():
    """The encoder asserts internally that the cursor lands on 934; this
    exercises that guard across many distinct states."""
    for seed in (1, 2, 3):
        for t, pid, _ in _play(seed, steps=12):
            obs = encode_mortal_obs(t, pid, derived=True)
            assert obs.shape == (MORTAL_V3_PLANES, 34)
            assert obs.dtype == np.float32


def test_obs_is_finite_and_bounded():
    for t, pid, _ in _play(11, steps=25):
        obs = encode_mortal_obs(t, pid, derived=True)
        assert np.isfinite(obs).all()
        assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_obs_deterministic():
    for t, pid, _ in _play(12, steps=8):
        a = encode_mortal_obs(t, pid, derived=True)
        b = encode_mortal_obs(t, pid, derived=True)
        assert np.array_equal(a, b)


def test_pure_arm_differs_only_in_derived_block():
    """Arm B must be bit-identical to arm A outside the derived planes, so the
    two arms' difference isolates exactly the tile-efficiency knowledge."""
    seen_any_difference = False
    for t, pid, _ in _play(13, steps=30):
        a = encode_mortal_obs(t, pid, derived=True)
        b = encode_mortal_obs(t, pid, derived=False)
        diff = np.where((a != b).any(axis=1))[0]
        # every differing plane must sit in the derived region (waits ..
        # discard-candidate block), never in the public-record region
        if len(diff):
            seen_any_difference = True
            assert diff.min() >= 900, f"derived leak at plane {diff.min()}"
    assert seen_any_difference, "derived gate never exercised"


# -------------------------------------------------------------- action space

@pytest.mark.parametrize("xml,want", [
    ('<action type="chi" tile="7s" with="8s 9s" />', IDX_CHI_LOW),
    ('<action type="chi" tile="7s" with="6s 8s" />', IDX_CHI_MID),
    ('<action type="chi" tile="7s" with="5s 6s" />', IDX_CHI_HIGH),
    ('<action type="pon" tile="3m" />', IDX_PON),
    ('<action type="kan" tile="3m" />', IDX_KAN),
    ('<action type="ron" tile="3m" />', IDX_AGARI),
    ('<action type="tsumo" tile="3m" />', IDX_AGARI),
    ('<action type="kyuushu" />', IDX_RYUKYOKU),
    ('<action type="skip" />', IDX_PASS),
    ('<action type="riichi" tile="3m" />', IDX_RIICHI),
])
def test_action_slot_mapping(xml, want):
    assert action_to_slot(xml) == want


def test_discard_tile_slots():
    assert action_to_slot('<action type="discard" tile="1m" />') == 0
    assert action_to_slot('<action type="discard" tile="9s" />') == 26
    assert action_to_slot('<action type="discard" tile="1z" />') == 27
    assert action_to_slot('<action type="discard" tile="7z" />') == 33
    # red fives get their own slots, distinct from the plain five
    assert action_to_slot('<action type="discard" tile="0m" />') == 34
    assert action_to_slot('<action type="discard" tile="5m" />') == tile_to_slot("5m")
    assert tile_to_slot("0m") != tile_to_slot("5m")


def test_two_step_selects_resolve_collisions():
    """Riichi and kan bundle a tile in our engine but are two-step in Mortal;
    the select flags must make each option land on its own slot."""
    riichi = ['<action type="riichi" tile="3m" />',
              '<action type="riichi" tile="6m" />']
    assert collisions(riichi) == {IDX_RIICHI: riichi}          # collide bundled
    assert collisions(riichi, at_riichi_select=True) == {}     # resolved
    kans = ['<action type="kan" tile="3m" />', '<action type="kan" tile="6m" />']
    assert collisions(kans) == {IDX_KAN: kans}
    assert collisions(kans, at_kan_select=True) == {}


def test_every_engine_action_maps_and_no_collisions_in_real_play():
    """Across real games, every legal action maps to a slot, and the only
    collisions are the two-step ones (riichi / kan), which the select flags fix."""
    unmapped = []
    for seed in (70000, 70001, 70002, 70003):
        for t, pid, acts in _play(seed, steps=60):
            for a in acts:
                if action_to_slot(a) is None:
                    unmapped.append(a)
            for slot, group in collisions(acts).items():
                assert slot in (IDX_RIICHI, IDX_KAN), (
                    f"unexpected collision on slot {slot}: {group}")
                resolved = collisions(
                    group,
                    at_riichi_select=(slot == IDX_RIICHI),
                    at_kan_select=(slot == IDX_KAN))
                assert resolved == {}, f"select flag failed to resolve {group}"
    assert not unmapped, f"unmapped engine actions: {unmapped[:5]}"


def test_mask_marks_only_legal_slots():
    for t, pid, acts in _play(70010, steps=40):
        mask, lookup = legal_mask_46(acts)
        assert len(mask) == MORTAL_ACTION_DIM
        assert any(mask), "a decision with legal actions produced an empty mask"
        for slot, xml in lookup.items():
            assert mask[slot]
            assert xml in acts
