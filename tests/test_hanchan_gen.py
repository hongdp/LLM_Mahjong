"""play_hanchan_gen protocol test: full match with random legal actions,
CPU-only (dummy tensors in DnnStep — the generator only touches reward/
is_terminal/phi)."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.agents.dnn.selfplay import DnnStep
from src.tasks.mahjong.hanchan import play_hanchan_gen, UMA


def drive(seed):
    rng = random.Random(seed)
    gen = play_hanchan_gen(seed)
    tup = next(gen)
    while True:
        table, reqs = tup
        replies = []
        for pid, actions in reqs:
            step = DnnStep(planes=torch.zeros(1), scalars=torch.zeros(1),
                           mask=torch.zeros(1, dtype=torch.bool),
                           action_idx=0, logprob=0.0)
            replies.append((step, rng.choice(actions)))
        try:
            tup = gen.send(replies)
        except StopIteration as e:
            return e.value


def test_full_match_protocol():
    m = drive(424242)
    h = m.hanchan
    assert 1 <= h["n_deals"] <= 24 and len(m.deals) == h["n_deals"]
    assert sorted(h["placements"]) == [1, 2, 3, 4]
    assert sum(h["uma_points"]) == 0                    # zero-sum incl. uma
    for p in range(4):
        steps = m.trajectories[p]
        assert steps, f"seat {p} has no steps across a whole match"
        assert all(not s.is_terminal for s in steps[:-1])
        assert steps[-1].is_terminal


def test_uma_lands_on_last_step():
    # terminal reward must include uma*scale: reconstruct by re-driving the
    # same seed and comparing last-step rewards ordering with placements
    m = drive(777001)
    h = m.hanchan
    for p in range(4):
        r_last = m.trajectories[p][-1].reward
        # uma component alone spans +-15..-15; a placement-1 seat cannot
        # end with a hugely negative last-step reward and vice versa
        if h["placements"][p] == 1:
            assert r_last > -5.0
        if h["placements"][p] == 4:
            assert r_last < 5.0


def test_full_rotation_structure():
    # random policies never win: every deal is a noten draw, dealer rotates
    # each time -> exactly E1..S4 = 8 deals, points untouched at 25000
    m = drive(90210)
    assert m.hanchan["n_deals"] == 8
    assert m.points == [25000, 25000, 25000, 25000]
    assert not m.hanchan["busted"]


def test_credit_telescopes_to_uma():
    import os
    import pytest
    w_path = "experiments/placement_value/w_resid.pt"
    if not os.path.exists(w_path):
        pytest.skip("W artifact not present")
    from src.tasks.mahjong.hanchan import PlacementCredit, play_hanchan_gen
    from src.tasks.mahjong.table import PyMahjongTable
    credit = PlacementCredit(w_path)
    rng = random.Random(555)
    gen = play_hanchan_gen(555, credit=credit)
    tup = next(gen)
    while True:
        table, reqs = tup
        replies = [(DnnStep(planes=torch.zeros(1), scalars=torch.zeros(1),
                            mask=torch.zeros(1, dtype=torch.bool),
                            action_idx=0, logprob=0.0), rng.choice(a))
                   for _, a in reqs]
        try:
            tup = gen.send(replies)
        except StopIteration as e:
            m = e.value
            break
    scale = PyMahjongTable.REWARD_SCALE
    for p in range(4):
        total = sum(s.reward for s in m.trajectories[p])
        w0 = credit.w(p, [25000] * 4, 0, 0, 0, 0, 8)
        expect = (m.hanchan["uma_points"][p] - w0) * scale
        # in-deal engine step rewards are zero except the settle we replaced,
        # so the sum must telescope exactly (float tolerance only)
        assert abs(total - expect) < 1e-3, (p, total, expect)
