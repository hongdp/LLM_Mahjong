"""exp89 style-conditioned population: Table.seat_style rewards (tau_d deal-in penalty, tau_a win bonus),
the v3s encoder (v3r + 2 style scalars) and VecEnv(seat_styles=...)."""
import random
import numpy as np
import pytest

riichi_rs = pytest.importorskip("riichi_rs")


def _random_play(seed, style=None, guard_max=600):
    """Deterministic random legal play (record actions on the first run and replay them for equality tests)."""
    rng = random.Random(seed)
    t = riichi_rs.Table(seed, True)
    if style is not None:
        t.seat_style = style
    guard = 0
    while not t.finished and guard < guard_max:
        guard += 1
        pid = t.turn
        acts = t.get_legal_actions(pid)
        if not acts:
            break
        win = [a for a in acts if 'type="tsumo"' in a or 'type="ron"' in a]
        _, done, info = t.step(pid, win[0] if win else rng.choice(acts))
        if done:
            break
        if not (info.get("discarded") or info.get("chankan")):
            continue
        cands = []
        for off in range(1, 4):
            o = (pid + off) % 4
            opts = t.get_interrupt_actions(o)
            if len(opts) > 1:
                ron = [a for a in opts if 'type="ron"' in a]
                cands.append((o, ron[0] if ron else rng.choice(opts)))
        executed, done = t.resolve_claims(cands)[:2]
        if done:
            break
        if not executed:
            if t.pending_kan is not None:
                t.resolve_pending_kan()
            elif t.advance_turn():
                break
    return t


def test_v3s_encoder_is_v3r_plus_style_scalars():
    t = riichi_rs.Table(4242, True)
    p3, s3 = t.encode(0, "v3r")
    ps, ss = t.encode(0, "v3s")
    assert ps == p3 and len(ss) == len(s3) + 2 == riichi_rs.N_SCALARS_V3S
    assert ss[:len(s3)] == s3 and ss[-2:] == [0.0, 0.0]
    t.seat_style = [[4.0, 0.5], [0, 0], [0, 0], [0, 0]]
    _, ss = t.encode(0, "v3s")
    assert ss[-2:] == pytest.approx([0.5, 0.5])
    _, ss1 = t.encode(1, "v3s")
    assert ss1[-2:] == [0.0, 0.0]


def test_seat_style_shifts_rewards_by_tau():
    """Same seed + same random actions: deal-in seat loses tau_d, winners gain tau_a * (own positive delta) — everything else unchanged."""
    style = [[3.0, 0.25], [1.0, 0.5], [5.0, 0.0], [0.0, 1.0]]
    checked_ron = checked_draw = 0
    for seed in range(1, 3000):
        t0 = _random_play(seed)
        t1 = _random_play(seed, style)
        assert t0.result_summary == t1.result_summary and t0.points == t1.points
        fr0, fr1 = t0.final_rewards, t1.final_rewards
        exp = list(fr0)
        if "荣和" in t0.result_summary:
            h = t0.last_discarder
            exp[h] -= style[h][0]
        for w in range(4):
            d = (t0.points[w] - t0.start_points[w]) * 0.001
            if d > 0 and ("荣和" in t0.result_summary or "自摸" in t0.result_summary):
                exp[w] += style[w][1] * d
        assert fr1 == pytest.approx(exp, abs=1e-9), (seed, t0.result_summary, fr0, fr1)
        checked_ron += "荣和" in t0.result_summary
        checked_draw += "流局" in t0.result_summary
        if checked_ron >= 10 and checked_draw >= 5:
            break
    assert checked_ron >= 10 and checked_draw >= 5, (checked_ron, checked_draw)


def test_vecenv_seat_styles_reach_the_observation():
    seeds = [7_000_001, 7_000_002]
    styles = [[2.0, 0.5, 0, 0, 0, 0, 4.0, 1.0], [0.0] * 8]
    env = riichi_rs.VecEnv(seeds, 2, 0.995, False, 1.0, True, "v3s", False, 24, 0.0, seat_styles=styles)
    planes, scalars, mask, seats, gids = env.observe()
    assert scalars.shape[1] == riichi_rs.N_SCALARS_V3S
    seed_of = env.slot_seeds()
    for row in range(scalars.shape[0]):
        seed = seed_of[gids[row]]
        st = styles[seeds.index(seed)]
        seat = int(seats[row])
        assert scalars[row, -2] == pytest.approx(st[2 * seat] / 8.0) and scalars[row, -1] == pytest.approx(st[2 * seat + 1])
    with pytest.raises(ValueError):
        riichi_rs.VecEnv(seeds, 2, 0.995, False, 1.0, True, "v3s", False, 24, 0.0, seat_styles=[[0.0] * 8])
