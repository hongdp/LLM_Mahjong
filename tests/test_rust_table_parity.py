"""P1: riichi_rs.Table deals exactly like PyMahjongTable after random.seed(seed)."""
import random

import pytest

riichi_rs = pytest.importorskip("riichi_rs")
from src.tasks.mahjong.table import PyMahjongTable


def _py(seed, rr):
    random.seed(seed)
    t = PyMahjongTable(randomize_round=rr)
    t.text_obs = False
    return t


@pytest.mark.parametrize("rr", [True, False])
def test_deal_parity_many_seeds(rr):
    seeds = list(range(0, 400)) + [6_000_000 + 9973 * k + d for k in range(1, 6) for d in range(0, 40, 7)] + [67_000_000, 68_000_000 + 15999]
    for s in seeds:
        py = _py(s, rr); rs = riichi_rs.Table(s, rr)
        assert rs.dealer == py.dealer and rs.round_wind_idx == py.round_wind_idx, s
        assert rs.round_number == py.round_number and rs.turn == py.turn, s
        assert rs.points == list(py.points) and rs.kyotaku == py.kyotaku, (s, rs.points, py.points)
        assert rs.start_points == list(py.start_points)
        assert rs.wall == list(py.wall), s
        assert rs.dead_wall == list(py.dead_wall), s
        assert rs.dora_indicators == list(py.dora_indicators) and rs.ura_indicators == list(py.ura_indicators)
        for p in range(4):
            assert rs.hands[p] == list(py.hands[p]), (s, p)
            assert rs.red[p] == [py.red[p]["m"], py.red[p]["p"], py.red[p]["s"]], (s, p)
        assert rs.last_drawn == list(py.last_drawn), s
        assert rs.last_drawn_red == list(py.last_drawn_red), s
