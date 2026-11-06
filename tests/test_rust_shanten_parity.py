"""P2a: riichi_rs shanten / waits / riichi-ankan decomposition == Python engine."""
import random

import pytest

riichi_rs = pytest.importorskip("riichi_rs")
from src.tasks.mahjong.table import PyMahjongTable, _tile_only_as_triplet, norm_tile

ALL = [f"{i}{s}" for s in "mps" for i in range(1, 10)] + [f"{i}z" for i in range(1, 8)]
WALL = ALL * 4


def _hand(rng, n):
    return sorted(rng.sample(WALL, n))


def test_shanten_parity_random_hands():
    rng = random.Random(2026)
    t = PyMahjongTable(randomize_round=False); t.text_obs = False
    n_checked = 0
    for _ in range(60000):
        m = rng.randrange(5)                     # melds 0..4
        size = rng.choice((13, 14)) - 3 * m
        h = _hand(rng, size)
        assert riichi_rs.shanten(h, m) == t._shanten(h, m), (h, m)
        n_checked += 1
    # biased-to-tenpai hands: build from sets/pairs so agari/tenpai branches are hit
    for _ in range(20000):
        pieces = []
        for _ in range(4):
            if rng.random() < 0.5:
                a = rng.randrange(34); pieces += [ALL[a]] * 3
            else:
                s = rng.choice("mps"); a = rng.randrange(1, 8); pieces += [f"{a}{s}", f"{a+1}{s}", f"{a+2}{s}"]
        p = rng.randrange(34); pieces += [ALL[p]] * 2
        counts = {x: pieces.count(x) for x in set(pieces)}
        if any(v > 4 for v in counts.values()):
            continue
        drop = rng.randrange(3)                  # 14, 13 or 12+... keep valid sizes
        h = list(pieces)
        for _ in range(drop):
            h.pop(rng.randrange(len(h)))
        if len(h) % 3 == 0:
            continue
        m = 0
        assert riichi_rs.shanten(h, m) == t._shanten(h, m), (h, m)


def test_waits_parity():
    rng = random.Random(7)
    t = PyMahjongTable(randomize_round=False); t.text_obs = False
    hits = 0
    for _ in range(30000):
        m = rng.randrange(3)
        h = _hand(rng, 13 - 3 * m)
        py = sorted(t._waits_of(h, m)); rs = sorted(riichi_rs.waits(h, m))
        assert py == rs, (h, m, py, rs)
        hits += bool(py)
    # structured tenpai hands
    for _ in range(5000):
        pieces = []
        for _ in range(4):
            if rng.random() < 0.5:
                a = rng.randrange(34); pieces += [ALL[a]] * 3
            else:
                s = rng.choice("mps"); a = rng.randrange(1, 8); pieces += [f"{a}{s}", f"{a+1}{s}", f"{a+2}{s}"]
        p = rng.randrange(34); pieces += [ALL[p]] * 2
        if any(pieces.count(x) > 4 for x in set(pieces)):
            continue
        pieces.pop(rng.randrange(14))
        py = sorted(t._waits_of(pieces, 0)); rs = sorted(riichi_rs.waits(pieces, 0))
        assert py == rs, (pieces, py, rs)
        hits += bool(py)
    assert hits > 1000


def test_tile_only_as_triplet_parity():
    rng = random.Random(11)
    for _ in range(4000):
        pieces = []
        for _ in range(4):
            if rng.random() < 0.6:
                a = rng.randrange(34); pieces += [ALL[a]] * 3
            else:
                s = rng.choice("mps"); a = rng.randrange(1, 8); pieces += [f"{a}{s}", f"{a+1}{s}", f"{a+2}{s}"]
        p = rng.randrange(34); pieces += [ALL[p]] * 2
        if any(pieces.count(x) > 4 for x in set(pieces)):
            continue
        tile = rng.choice(pieces)
        n_sets = 4 - rng.randrange(2)
        assert riichi_rs.tile_only_as_triplet(pieces, tile, n_sets) == _tile_only_as_triplet(pieces, tile, n_sets), (pieces, tile, n_sets)
