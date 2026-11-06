"""P0: riichi_rs.PyRandom must be bit-exact with CPython's random module."""
import random

import pytest

riichi_rs = pytest.importorskip("riichi_rs")


@pytest.mark.parametrize("seed", [0, 1, 7, 12345, 6_000_000 + 9973, 67_000_000, 2**32 + 5, 2**40 + 12345])
def test_random_and_getrandbits_match(seed):
    py = random.Random(seed); rs = riichi_rs.PyRandom(seed)
    for _ in range(2000):
        assert py.random() == rs.random()
    for k in (1, 3, 7, 16, 31, 32, 33, 40, 53, 64):
        for _ in range(50):
            assert py.getrandbits(k) == rs.getrandbits(k)


@pytest.mark.parametrize("seed", [3, 999, 6_000_000 + 9973 * 5 + 2, 53_100_000])
def test_shuffle_randrange_choices_gauss_match(seed):
    py = random.Random(seed); rs = riichi_rs.PyRandom(seed)
    base = list(range(136))
    a = list(base); py.shuffle(a)
    b = rs.shuffle(base)
    assert a == b
    for n in (4, 5, 14, 34, 137):
        for _ in range(20):
            assert py.randrange(n) == rs.randrange(n)
    for _ in range(30):
        assert py.uniform(0.5, 1.5) == rs.uniform(0.5, 1.5)
    for _ in range(41):                                   # odd count exercises the cached value
        assert py.gauss(0.0, 1.0) == rs.gauss(0.0, 1.0)
    for _ in range(200):
        assert py.choices((0, 1, 2, 3), weights=(70, 20, 8, 2))[0] == rs.choices_index([70.0, 20.0, 8.0, 2.0])


def test_reseed_resets_gauss_cache():
    py = random.Random(11); rs = riichi_rs.PyRandom(11)
    py.gauss(0, 1); rs.gauss(0, 1)
    py.seed(11); rs.seed(11)
    assert py.gauss(0, 1) == rs.gauss(0, 1)
