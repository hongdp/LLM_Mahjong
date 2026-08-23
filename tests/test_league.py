"""exp22 league: deterministic seat plan across dup replicas; only learner
seats yield episodes; mirror games (frac 0) unchanged."""
import sys
sys.path.insert(0, ".")
from src.agents.dnn.parallel_rollout import league_plan   # noqa: E402


def test_plan_deterministic_and_valid():
    cfg = {"league": [{"name": "a", "path": "x"}, {"name": "b", "path": "y"}], "league_frac": 1.0}
    for seed in range(50):
        l1, o1 = league_plan(seed, cfg)
        l2, o2 = league_plan(seed, cfg)
        assert (l1, o1) == (l2, o2)
        assert 1 <= len(l1) <= 2 and set(l1).isdisjoint(o1)
        assert set(l1) | set(o1) == {0, 1, 2, 3}
        assert all(0 <= j < 2 for j in o1.values())


def test_frac_zero_is_mirror():
    cfg = {"league": [{"name": "a", "path": "x"}], "league_frac": 0.0}
    assert league_plan(123, cfg) == ([0, 1, 2, 3], {})


def test_frac_respected_roughly():
    cfg = {"league": [{"name": "a", "path": "x"}], "league_frac": 0.5}
    n = sum(1 for s in range(2000) if league_plan(s, cfg)[1])
    assert 850 < n < 1150
