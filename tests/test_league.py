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


def test_variant_of_arch_ignores_action_space_suffix():
    # exp46-C regression: '_m46' shadowed the encoder suffix under endswith,
    # so 46-slot league opponents were encoded as v1 and crashed the stack.
    from src.agents.dnn.encoder import variant_of_arch
    assert variant_of_arch("convformer_m_v3r_m46") == "v3r"
    assert variant_of_arch("mortal_bb_xl_v3r_m46") == "v3r"
    assert variant_of_arch("hrf_xl_v4_m46") == "v4"
    assert variant_of_arch("cnn_m_v3r") == "v3r"
    assert variant_of_arch("cnn_m_r") == "v1r"
    assert variant_of_arch("cnn_m") == "v1"
