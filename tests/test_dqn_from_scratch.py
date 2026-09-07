import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "train_dnn_dqn", os.path.join(os.path.dirname(__file__), "..", "scripts", "train_dnn_dqn.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_temp_schedule_interpolates_and_clamps():
    s = mod.parse_temp_schedule("300000:0.05,0:1.0,50000:0.1")
    assert s == [(0, 1.0), (50000, 0.1), (300000, 0.05)]
    assert mod.temp_at(s, 0) == 1.0
    assert abs(mod.temp_at(s, 25000) - 0.55) < 1e-9
    assert abs(mod.temp_at(s, 175000) - 0.075) < 1e-9
    assert mod.temp_at(s, 10_000_000) == 0.05
    assert mod.parse_temp_schedule(None) == []


def test_dqn_args_allow_random_init_and_mirror(monkeypatch):
    monkeypatch.setattr("sys.argv", ["x", "--arch", "cnn_m_r", "--exp_dir", "/tmp/x"])
    a = mod.parse_args()
    assert a.init is None and a.league is None and a.temp_schedule is None
