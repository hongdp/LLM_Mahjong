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


def test_solve_temperature_hits_target_entropy():
    import torch
    torch.manual_seed(0)
    n, a = 256, 12
    mask = torch.rand(n, a) < 0.6
    mask[:, 0] = True
    q = (torch.randn(n, a) * 0.002).masked_fill(~mask, float("-inf"))   # exp71 r1 gap scale
    for frac in (0.9, 0.5, 0.2):
        T = mod.solve_temperature(q, mask, frac)
        p = torch.softmax(q / T, 1)
        rows = mask.sum(1) > 1
        H = float(-(p * torch.log(p.clamp_min(1e-12))).sum(1)[rows].mean())
        target = frac * float(torch.log(mask.sum(1)[rows].float()).mean())
        assert abs(H - target) < 0.02, (frac, T, H, target)
    assert mod.solve_temperature(q, mask, 0.2) < mod.solve_temperature(q, mask, 0.9) < 0.01
