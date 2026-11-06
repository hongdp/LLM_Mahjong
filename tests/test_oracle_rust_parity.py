"""exp108: Rust `Table.oracle(pid)` == Python `oracle_features(table, pid)` on states reached by
shared random play (same harness as test_rust_encoder_parity), and the separate critic's shapes /
blind control / all-action advantage identities."""
import random

import numpy as np
import pytest
import torch

riichi_rs = pytest.importorskip("riichi_rs")
from src.agents.dnn.oracle_features import ORACLE_DIM, oracle_features  # noqa: E402
from src.agents.dnn.oracle_critic import SeparateCritic, q_advantage      # noqa: E402
from src.agents.dnn.encoder import ACTION_DIM, N_PLANES_V3R, N_SCALARS_V3  # noqa: E402
from src.tasks.mahjong.claims import _resolve_claims                      # noqa: E402
from src.tasks.mahjong.table import PyMahjongTable, ACTION_RE             # noqa: E402


def _check(py, rs, seed, step):
    for pid in range(4):
        a = oracle_features(py, pid).numpy()
        b = np.asarray(rs.oracle(pid), dtype=np.float32)
        assert a.shape == (ORACLE_DIM,) and b.shape == (ORACLE_DIM,)
        if not np.array_equal(a, b):
            bad = np.nonzero(a != b)[0]
            raise AssertionError(f"seed {seed} step {step} pid {pid}: slots {bad[:10].tolist()} py={a[bad[:10]]} rs={b[bad[:10]]}")


def test_oracle_vector_parity_on_random_games():
    n_states, n_waits = 0, 0
    for seed in range(120):
        random.seed(seed)
        py = PyMahjongTable(randomize_round=True); py.text_obs = False
        rs = riichi_rs.Table(seed, True)
        rng = random.Random(seed + 7)
        guard = 0
        while not py.finished and guard < 600:
            guard += 1
            pid = py.turn
            acts = py.get_legal_actions(pid)
            if not acts:
                break
            _check(py, rs, seed, guard); n_states += 1
            n_waits += int(oracle_features(py, pid)[111:213].sum() > 0)
            a = acts[rng.randrange(len(acts))]
            _, _, done, info = py.step(pid, a); rs.step(pid, a)
            if done:
                break
            if not (info.get("discarded") or info.get("chankan")):
                continue
            cands = []
            for off in range(1, 4):
                other = (pid + off) % 4
                opts = py.get_interrupt_actions(other)
                if len(opts) == 1:
                    continue
                _check(py, rs, seed, guard); n_states += 1
                chosen = opts[rng.randrange(len(opts))]
                m = ACTION_RE.search(chosen)
                cands.append({"player_id": other, "parsed": chosen, "type": m.group(1), "reward": 0.0})
            executed, done = _resolve_claims(py, cands); rs.resolve_claims([(c["player_id"], c["parsed"]) for c in cands])
            if done:
                break
            if not executed:
                if py.pending_kan:
                    py.resolve_pending_kan(); rs.resolve_pending_kan()
                else:
                    _, rd = py.advance_turn(); rs.advance_turn()
                    if rd:
                        break
    assert n_states > 3000 and n_waits > 50, (n_states, n_waits)


def test_vecenv_oracle_info_matches_rows():
    env = riichi_rs.VecEnv([3, 4, 5, 6], 4, variant="v3r")
    planes, scalars, mask, seats, gids = env.observe()
    o = np.asarray(env.oracle_info())
    assert o.shape == (planes.shape[0], ORACLE_DIM)
    # hidden hands of the three opponents: 13 or 14 tiles each -> counts/4 sum in [3*13/4, 3*14/4]
    hs = o[:, :102].sum(1) * 4
    assert np.all((hs >= 39 - 1e-4) & (hs <= 42 + 1e-4)), hs
    assert np.all(o[:, 250] <= 1.0) and np.all(o[:, 250] > 0.5)      # fresh deals: ~70 live tiles
    assert np.all(o[:, 285:319].sum(1) == 1.0)                       # ura indicator always present


def test_separate_critic_shapes_blind_and_q_advantage():
    B = 6
    P = torch.randn(B, N_PLANES_V3R, 34); S = torch.randn(B, N_SCALARS_V3); C = torch.rand(B, ORACLE_DIM)
    v = SeparateCritic("v", channels=16, blocks=1)(P, S, C)
    q = SeparateCritic("q", channels=16, blocks=1)(P, S, C)
    assert v.shape == (B,) and q.shape == (B, ACTION_DIM)
    bl = SeparateCritic("q", channels=16, blocks=1, blind=True)
    assert torch.equal(bl(P, S, C), bl(P, S, torch.zeros_like(C)))   # blind control ignores the oracle input
    mask = torch.zeros(B, ACTION_DIM, dtype=torch.bool); mask[:, :5] = True
    logits = torch.randn(B, ACTION_DIM); acts = torch.randint(0, 5, (B,))
    adv, vpi = q_advantage(torch.full((B, ACTION_DIM), 3.0), logits, mask, acts)
    assert torch.allclose(adv, torch.zeros(B), atol=1e-6) and torch.allclose(vpi, torch.full((B,), 3.0))
    qq = torch.randn(B, ACTION_DIM)
    adv, vpi = q_advantage(qq, logits, mask, acts)
    pi = torch.softmax(logits.masked_fill(~mask, float("-inf")), 1)
    assert torch.allclose((pi * adv.new_tensor(0) + pi * (qq.masked_fill(~mask, 0) - vpi[:, None])).sum(1),
                          torch.zeros(B), atol=1e-5)                 # sum_a pi(a) A(s,a) == 0
