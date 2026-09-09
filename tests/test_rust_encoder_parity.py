"""P4a: Rust v1r encoder / legal_mask == Python encoder on states reached by shared random play."""
import random

import numpy as np
import pytest
import torch

riichi_rs = pytest.importorskip("riichi_rs")
from src.agents.dnn.encoder import encode_state, legal_mask
from src.agents.dnn.selfplay import potential
from src.tasks.mahjong.claims import _resolve_claims
from src.tasks.mahjong.table import PyMahjongTable, ACTION_RE


def _check_obs(py, rs, pid, seed, step):
    P, S = encode_state(py, pid, variant="v1r")
    rp, rsc = rs.encode(pid)
    assert np.array_equal(P.numpy().reshape(-1), np.asarray(rp, dtype=np.float32)), (seed, step, pid)
    assert np.array_equal(S.numpy(), np.asarray(rsc, dtype=np.float32)), (seed, step, pid, S.numpy(), rsc)
    assert abs(potential(py, pid) - rs.potential(pid)) < 1e-9, (seed, step, pid)


def _check_mask(actions):
    m, lk = legal_mask(actions)
    rm, rlk = riichi_rs.legal_mask(actions)
    assert m.numpy().tolist() == rm and lk == rlk, actions


def test_encoder_parity_on_random_games():
    n_states = 0
    for seed in range(300):
        random.seed(seed)
        py = PyMahjongTable(randomize_round=True); py.text_obs = False
        rs = riichi_rs.Table(seed, True)
        rng = random.Random(seed + 99)
        guard = 0
        while not py.finished and guard < 600:
            guard += 1
            pid = py.turn
            acts = py.get_legal_actions(pid)
            if not acts:
                break
            _check_obs(py, rs, pid, seed, guard); _check_mask(acts); n_states += 1
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
                _check_obs(py, rs, other, seed, guard); _check_mask(opts); n_states += 1
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
    assert n_states > 20000
