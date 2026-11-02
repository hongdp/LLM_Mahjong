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
    for variant in ("v1r", "v3r", "v3rh", "v3rg"):
        P, S = encode_state(py, pid, variant=variant)
        rp, rsc = rs.encode(pid, variant)
        assert np.array_equal(P.numpy().reshape(-1), np.asarray(rp, dtype=np.float32)), (variant, seed, step, pid)
        # PLANE_Q grid: uint8 = rint(v*20) round-trips to the encoder's f32 values bit-exactly
        # (this is what lets VecEnv ship planes as u8 and the trainer widen with `.float() / 20`)
        q = np.rint(P.numpy().reshape(-1) * riichi_rs.PLANE_Q).astype(np.uint8)
        assert np.array_equal(torch.from_numpy(q).float().div_(float(riichi_rs.PLANE_Q)).numpy(),
                              P.numpy().reshape(-1)), (variant, seed, step, pid)
        assert np.array_equal(S.numpy(), np.asarray(rsc, dtype=np.float32)), (variant, seed, step, pid, S.numpy(), rsc)
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


def test_v3rg_parity_under_riichi():
    """exp116: random play almost never reaches riichi, so the v3rg genbutsu plane / shanten one-hot are exercised on
    policy-driven lockstep games (Python table drives, Rust table mirrors); every decision is compared bit-exactly and
    enough of them must carry a non-empty genbutsu plane."""
    import os
    from scripts.run_elo_league import load_dnn
    ck = "/home/hongdp/Workspace/LLM_Mahjong/experiments/exp107_P/games_256000000.pt"
    if not os.path.exists(ck) or not torch.cuda.is_available():
        pytest.skip("needs a trained cnn_l_v3r checkpoint + CUDA")
    net = load_dnn(ck, "cuda").eval()
    n_threat, n_states = 0, 0

    def pick(table, pid, acts, rng):
        P, S = encode_state(table, pid, variant="v3r")
        m, lk = legal_mask(acts)
        with torch.no_grad():
            lg = net(P[None].cuda(), S[None].cuda(), m[None].cuda())[0]
        return lk[int(torch.multinomial(torch.softmax(lg, 0), 1))]

    def check(py, rs, pid):
        nonlocal n_threat, n_states
        P, S = encode_state(py, pid, variant="v3rg")
        rp, rsc = rs.encode(pid, "v3rg")
        assert np.array_equal(P.numpy().reshape(-1), np.asarray(rp, dtype=np.float32))
        assert np.array_equal(S.numpy(), np.asarray(rsc, dtype=np.float32)), (S.numpy()[-3:], rsc[-3:])
        n_states += 1; n_threat += int(P.numpy()[-1].sum() > 0)

    for seed in range(120):
        random.seed(seed)
        py = PyMahjongTable(randomize_round=True); py.text_obs = False
        rs = riichi_rs.Table(seed, True)
        rng = random.Random(seed + 7)
        for _ in range(600):
            if py.finished:
                break
            pid = py.turn
            acts = py.get_legal_actions(pid)
            if not acts:
                break
            check(py, rs, pid)
            a = pick(py, pid, acts, rng)
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
                check(py, rs, other)
                chosen = pick(py, other, opts, rng)
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
    assert n_threat >= 200, (n_threat, n_states)
