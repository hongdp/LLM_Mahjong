"""P3: full-game parity. Both engines are driven by the same scripted random policy
(same seed -> same deal; the chooser picks the same legal-action index on both) and must
agree step by step: legal/interrupt actions (strings + order), rewards, done/info, visible
state, and the final result string / points / final rewards."""
import random

import pytest

riichi_rs = pytest.importorskip("riichi_rs")
from src.tasks.mahjong.claims import _resolve_claims
from src.tasks.mahjong.table import PyMahjongTable, ACTION_RE


def _snap_py(t):
    return dict(hands=[list(t.hands[i]) for i in range(4)], red=[[t.red[i]["m"], t.red[i]["p"], t.red[i]["s"]] for i in range(4)],
                discards=[list(t.discards[i]) for i in range(4)], furiten=[list(t.furiten_river[i]) for i in range(4)],
                melds=[[(m["type"], list(m["tiles"]), m["opened"], m.get("red", 0)) for m in t.melds[i]] for i in range(4)],
                points=list(t.points), kyotaku=t.kyotaku, riichi=list(t.riichi), turn=t.turn,
                last_discard=t.last_discard, last_discarder=t.last_discarder, wall=len(t.wall),
                dora=list(t.dora_indicators), dc=list(t.discard_count), kan=t.kan_count, any_call=t.any_call,
                tf=list(t.temp_furiten), pf=list(t.perm_furiten), finished=t.finished)


def _snap_rs(r):
    return dict(hands=r.hands, red=r.red, discards=r.discards, furiten=r.furiten_river,
                melds=[[(m["type"], m["tiles"], m["opened"], m["red"]) for m in ms] for ms in r.melds],
                points=r.points, kyotaku=r.kyotaku, riichi=r.riichi, turn=r.turn,
                last_discard=r.last_discard, last_discarder=r.last_discarder, wall=r.wall_len,
                dora=r.dora_indicators, dc=r.discard_count, kan=r.kan_count, any_call=r.any_call,
                tf=r.temp_furiten, pf=r.perm_furiten, finished=r.finished)


def _play(seed, chooser_seed, bias):
    random.seed(seed)
    py = PyMahjongTable(randomize_round=True); py.text_obs = False
    rs = riichi_rs.Table(seed, True)
    rng = random.Random(chooser_seed)
    assert _snap_py(py) == _snap_rs(rs), ("deal", seed)

    def pick(actions):
        # bias toward "interesting" actions (calls, riichi, kan, tsumo, ron) so rare paths get exercised
        if bias and len(actions) > 1 and rng.random() < 0.7:
            special = [i for i, a in enumerate(actions) if 'type="discard"' not in a and 'type="skip"' not in a]
            if special:
                return rng.choice(special)
        return rng.randrange(len(actions))

    guard = 0
    while not py.finished and guard < 600:
        guard += 1
        pid = py.turn
        assert rs.turn == pid
        acts = py.get_legal_actions(pid)
        assert rs.get_legal_actions(pid) == acts, (seed, guard, acts, rs.get_legal_actions(pid))
        if not acts:
            break
        a = acts[pick(acts)]
        _, rew, done, info = py.step(pid, a)
        rrew, rdone, rinfo = rs.step(pid, a)
        assert [rew[i] for i in range(4)] == rrew and done == rdone, (seed, guard, a, rew, rrew, done, rdone)
        assert bool(info.get("discarded")) == rinfo["discarded"] and info.get("chankan") == rinfo.get("chankan"), (seed, guard, a, info, rinfo)
        assert _snap_py(py) == _snap_rs(rs), (seed, guard, a)
        if done:
            break
        if not (info.get("discarded") or info.get("chankan")):
            continue
        cands = []
        for off in range(1, 4):
            other = (pid + off) % 4
            opts = py.get_interrupt_actions(other)
            assert rs.get_interrupt_actions(other) == opts, (seed, guard, other, opts, rs.get_interrupt_actions(other))
            if len(opts) == 1:
                continue
            chosen = opts[pick(opts)]
            m = ACTION_RE.search(chosen)
            cands.append({"player_id": other, "parsed": chosen, "type": m.group(1) if m else None, "reward": 0.0})
        executed, done = _resolve_claims(py, cands)
        rexec, rdone, rrews = rs.resolve_claims([(c["player_id"], c["parsed"]) for c in cands])
        assert [c["player_id"] for c in executed] == rexec and done == rdone, (seed, guard, cands, executed, rexec)
        assert [c["reward"] for c in cands] == rrews, (seed, guard, cands, rrews)
        assert _snap_py(py) == _snap_rs(rs), (seed, guard, "claims", cands)
        if done:
            break
        if not executed:
            if py.pending_kan:
                assert rs.pending_kan is not None
                py.resolve_pending_kan(); rs.resolve_pending_kan()
            else:
                _, r_done = py.advance_turn()
                assert rs.advance_turn() == r_done, (seed, guard)
                if r_done:
                    break
        assert _snap_py(py) == _snap_rs(rs), (seed, guard, "advance")
    assert py.finished == rs.finished, seed
    assert py.result_summary == rs.result_summary, (seed, py.result_summary, rs.result_summary)
    assert list(py.points) == rs.points
    if not py.finished:                      # guard / no-legal-action exit: both unfinished, no rewards
        assert py.final_rewards is None and rs.final_rewards is None
        return "unfinished"
    assert py.final_rewards is not None and rs.final_rewards is not None
    for a, b in zip(py.final_rewards, rs.final_rewards):
        assert abs(a - b) < 1e-9, (seed, py.final_rewards, rs.final_rewards)
    return py.result_summary


def test_full_game_parity_random_policy():
    kinds = {}
    for seed in range(1500):
        s = _play(seed, seed * 7 + 1, bias=(seed % 2 == 0))
        k = s.split(" |")[0].split("(")[0][:6]
        kinds[k] = kinds.get(k, 0) + 1
    # make sure the distribution actually exercised wins, draws and aborts
    assert sum(kinds.values()) == 1500 and len(kinds) >= 3, kinds
    assert kinds.get("unfini", 0) < 30, kinds
