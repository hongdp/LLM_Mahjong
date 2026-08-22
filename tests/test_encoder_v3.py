"""encoder v3: complete public record — shape, determinism, fact consistency."""
import random
import sys

import torch

sys.path.insert(0, ".")
from src.agents.dnn.arch_zoo import ZOO                       # noqa: E402
from src.agents.dnn.encoder import (N_PLANES_V3, N_SCALARS_V3,   # noqa: E402
                                    encode_state, legal_mask, tile_to_34)
from src.agents.dnn.selfplay import play_game                  # noqa: E402
from src.tasks.mahjong.table import PyMahjongTable              # noqa: E402


def _midgame_table(seed=11, steps=40):
    random.seed(seed)
    t = PyMahjongTable(randomize_round=True)
    t.text_obs = False
    for _ in range(steps):
        if t.finished:
            break
        pid = t.turn
        acts = t.get_legal_actions(pid)
        if not acts:
            break
        a = next((x for x in acts if "discard" in x), acts[0])
        _, _, done, info = t.step(pid, a)
        if done:
            break
        if info.get("discarded"):
            if t.pending_kan:
                t.resolve_pending_kan()
            else:
                t.advance_turn()
    return t


def test_v3_shapes_and_determinism():
    t = _midgame_table()
    p1, s1 = encode_state(t, t.turn, variant="v3")
    p2, s2 = encode_state(t, t.turn, variant="v3")
    assert p1.shape == (N_PLANES_V3, 34) and s1.shape == (N_SCALARS_V3,)
    assert torch.equal(p1, p2) and torch.equal(s1, s2)


def test_v3_river_facts_match_engine():
    t = _midgame_table()
    pid = t.turn
    p, _ = encode_state(t, pid, variant="v3")
    for off in range(4):
        seat = (pid + off) % 4
        ev = t.river_events[seat]
        base = 15 + 4 * off
        # every recorded discard appears in the order plane with rank order
        for j, (tile, tsumogiri, rdecl, called, _i) in enumerate(ev):
            idx = tile_to_34(tile)
            assert p[base][idx] > 0
            if tsumogiri:
                assert p[base + 1][idx] == 1.0
        # v1 river plane (visible river) agrees with non-called events
        visible = {tile_to_34(x.replace("*", "")) for x in t.discards[seat]}
        assert all(p[8 + off][i] == 1.0 for i in visible)


def test_v3_nets_play_games():
    for name in ("cnn_m_v3", "convformer_m_v3"):
        net = ZOO[name][0]().eval()
        assert net.encoder_variant == "v3"
        g = play_game(net, deal_seed=5)
        assert sum(len(v) for v in g.trajectories.values()) > 20
        st = g.trajectories[0][0]
        assert st.planes.shape[0] == N_PLANES_V3 and st.scalars.shape[0] == N_SCALARS_V3


def test_v1_path_unchanged_signature():
    t = _midgame_table()
    p, s = encode_state(t, t.turn)
    assert p.shape == (15, 34) and s.shape == (20,)
