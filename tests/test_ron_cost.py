"""exp109: `Table.ron_costs(seat)` must agree with what the engine actually does when that tile is discarded:
a positive cost means some opponent can ron right now and the discarder's point loss equals the cost; zero for
every tile means no opponent can ron whatever is discarded. Also: VecEnv.ron_cost_info / collect_rust plumbing."""
import random

import numpy as np
import pytest

riichi_rs = pytest.importorskip("riichi_rs")
from src.agents.dnn.encoder import TILE_TYPES, TYPE_TO_ID  # noqa: E402

RS = 1000.0  # 1 / REWARD_SCALE


def test_ron_costs_match_engine_ron_offers():
    """Policy-driven play on the Rust PyTable (random play never reaches tenpai): for every discard the policy
    makes, `ron_costs[tile] > 0` must hold exactly when some opponent is offered a ron on it, and the cost must
    equal the sum of those opponents' `win_result(...)["main"]` (what settle_ron charges the discarder)."""
    import os, torch
    from src.agents.dnn.encoder import encode_state, legal_mask, N_PLANES_V3R  # noqa: F401
    from scripts.run_elo_league import load_dnn
    ck = "/home/hongdp/Workspace/LLM_Mahjong/experiments/exp88_Q1x/games_final.pt"
    if not os.path.exists(ck) or not torch.cuda.is_available():
        pytest.skip("needs a trained cnn_m_v3r checkpoint + CUDA")
    net = load_dnn(ck, "cuda").eval()
    n_pos, n_neg = 0, 0
    with torch.no_grad():
        for seed in range(160):
            t = riichi_rs.Table(seed, True); rng = random.Random(seed)
            for _ in range(400):
                if t.finished:
                    break
                pid = t.turn
                acts = t.get_legal_actions(pid)
                if not acts:
                    break
                costs = np.asarray(t.ron_costs(pid), dtype=np.float32)
                pl, sc = t.encode(pid, "v3r")
                P = torch.tensor(pl, device="cuda").view(1, -1, TILE_TYPES); S = torch.tensor(sc, device="cuda").view(1, -1)
                m, lookup = legal_mask(acts)
                lg = net(P, S, m.view(1, -1).cuda())
                a = lookup[int(torch.multinomial(torch.softmax(lg[0], 0), 1))]
                _, done, info = t.step(pid, a)
                if 'type="discard"' in a:
                    tile = a.split('tile="')[1].split('"')[0]
                    t34 = _t34(tile) + (TILE_TYPES if tile[0] == "0" else 0)     # red five -> discard0 slot block
                    offered = [o for o in range(4) if o != pid and any("ron" in x for x in t.get_interrupt_actions(o))]
                    if costs[t34] > 0:
                        assert offered, (seed, tile, costs[t34])
                        exp = sum(t.win_result(o, tile, False, False)["main"] for o in offered)
                        assert abs(exp - costs[t34] * RS) <= 1.0, (seed, tile, exp, costs[t34] * RS)
                        n_pos += 1
                    else:
                        assert not offered, (seed, tile, "cost 0 but ron offered")
                        n_neg += 1
                if done:
                    break
                if info.get("discarded") or info.get("chankan"):
                    cands = []
                    for off in range(1, 4):
                        other = (pid + off) % 4
                        opts = t.get_interrupt_actions(other)
                        if len(opts) > 1:
                            ron = [x for x in opts if "ron" in x]
                            cands.append((other, ron[0] if ron else opts[rng.randrange(len(opts))]))
                    executed, done, _ = t.resolve_claims(cands)      # always: an empty claim list still advances the deal
                    if done:
                        break
                    if not executed:
                        if t.pending_kan:
                            t.resolve_pending_kan()
                        elif t.advance_turn():
                            break
    assert n_pos >= 30 and n_neg >= 1000, (n_pos, n_neg)


def _t34(tile):
    v, s = int(tile[0]) or 5, tile[1]
    return 27 + v - 1 if s == "z" else "mps".index(s) * 9 + v - 1


def test_vecenv_ron_cost_plumbing():
    import torch
    from src.agents.dnn.rust_rollout import collect_rust
    from src.agents.dnn.arch_zoo import ZOO
    env = riichi_rs.VecEnv([21, 22, 23, 24], 4, variant="v3r")
    planes, scalars, mask, seats, gids = env.observe()
    rc = np.asarray(env.ron_cost_info())
    assert rc.shape == (planes.shape[0], 2 * TILE_TYPES) and (rc >= 0).all()
    if not torch.cuda.is_available():
        pytest.skip("collect_rust needs CUDA here")
    net = ZOO["cnn_m_v3r"][0]().cuda().eval()
    cfg = {"temperature": 1.0, "gamma": 0.995, "games_per_worker": 8, "encoder_variant": "v3r", "ron_cost": True}
    eps, _ = collect_rust(net, 16, cfg, 1, seeds=[7_200_000 + i for i in range(16)], device="cuda")
    for e in eps:
        assert e["ron_cost"].shape == (len(e["actions"]), 2 * TILE_TYPES)
        # interrupt rows (no discard legal) carry zeros
        disc_legal = e["mask"].reshape(len(e["actions"]), -1, TILE_TYPES)[:, [TYPE_TO_ID["discard"], TYPE_TO_ID["discard0"]], :].any((1, 2))
        assert not e["ron_cost"][~disc_legal].any()
