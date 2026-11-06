"""Four-entity hanchan arena over the vectorized worker (design D5).

Per wall, four rotations: entity i sits at seat (i + rot) % 4, so every
entity plays every seat once against the same wall sequence. One match
yields a full ranking, i.e. six pairwise contrasts instead of the one a
2v2 duplicate gives — and the rotation cancels seat/dealer luck the way
duplicate pairing cancels wall luck.

Writes ledger rows directly: the ledger is the primary record, and every
table exports the same shape whatever driver produced it.
"""

import time

import torch

from src.eval import ledger, registry
from src.eval.fingerprints import all_fingerprints

UMA_RANK = [15000, 5000, -5000, -15000]
DRIVER = "vec-table/1"


def _cfg(paths, temps, device, seed0, parallel):
    """Rollout config for one four-entity table: the shared vectorised-arena
    base (src/eval/vec_arena.base_cfg) plus the table seating keys."""
    from src.agents.dnn.checkpoint import load_dnn
    from src.eval.vec_arena import base_cfg
    net = load_dnn(paths[0], "cpu")
    blob = torch.load(paths[0], map_location="cpu", weights_only=False)
    cfg = base_cfg(net, blob, device, seed0)
    cfg.update(table=True, hanchan=True,
               table_temps=[float(t) for t in temps],
               league=[{"name": f"e{i+1}", "path": p} for i, p in enumerate(paths[1:])])
    return net, cfg


def play_table(eids, walls, seed0, device="cuda", parallel=20,
               rotations=(0, 1, 2, 3)):
    """`walls` walls x len(rotations) matches. Returns ledger rows."""
    from src.agents.dnn.parallel_rollout import collect_parallel
    if len(eids) != 4:
        raise ValueError("a table seats exactly four entities")
    ents = registry.entities()
    paths = [registry.path_of(e) for e in eids]
    temps = [ents[e]["temperature"] for e in eids]
    net, cfg = _cfg(paths, temps, device, seed0, parallel)
    seeds = [(seed0 + w) * 4 + r for w in range(walls) for r in rotations]
    t0 = time.time()
    collect_parallel(net, len(seeds), cfg, parallel, seeds)
    games = collect_parallel.last_games
    fp = all_fingerprints()
    rows = []
    for g in games:
        h = g["hanchan"]
        rot = g["seed"] & 3
        seats = [eids[(s - rot) % 4] for s in range(4)]
        st = [temps[(s - rot) % 4] for s in range(4)]
        uma = h["uma_points"]
        plc = h["placements"]
        pts = [uma[s] + 25000 - UMA_RANK[plc[s] - 1] for s in range(4)]
        rows.append(ledger.row(DRIVER, "hanchan", g["seed"], g["seed"] >> 2,
                               seats, st, pts, uma, plc, h["n_deals"],
                               h["busted"], fp, {"rot": rot}))
    return rows, round(len(games) / max(time.time() - t0, 1e-9) * 60, 1)
