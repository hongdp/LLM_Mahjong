"""exp115 phase 2: does the persistent genbutsu bias help against bc49? (Python engine, all T=0, duplicate 2v2.)

Per wall: 2 seat orientations (RL on seats {0,2} or {1,3}) x 2 RL variants (plain / biased) = 4 games; bc49 holds the
other two seats in every game. Per wall and variant: A_pts = sum over both orientations of RL seats' points minus
bc49 seats' points; share = 1 / 0.5 / 0 for A_pts > / = / < 0 (same per-wall share as pairdiff_eval). The two
variants share walls and orientations, so (biased - plain) is paired per wall.
Bias: +f on genbutsu discard slots (discard + discard0; fold_option.genbutsu_34 = VecEnv.safe_info rule) when an
opponent is in riichi and the RL seat is not, by own shanten (f0 tenpai / f1 one-shanten / f2 two+).
usage: posture_bias_vs_bc.py --bc bc49.pt --rl rl.pt --f1 16 --f2 16 [--f0 0] --walls N --seed0 S --out part.json
       posture_bias_vs_bc.py --report part*.json
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.policy_diff_atlas import _DummyStep  # noqa: E402


def collect(a):
    from src.agents.dnn.selfplay import play_game_gen, _choose, encode_state
    from src.agents.dnn.action_space import get_space
    from src.agents.dnn.fold_option import genbutsu_34
    from src.agents.dnn.encoder import TYPE_TO_ID, TILE_TYPES
    from scripts.run_elo_league import load_dnn
    dev = "cuda"
    bc = load_dnn(a.bc, dev).eval(); rl = load_dnn(a.rl, dev).eval()
    space = get_space(rl); variant = getattr(rl, "encoder_variant", "v3r")
    D, D0 = TYPE_TO_ID["discard"] * TILE_TYPES, TYPE_TO_ID["discard0"] * TILE_TYPES

    def rl_choose(table, pid, actions, biased):
        planes, scalars = encode_state(table, pid, variant=variant)
        mask, lookup = space.mask(actions, mode=None, table=table, pid=pid)
        with torch.no_grad():
            lo = rl(planes[None].to(dev), scalars[None].to(dev), mask[None].to(dev))[0].float()
        if biased and not table.riichi[pid]:
            g = genbutsu_34(table, pid)
            if g is not None:
                sh = int(table._shanten(table.hands[pid], len(table.melds[pid])))
                f = a.f0 if sh <= 0 else (a.f1 if sh == 1 else a.f2)
                if f:
                    gb = torch.from_numpy(g.astype(np.float32)).to(dev) * f
                    lo[D:D + 34] += gb; lo[D0:D0 + 34] += gb
        i = int(lo.argmax())
        assert space.follow_up(i, actions, mode=None, table=table, pid=pid) is None
        return space.resolve(i, lookup)

    def bc_choose(table, pid, actions):
        with torch.no_grad():
            _, s = _choose(bc, table, pid, actions, 0.0, dev)
        return s

    def play(wall, rl_seats, biased):
        gen = play_game_gen(deal_seed=wall, randomize_round=True)
        try:
            t, reqs = next(gen)
            while True:
                rep = [rl_choose(t, p, acts, biased) if p in rl_seats else bc_choose(t, p, acts) for p, acts in reqs]
                t, reqs = gen.send([(_DummyStep(), r) for r in rep])
        except StopIteration as stop:
            g = stop.value
            return sum(g.points[p] for p in rl_seats) - sum(g.points[p] for p in range(4) if p not in rl_seats)

    rows = []
    for wi in range(a.walls):
        wall = a.seed0 + wi
        r = {"wall": wall}
        for tag, biased in (("plain", False), ("bias", True)):
            r[tag] = play(wall, (0, 2), biased) + play(wall, (1, 3), biased)
        rows.append(r)
        if (wi + 1) % 100 == 0:
            print(f"{wi + 1}/{a.walls} walls", flush=True)
    json.dump({"bc": a.bc, "rl": a.rl, "f": [a.f0, a.f1, a.f2], "rows": rows}, open(a.out, "w")); print("->", a.out)


def report(paths):
    rows = []; meta = None
    for p in paths:
        d = json.load(open(p)); rows += d["rows"]; meta = d
    sh = lambda x: 1.0 if x > 0 else (0.5 if x == 0 else 0.0)
    sp = np.array([sh(r["plain"]) for r in rows]); sb = np.array([sh(r["bias"]) for r in rows])
    pp = np.array([r["plain"] for r in rows], float); pb = np.array([r["bias"] for r in rows], float)
    n = len(rows); dsh = sb - sp; dpt = pb - pp
    print(f"RL = {meta['rl']}\nbias f0/f1/f2 = {meta['f']}   walls = {n}")
    print(f"plain  share {sp.mean():.4f} ± {sp.std(ddof=1)/np.sqrt(n):.4f}   pts/wall {pp.mean():+.0f}")
    print(f"biased share {sb.mean():.4f} ± {sb.std(ddof=1)/np.sqrt(n):.4f}   pts/wall {pb.mean():+.0f}")
    print(f"paired Δshare {dsh.mean()*100:+.2f} ± {dsh.std(ddof=1)/np.sqrt(n)*100:.2f} pp   Δpts/wall {dpt.mean():+.0f} ± {dpt.std(ddof=1)/np.sqrt(n):.0f}   "
          f"walls changed {np.mean(dpt != 0):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc"); ap.add_argument("--rl")
    ap.add_argument("--f0", type=float, default=0.0); ap.add_argument("--f1", type=float, default=8.0); ap.add_argument("--f2", type=float, default=8.0)
    ap.add_argument("--walls", type=int, default=500); ap.add_argument("--seed0", type=int, default=102224000)
    ap.add_argument("--out"); ap.add_argument("--report", nargs="*")
    a = ap.parse_args()
    report(a.report) if a.report else collect(a)


if __name__ == "__main__":
    main()
