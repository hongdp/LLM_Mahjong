"""L6b: one 171M seat (optionally fold-wrapped) vs three bc49 seats, T=0, hanchan, batch-1. Seed w, rotation rot:
the 171M seat is seat rot, walls from play_hanchan(seed0 + w) for every rotation (== vector table4: entity i at seat
(i + rot) % 4, wall seed = seed >> 2). Records per (w, rot): 171M seat uma, placement, firing stats, and per firing
whether the 2nd-highest score belonged to a bc49 seat (always true in 1v3; sanity check).
usage: fold1v3_driver.py VARIANT(none|G|GS) X W0 W1 SEED0 OUT.json"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/home/hongdp/Workspace/LLM_Mahjong/.claude/worktrees/rl-model-defense-learning-8fb5c0/experiments/placement_fold_20261106_200118")
from fold_policy import fold_policy  # noqa: E402
from scripts.arena_mortal_mjai import net_policy  # noqa: E402
from scripts.run_arena_dnn import load_dnn  # noqa: E402
from src.tasks.mahjong.hanchan import play_hanchan  # noqa: E402

E = "/home/hongdp/Workspace/LLM_Mahjong/experiments"
A_CK = f"{E}/probes/snap_exp124_A_games_171212800.pt"
BC49 = f"{E}/_anchors_epoch6/bc49.pt"


def main():
    variant, x, w0, w1, seed0, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), \
        int(sys.argv[5]), sys.argv[6]
    base = net_policy(load_dnn(A_CK, "cuda"), "cuda", temperature=0.0)
    bc = net_policy(load_dnn(BC49, "cuda"), "cuda", temperature=0.0)
    rows, t0, tot = [], time.time(), Counter()
    for w in range(w0, w1):
        for rot in range(4):
            st = Counter()
            if variant == "none":
                a_pol = base
            else:
                inner = fold_policy(base, x, variant, st)

                def a_pol(table, pid, actions, _inner=inner, _st=st):
                    before = _st["triggered"]
                    r = _inner(table, pid, actions)
                    if _st["triggered"] > before:
                        others = sorted((p, q) for q, p in enumerate(table.points) if q != pid)
                        _st["second_is_bc49"] += 1 if others[-1][1] != pid else 0
                    return r
            pols = {s: (a_pol if s == rot else bc) for s in range(4)}
            res = play_hanchan(pols, seed0 + w)
            rows.append({"wall": seed0 + w, "d": w, "rot": rot, "seat": rot, "uma": res.uma_points[rot],
                         "uma_all": list(res.uma_points), "placement": res.placements[rot], "stats": dict(st)})
            tot.update(st)
    json.dump({"variant": variant, "x": x, "seed0": seed0, "w0": w0, "w1": w1, "rows": rows, "stats": dict(tot),
               "min": (time.time() - t0) / 60}, open(out, "w"))
    print(f"1v3 {variant} x={x} {w0}-{w1}: {len(rows)} hanchan {(time.time() - t0) / 60:.1f} min stats {dict(tot)}",
          flush=True)


if __name__ == "__main__":
    main()
