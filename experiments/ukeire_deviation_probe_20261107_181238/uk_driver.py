"""Driver: A = exp124 A 192M (optionally wrapped by uk_policy), B = bc49, T=0, 2v2 duplicate single deals, batch-1,
same walls / deal generator as L6 (random.seed(wall) -> PyMahjongTable(randomize_round=True), validated in L1/L6).
usage: uk_driver.py VARIANT(none|UK<min_sh>) W0 W1 SEED0 OUT.json      e.g. UK2 = force ukeire-max at shanten >= 2"""
import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uk_policy import uk_policy  # noqa: E402
from scripts.arena_mortal_mjai import net_policy  # noqa: E402
from scripts.run_arena_dnn import load_dnn  # noqa: E402
from src.agents.dnn.mjai_export import play_game_mjai  # noqa: E402
from src.tasks.mahjong.table import PyMahjongTable  # noqa: E402

E = "/home/hongdp/Workspace/LLM_Mahjong/experiments"
A_CK = f"{E}/exp124_A_p2/games_192000000.pt"
BC49 = f"{E}/_anchors_epoch6/bc49.pt"


def main():
    variant, w0, w1, seed0, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    base = net_policy(load_dnn(A_CK, "cuda"), "cuda", temperature=0.0)
    bc = net_policy(load_dnn(BC49, "cuda"), "cuda", temperature=0.0)
    stats = Counter()
    a_pol = base if variant == "none" else uk_policy(base, int(variant[2:]), stats)
    rows, t0 = [], time.time()
    for d in range(w0, w1):
        w = seed0 + d
        diff = 0.0
        for orient in (0, 1):
            a_seats = (0, 2) if orient == 0 else (1, 3)
            pols = {s: (a_pol if s in a_seats else bc) for s in range(4)}
            random.seed(w)
            table = PyMahjongTable(randomize_round=True)
            table.text_obs = False
            play_game_mjai(table, pols, observer=None, sink=lambda ev: None)
            pts = table.points
            diff += sum(pts[s] for s in a_seats) - sum(pts[s] for s in range(4) if s not in a_seats)
        rows.append({"wall": w, "d": d, "diff": diff})
    json.dump({"variant": variant, "seed0": seed0, "w0": w0, "w1": w1, "rows": rows, "stats": dict(stats),
               "min": (time.time() - t0) / 60}, open(out, "w"))
    print(f"{variant} {w0}-{w1}: {(time.time() - t0) / 60:.1f} min changed "
          f"{sum(v for k, v in stats.items() if k.startswith('changed'))}", flush=True)


if __name__ == "__main__":
    main()
