"""exp112: hanchan head-to-head vs bc49 (exp83/84/91 protocol): 2v2 duplicate FULL MATCHES scored by uma,
T=0 both sides, `walls` duplicate walls (each played twice with seats swapped), seed0 59300000.
share = P(A's uma sum beats B's) per wall; also mean uma margin. Writes probes/hanchan_<run>_<ckpt>_<seed>.json.
usage: hanchan_eval.py <ckpt> <run_name> [walls=1200] [seed0=59300000] [workers=8]
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_elo_league import play_pair_vector


def main():
    ck, run = sys.argv[1], sys.argv[2]
    walls = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
    seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 59_300_000
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    bc49 = "experiments/_anchors_epoch6/bc49.pt"
    t0 = time.time()
    scores, diffs, rows = play_pair_vector(ck, bc49, walls, seed0, workers, "cuda", temp_a=0.0, temp_b=0.0, hanchan=True)
    s, d = np.asarray(scores), np.asarray(diffs)
    pl = np.asarray([r["placements"][p] for r in rows for p in r["a_seats"]])          # A's placements over all its seats
    out = {"run": run, "ckpt": ck, "walls": walls, "seed0": seed0, "share": float(s.mean()), "share_se": float(s.std(ddof=1) / np.sqrt(len(s))),
           "uma_margin_per_wall": float(d.mean()), "uma_margin_se": float(d.std(ddof=1) / np.sqrt(len(d))),
           "a_placement_dist": [float((pl == k).mean()) for k in (1, 2, 3, 4)], "a_mean_placement": float(pl.mean()),
           "busted_frac": float(np.mean([r["busted"] for r in rows])), "mean_deals": float(np.mean([r["n_deals"] for r in rows])),
           "minutes": round((time.time() - t0) / 60, 1)}
    os.makedirs("/home/hongdp/Workspace/LLM_Mahjong/experiments/probes", exist_ok=True)
    f = f"/home/hongdp/Workspace/LLM_Mahjong/experiments/probes/hanchan_{run}_{os.path.basename(ck)}_{seed0}.json"
    json.dump({**out, "share_per_wall": s.tolist(), "uma_per_wall": d.tolist()}, open(f, "w"))
    print(json.dumps(out)); print("->", f)


if __name__ == "__main__":
    main()
