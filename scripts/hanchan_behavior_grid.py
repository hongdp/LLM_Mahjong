"""exp112: behaviour grid from Rust mirror-hanchan deal facts — per (round wind, seat's rank at deal start):
riichi rate, deal-in (houjuu) rate, win rate. The signature of placement-aware play is a LOWER deal-in rate when
leading in South than when leading in East (exp91: P4 went the other way, 0.112 -> 0.156).
usage: hanchan_behavior_grid.py <ckpt> <run_name> [matches=2000]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rank_order(points, start_dealer=0):
    """seats sorted best-first: higher points first, ties by seat order from the start dealer (engine rule)."""
    return sorted(range(4), key=lambda s: (-points[s], (s - start_dealer) % 4))


def main():
    from src.agents.dnn.rust_rollout import collect_rust
    from scripts.run_elo_league import load_dnn
    ck, run = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    net = load_dnn(ck, "cuda").eval()
    cfg = {"temperature": 0.0, "gamma": 0.995, "games_per_worker": 256, "encoder_variant": getattr(net, "encoder_variant", "v3r"),
           "hanchan": True, "hanchan_pure": True, "hanchan_credit": "none", "shaping": False}
    collect_rust(net, n, cfg, 1, [59_400_000 + i for i in range(n)], device="cuda")
    cells = {}
    for g in collect_rust.last_games:
        for d in g["deals"]:
            order = rank_order(d["start_points"], 0)
            for seat in range(4):
                rank = order.index(seat) + 1
                key = (("E", "S", "W")[min(d["rw"], 2)], "top2" if rank <= 2 else "bot2")
                c = cells.setdefault(key, {"n": 0, "riichi": 0, "houjuu": 0, "win": 0})
                c["n"] += 1; c["riichi"] += int(d["riichi"][seat]); c["houjuu"] += int(d["houjuu"] == seat); c["win"] += int(seat in d["winners"])
    out = {f"{w}/{r}": {"n": c["n"], "riichi": round(c["riichi"] / c["n"], 4), "houjuu": round(c["houjuu"] / c["n"], 4), "win": round(c["win"] / c["n"], 4)}
           for (w, r), c in sorted(cells.items())}
    f = f"/home/hongdp/Workspace/LLM_Mahjong/experiments/probes/hanchan_grid_{run}_{os.path.basename(ck)}.json"
    json.dump(out, open(f, "w"), indent=1); print(json.dumps(out, indent=1)); print("->", f)


if __name__ == "__main__":
    main()
