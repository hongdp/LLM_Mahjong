"""Yaku census of training-distribution self-play: one net on all four seats, the trainer's own Rust rollout
collector (collect_rust, temperature T, no shaping), every win's yaku counted by style_stats.add_game.
Writes OUT after every model, so a killed run keeps its finished models (rerun skips them).
usage (repo root): yaku_census.py OUT.json GAMES name=path[@T] ...      (T defaults to 1.0)"""
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
from src.agents.dnn import style_stats as ss  # noqa: E402
from src.agents.dnn.checkpoint import load_dnn  # noqa: E402
from src.agents.dnn.rust_rollout import collect_rust  # noqa: E402

CHUNK = 2048
SEED0 = 50_000_000


def census(path, temperature, games):
    net = load_dnn(path, "cuda")
    cfg = dict(temperature=temperature, gamma=0.995, games_per_worker=256, shaping=False, bf16_rollout=False,
               rollout_values=False, encoder_variant=getattr(net, "encoder_variant", "v3r"))
    agg, t0 = ss.new_agg(), time.time()
    for c0 in range(0, games, CHUNK):
        seeds = [SEED0 + i for i in range(c0, min(c0 + CHUNK, games))]
        collect_rust(net, len(seeds), cfg, 1, seeds, device="cuda")
        for g in collect_rust.last_games:
            ss.add_game(agg, g["result"], g.get("riichi"), g.get("n_melds"), g.get("n_discards"),
                        points=g.get("points"), start_points=g.get("start_points"))
        collect_rust.last_games = None
    return agg, time.time() - t0


def main():
    out, games = sys.argv[1], int(sys.argv[2])
    res = json.load(open(out)) if os.path.exists(out) else {}
    for spec in sys.argv[3:]:
        name, rest = spec.split("=", 1)
        path, _, t = rest.partition("@")
        if name in res:
            continue
        agg, sec = census(path, float(t or 1.0), games)
        res[name] = {"path": path, "temperature": float(t or 1.0), "games": agg["games"], "sec": round(sec, 1),
                     "summary": ss.summarize(agg), "counts": dict(sorted(ss.yaku_counts(agg).items()))}
        json.dump(res, open(out + ".tmp", "w"), indent=1)
        os.replace(out + ".tmp", out)
        s = res[name]["summary"]
        print(f"{name}: {agg['games']} deals {sec / 60:.1f} min agari {s['agari_rate']:.4f} "
              f"yakuman {res[name]['counts']['_yakuman']} kazoe {res[name]['counts']['_kazoe']}", flush=True)


if __name__ == "__main__":
    main()
