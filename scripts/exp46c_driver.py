"""exp46-C driver: self-history league gens, self-contained on the VM.

User-designed ecology fix (2026-08-29, rev3): the learner never plays
its current self; opponents come only from its own training history
(self-improvement purity). Pool per chunk = init (bc49, permanent
floor) + best (gated champion-so-far) + newest gen + the 2 strongest
older gens by ROLLOUT RATINGS (the trainer scores learner-vs-pool
point-share each iteration into league_stats.jsonl; the learner is the
common opponent, so the share order is Elo-consistent for free) + 1
random leftover gen (diversity slot) — cap 6. End-to-end bench
2026-08-29: N=6 costs only ~11% throughput vs N=3 (80.1 vs 89.8
games/s local; the eager 3.4x window estimate is absorbed by CUDA
graphs + overlap). league_frac 1.0 kills pure mirror games; entropy
0.003 (no exploration tax on a sharp prior); learner-seat trajectories
only (trainer default).

Chunked resume implements the dynamic pool with zero trainer changes:
after each 100k-game chunk the snapshot joins the gen history, and
best := snapshot only if it beats the incumbent (100 duplicate deals,
share > 0.55 — the promotion gate that keeps the pool unpolluted).

Runs under run_dnn_cloud.sh via launch_g4_git.sh; --exp_dir is appended
by the launcher. Gate results land in <exp_dir>/gens.jsonl (synced to
GCS with everything else).
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FLAG_GS = ("gs://llm-mahjong-experiments/checkpoints/human_lineage/"
           "bc_convformer_m_v3r_m46_best.pt")
CHUNK = 100_000
N_CHUNKS = 10
GATE_DEALS = 100
GATE_SHARE = 0.55
POOL_TOP = 2             # strongest older gens by rollout rating
POOL_CAP_NOTE = 6        # init + best + newest + top-2 strongest + 1 random


def rollout_ratings(exp, since_line):
    """Aggregate league_stats.jsonl rows written after `since_line` into
    {name: (learner_share, n)}. Written by the trainer once per iteration;
    the learner is every pool member's common opponent, so ordering by
    learner_share IS an Elo-consistent strength order (low = strong)."""
    path = os.path.join(exp, "league_stats.jsonl")
    if not os.path.exists(path):
        return {}
    agg = {}
    for ln in open(path).readlines()[since_line:]:
        try:
            rec = json.loads(ln)
        except ValueError:
            continue
        for k, v in rec.get("vs", {}).items():
            w, n = agg.get(k, (0.0, 0))
            agg[k] = (w + v["learner_share"] * v["n"], n + v["n"])
    return {k: (round(w / n, 4), n) for k, (w, n) in agg.items() if n > 0}


def stats_lines(exp):
    path = os.path.join(exp, "league_stats.jsonl")
    return sum(1 for _ in open(path)) if os.path.exists(path) else 0


def build_league(pool, init, best, k, ratings):
    """Self-history pool for chunk k (user design rev3, cap 6): init
    (permanent floor) + best (gated champion) + newest gen + the
    POOL_TOP strongest older gens by last chunk's rollout ratings +
    1 random unrated/leftover gen (diversity slot)."""
    entries = [{"name": "init", "path": init}, {"name": "best", "path": best}]
    gens = [g for g in range(1, k) if os.path.exists(
        os.path.join(pool, f"gen_{g}.pt"))]
    chosen = []
    if gens:
        newest, rest = gens[-1], gens[:-1]
        strongest = sorted((g for g in rest if f"gen_{g}" in ratings),
                           key=lambda g: ratings[f"gen_{g}"][0])[:POOL_TOP]
        leftover = [g for g in rest if g not in strongest]
        if leftover:
            chosen.append(random.Random(4600 + k).choice(leftover))
        chosen += sorted(strongest) + [newest]
    entries += [{"name": f"gen_{g}", "path": os.path.join(pool, f"gen_{g}.pt")}
                for g in chosen]
    return entries


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    a = ap.parse_args()
    exp = a.exp_dir
    pool = os.path.join(exp, "pool")
    os.makedirs(pool, exist_ok=True)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    init = os.path.join(pool, "init.pt")
    best = os.path.join(pool, "best.pt")
    if not os.path.exists(init):
        assert sh(["gsutil", "-q", "cp", FLAG_GS, init]) == 0
        shutil.copy(init, best)

    league_file = os.path.join(pool, "league.json")
    ratings = {}
    for k in range(1, N_CHUNKS + 1):
        entries = build_league(pool, init, best, k, ratings)
        json.dump(entries, open(league_file, "w"))
        print(f"chunk {k} pool: {[e['name'] for e in entries]} "
              f"ratings={ratings}", flush=True)
        mark = stats_lines(exp)
        target = k * CHUNK
        cmd = [sys.executable, "scripts/train_dnn_ppo.py",
               "--arch", "convformer_m_v3r_m46",
               "--total_games", str(target),
               "--gpu_infer", "--gpu_infer_opponents",
               "--games_per_worker", "32", "--infer_max_batch", "128",
               "--lr", "6e-5", "--warmup_updates", "150",
               "--entropy_coef", "0.003", "--gae_lambda", "0.95",
               # learner fixed at 1 seat (user 2026-08-29): 3 opponent seats
               # per game, zero learner-learner correlation; 1 trajectory per
               # game means 8192 games/iter restores mirror's 8192 episodes
               # per update cycle exactly
               "--league_learner_seats", "1",
               "--games_per_iter", "8192",
               "--league", league_file, "--league_frac", "1.0",
               "--milestones", ",".join(str(i * CHUNK) for i in range(1, N_CHUNKS)),
               "--exp_dir", exp]
        if k == 1:
            cmd += ["--init", init]
        else:
            cmd += ["--resume", os.path.join(exp, "games_final.pt")]
        rc = sh(cmd)
        if rc != 0:
            print(f"!!! chunk {k} failed rc={rc}", flush=True)
            break
        snap_src = os.path.join(exp, "games_final.pt")
        snap = os.path.join(pool, f"gen_{k}.pt")
        shutil.copy(snap_src, snap)
        ratings = rollout_ratings(exp, mark)     # this chunk's rollout Elo

        # promotion gate: snapshot vs incumbent best, duplicate deals
        from scripts.run_elo_league import play_pair
        t0 = time.time()
        scores = play_pair("cand", snap, "best", best, GATE_DEALS,
                           48100000 + k * 1000, 16, "cuda", temp_a=1.0)
        share = sum(scores) / GATE_DEALS
        promoted = share > GATE_SHARE
        if promoted:
            shutil.copy(snap, best)
        rec = {"gen": k, "games": target, "gate_share": share,
               "promoted": promoted, "gate_s": round(time.time() - t0, 1),
               "rollout_ratings": ratings,
               "pool": [e["name"] for e in entries]}
        with open(os.path.join(exp, "gens.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"GEN {k}: share={share:.3f} promoted={promoted}", flush=True)
    print("EXP46C DRIVER DONE", flush=True)


if __name__ == "__main__":
    main()
