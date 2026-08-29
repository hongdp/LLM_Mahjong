"""exp46-C driver: self-history league gens, self-contained on the VM.

User-designed ecology fix (2026-08-29, rev2): the learner never plays
its current self; opponents come only from its own training history
(self-improvement purity). Pool per chunk = init (bc49, permanent
floor) + best (gated champion-so-far) + 3 most recent gen snapshots +
1 uniformly random older gen — cap 6, the measured knee of the infer
server's per-model batch-fragmentation cost (bench 2026-08-29: eager
window 1.9x at N=3, 3.4x at N=6, 5.3x at N=10). league_frac 1.0 kills
pure mirror games; entropy 0.003 (no exploration tax on a sharp
prior); learner-seat trajectories only (trainer default).

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
POOL_RECENT = 3          # most recent gen snapshots always in the pool
POOL_CAP_NOTE = 6        # init + best + recent 3 + 1 random older


def build_league(pool, init, best, k):
    """Self-history pool for chunk k: init + best + last POOL_RECENT gens
    + 1 random older gen. Deterministic per chunk (seeded by k)."""
    entries = [{"name": "init", "path": init}, {"name": "best", "path": best}]
    gens = [g for g in range(1, k) if os.path.exists(
        os.path.join(pool, f"gen_{g}.pt"))]
    recent, older = gens[-POOL_RECENT:], gens[:-POOL_RECENT]
    if older:
        pick = random.Random(4600 + k).choice(older)
        recent = [pick] + recent
    entries += [{"name": f"gen_{g}", "path": os.path.join(pool, f"gen_{g}.pt")}
                for g in recent]
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
    for k in range(1, N_CHUNKS + 1):
        entries = build_league(pool, init, best, k)
        json.dump(entries, open(league_file, "w"))
        print(f"chunk {k} pool: {[e['name'] for e in entries]}", flush=True)
        target = k * CHUNK
        cmd = [sys.executable, "scripts/train_dnn_ppo.py",
               "--arch", "convformer_m_v3r_m46",
               "--total_games", str(target),
               "--gpu_infer", "--gpu_infer_opponents",
               "--games_per_worker", "32", "--infer_max_batch", "128",
               "--lr", "6e-5", "--warmup_updates", "150",
               "--entropy_coef", "0.003", "--gae_lambda", "0.95",
               "--league", league_file, "--league_frac", "1.0",
               "--milestones", ",".join(str(i * CHUNK) for i in range(1, N_CHUNKS)),
               "--exp_dir", exp]
        if k == 1:
            cmd += ["--init", init]
        else:
            cmd += ["--resume", os.path.join(exp, "latest.pt")]
        rc = sh(cmd)
        if rc != 0:
            print(f"!!! chunk {k} failed rc={rc}", flush=True)
            break
        snap_src = os.path.join(exp, "latest.pt")
        snap = os.path.join(pool, f"gen_{k}.pt")
        shutil.copy(snap_src, snap)

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
               "promoted": promoted, "gate_s": round(time.time() - t0, 1)}
        with open(os.path.join(exp, "gens.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"GEN {k}: share={share:.3f} promoted={promoted}", flush=True)
    print("EXP46C DRIVER DONE", flush=True)


if __name__ == "__main__":
    main()
