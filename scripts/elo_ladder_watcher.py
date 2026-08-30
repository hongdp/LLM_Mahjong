"""Ranked-ladder strength tracking for checkpoints DURING training.

User ask (2026-08-16): "训练中多进行一些 checkpoint 打排位赛测量强度" —
self-play win_rate is a dynamics metric, not strength; this sidecar turns
milestone/periodic checkpoints into an Elo-vs-time curve on the local GPU
at zero cost to cloud rollout throughput.

Modes:
  backfill — rate every games_*.pt already in a local run dir
  watch    — poll a cloud run's GCS dir; whenever latest.pt has advanced
             by >= --min_games since the last rating, snapshot + rate it

Each rating: 3 anchors nearest the candidate's previous rating (adaptive
bracket), --deals per anchor. Results append to the shared league history
and to a per-run TensorBoard scalar (elo/rating, step = games), so the
curve renders next to the run's training metrics.

Usage:
  python scripts/elo_ladder_watcher.py backfill --run_dir experiments/dnn_ppo_massive_20260815
  python scripts/elo_ladder_watcher.py watch --gcs_run dnn_exp11_a2_20260815 --poll 600
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_elo_league import LEAGUE_DIR, rate_checkpoint  # noqa: E402

GCS_BUCKET = "gs://llm-mahjong-experiments"


def nearest_anchors(prev_elo, k=3):
    anchors = json.load(open(f"{LEAGUE_DIR}/anchors.json"))["anchors"]
    return sorted(anchors, key=lambda n: abs(anchors[n]["rating"] - prev_elo))[:k]


def tb_writer(run_name):
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(f"experiments/_cloud_mirror/{run_name}_elo")


def rate_one(ckpt, run_name, games, prev_elo, args, device, writer):
    use = nearest_anchors(prev_elo, args.k_anchors)
    rec = rate_checkpoint(ckpt, f"{run_name}@{games}", args.deals,
                          args.seed_base + games, args.parallel, device,
                          use=use, init_guess=prev_elo,
                          allow_engine_mismatch=args.allow_engine_mismatch)
    writer.add_scalar("elo/rating", rec["elo"], global_step=games)
    writer.add_scalar("elo/se", rec["se"], global_step=games)
    # capability-ordering metrics vs FIXED opponents (the anchors just used):
    # greedy, candidate seat only -> style/* next to elo/* in TensorBoard
    try:
        from src.agents.dnn.style_stats import style_vs_anchors
        from scripts.run_arena_dnn import load_dnn
        anchors = json.load(open(f"{LEAGUE_DIR}/anchors.json"))["anchors"]
        cand = load_dnn(ckpt, device)
        opp = [load_dnn(anchors[n]["path"], device) for n in use]
        sty = style_vs_anchors(cand, opp, games=getattr(args, "style_games", 200),
                               seed0=args.seed_base + games + 7, temperature=0.0,
                               device=device)
        for k, v in sty.items():
            if k != "games":
                writer.add_scalar(f"style/{k}", float(v), global_step=games)
        rec["style"] = sty
        print(f"STYLE {run_name}@{games}: " + " ".join(f"{k}={v:.3f}" for k, v in sty.items() if k != "games"), flush=True)
    except Exception as e:                     # never let a probe kill the watcher
        print(f"style probe failed: {e}", flush=True)
    writer.flush()
    return rec["elo"]


def cmd_backfill(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = os.path.basename(args.run_dir.rstrip("/"))
    ckpts = []
    for p in glob.glob(f"{args.run_dir}/games_*.pt"):
        m = re.match(r"games_(\d+)\.pt", os.path.basename(p))
        if m:
            ckpts.append((int(m.group(1)), p))
    writer = tb_writer(run_name)
    prev = 1000.0
    for games, p in sorted(ckpts):
        if games < args.min_games:
            continue
        prev = rate_one(p, run_name, games, prev, args, device, writer)


def cmd_localwatch(args):
    """Continuous ladder for a LOCAL run dir (exp46): poll for new
    milestone checkpoints, rate each once (history-label dedup like
    watch mode), write elo/rating scalars into the run's TB dir."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = os.path.basename(args.run_dir.rstrip("/"))
    writer = tb_writer(run_name)
    prev, rated = 1000.0, set()
    if os.path.exists(f"{LEAGUE_DIR}/history.jsonl"):
        for line in open(f"{LEAGUE_DIR}/history.jsonl"):
            r = json.loads(line)
            if r["label"].startswith(f"{run_name}@"):
                rated.add(int(r["label"].split("@")[1]))
                prev = r["elo"]
    while True:
        ckpts = []
        for pth in glob.glob(f"{args.run_dir}/games_*.pt"):
            m = re.match(r"games_(\d+)\.pt", os.path.basename(pth))
            if m and int(m.group(1)) >= args.min_games \
                    and int(m.group(1)) not in rated:
                ckpts.append((int(m.group(1)), pth))
        for games, pth in sorted(ckpts):
            prev = rate_one(pth, run_name, games, prev, args, device, writer)
            rated.add(games)
        if os.path.exists(f"{args.run_dir}/games_final.pt") and not ckpts:
            done_marker = f"{args.run_dir}/.ladder_done"
            if os.path.exists(done_marker):
                return
            open(done_marker, "w").write("1")
        time.sleep(args.poll)


def gcs_train_log_games(run):
    try:
        out = subprocess.run(
            ["gsutil", "cat", f"{GCS_BUCKET}/{run}/train_log.json"],
            capture_output=True, timeout=60).stdout
        return json.loads(out)[-1]["games"]
    except Exception:
        return None


def cmd_watch(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = args.gcs_run
    snap_dir = f"experiments/_cloud_ckpts/{run}"
    os.makedirs(snap_dir, exist_ok=True)
    writer = tb_writer(run)
    prev, last_rated = 1000.0, 0
    # resume: pick up where a previous watcher (or crash) left off
    if os.path.exists(f"{LEAGUE_DIR}/history.jsonl"):
        for line in open(f"{LEAGUE_DIR}/history.jsonl"):
            r = json.loads(line)
            if r["label"].startswith(f"{run}@"):
                last_rated = max(last_rated, int(r["label"].split("@")[1]))
                prev = r["elo"]
    while True:
        games = gcs_train_log_games(run)
        if games and games - last_rated >= args.min_games:
            snap = f"{snap_dir}/ladder_{games}.pt"
            try:
                got = subprocess.run(
                    ["gsutil", "cp", f"{GCS_BUCKET}/{run}/latest.pt", snap],
                    capture_output=True, timeout=300).returncode == 0
            except subprocess.TimeoutExpired:
                # slow network must delay a tick, not kill the watcher
                # (2026-08-18: cnn arm curve lost past 194k to exactly this)
                got = False
            if got:
                prev = rate_one(snap, run, games, prev, args, device, writer)
                last_rated = games
        # exit when the run is over (final artifact present) and rated
        done = subprocess.run(
            ["gsutil", "-q", "stat", f"{GCS_BUCKET}/{run}/games_final.pt"],
            capture_output=True, timeout=60).returncode == 0
        if done and games and games - last_rated < args.min_games:
            print(f"[ladder] {run} finished; watcher exiting", flush=True)
            return
        time.sleep(args.poll)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = dict(deals=("--deals", 60), k=("--k_anchors", 3),
                  par=("--parallel", 20), mg=("--min_games", 40000),
                  sb=("--seed_base", 30000000))
    bf = sub.add_parser("backfill")
    bf.add_argument("--run_dir", required=True)
    wa = sub.add_parser("watch")
    wa.add_argument("--gcs_run", required=True)
    wa.add_argument("--poll", type=int, default=600)
    lw = sub.add_parser("localwatch")
    lw.add_argument("--run_dir", required=True)
    lw.add_argument("--poll", type=int, default=600)
    lw.set_defaults(fn=cmd_localwatch)
    for p in (bf, wa, lw):
        for flag, dv in common.values():
            p.add_argument(flag, type=int, default=dv)
        p.add_argument("--allow_engine_mismatch", action="store_true")
        p.add_argument("--style_games", type=int, default=200)
    bf.set_defaults(fn=cmd_backfill)
    wa.set_defaults(fn=cmd_watch)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
