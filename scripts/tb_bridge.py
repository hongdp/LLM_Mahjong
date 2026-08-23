"""Stream train_log.json files into TensorBoard event files.

The DNN trainers log JSON rows (one per iteration) rather than TB events.
Rather than restart running jobs to add native logging, this bridge polls
those files and appends any new rows as scalars, so TensorBoard can watch
live training. Idempotent BY CONSTRUCTION: the JSON is the single source of truth, so on
start the bridge wipes its own output dir and rewrites every row. An
earlier version tracked counts in memory only, so restarting it wrote a
SECOND event file with the same steps — TensorBoard merged both and the
x-axis folded back on itself. Writing into a dedicated `tb_bridge/`
subdir also means it can never clobber event files a trainer wrote.

Scalars are indexed by GAMES PLAYED, not iteration, because the two arms
(REINFORCE 1 update/iter vs PPO 44) are only comparable per game.
"""

import argparse
import json
import os
import shutil
import time

from torch.utils.tensorboard import SummaryWriter

SKIP = {"iter", "games", "wall_s"}


def sync(run_dir: str, writers: dict, counts: dict) -> int:
    log_path = os.path.join(run_dir, "train_log.json")
    if not os.path.exists(log_path):
        return 0
    try:
        with open(log_path) as f:
            rows = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return 0                      # mid-write; try again next poll
    done = counts.get(run_dir, 0)
    if len(rows) <= done:
        return 0
    if run_dir not in writers:
        out = os.path.join(run_dir, "tb_bridge")
        shutil.rmtree(out, ignore_errors=True)   # rewrite from scratch
        writers[run_dir] = SummaryWriter(out)
        counts[run_dir] = 0
        done = 0
    w = writers[run_dir]
    for row in rows[done:]:
        step = int(row.get("games", row.get("iter", 0)))
        for k, v in row.items():
            if k in SKIP or not isinstance(v, (int, float)):
                continue
            w.add_scalar(k, float(v), step)
        # wall-clock efficiency, useful when the two arms share a machine
        if row.get("wall_s"):
            w.add_scalar("games_per_sec", row["games"] / row["wall_s"], step)
    w.flush()
    new = len(rows) - done
    counts[run_dir] = len(rows)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="experiment dirs holding train_log.json")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    writers, counts = {}, {}
    while True:
        total = 0
        for r in args.runs:
            n = sync(r, writers, counts)
            if n:
                print(f"[tb-bridge] +{n} rows -> {r} (total {counts[r]})", flush=True)
            total += n
        if args.once:
            break
        time.sleep(args.interval)
    for w in writers.values():
        w.close()


if __name__ == "__main__":
    main()
