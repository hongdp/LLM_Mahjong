"""Rebuild a run's TensorBoard events from its train_log.json.

Used for the win_rate -> decisive_rate display rename: completed runs get a
fresh `tensorboard_r/` dir with the new tag; the original `tensorboard/` dir
is left untouched as the raw record. train_log.json field names never change
(downstream reports/probes read them); only the TB display tag is mapped.

Usage: python scripts/rebuild_tb_from_log.py <exp_dir> [<exp_dir> ...]
"""

import json
import os
import sys

from torch.utils.tensorboard import SummaryWriter

TB_TAG = {"win_rate": "decisive_rate"}
SKIP = ("iter", "games", "wall_s")
OUT = "tensorboard_r"


def rebuild(exp_dir):
    log_path = os.path.join(exp_dir, "train_log.json")
    out_dir = os.path.join(exp_dir, OUT)
    if not os.path.exists(log_path):
        print(f"[skip] {exp_dir}: no train_log.json")
        return
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        print(f"[skip] {exp_dir}: {OUT}/ already populated")
        return
    rows = json.load(open(log_path))
    writer = SummaryWriter(out_dir)
    n = 0
    for r in rows:
        step = int(r.get("games", 0))
        for k, v in r.items():
            if k not in SKIP and isinstance(v, (int, float)):
                writer.add_scalar(TB_TAG.get(k, k), float(v), step)
                n += 1
    writer.flush()
    writer.close()
    print(f"[ok] {exp_dir}: {len(rows)} rows, {n} scalars -> {OUT}/")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        rebuild(d)
