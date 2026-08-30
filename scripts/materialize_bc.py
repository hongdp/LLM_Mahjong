"""One-pass tensor materialization for human-BC training (exp51 optimization).

The streaming pipeline re-replays all mjlogs through the MJAI bridge on
EVERY epoch (~10-13 epochs to converge = ~13x redundant CPU work; measured
9.5k rows/s with the GPU mostly idle). This script does the replay exactly
once, in parallel, and writes memory-mappable shards; training then streams
from page cache and becomes GPU-bound.

Quantization: every v3r/v3r2 plane value lives on the k/20 grid or {0,1},
so uint8 x 240 round-trips EXACTLY (asserted per shard). Scalars stay f32.

Layout: <out>/{train|holdout}/shard_NNN/{planes,scalars,mask,label,meta}.bin
        + manifest.json {rows, planes_shape, ...}

Usage:
  PYTHONPATH=. python scripts/materialize_bc.py --variant v3r2 \
      --action_space mortal46 --out data/tenhou/cache_v3r2_m46 --workers 14
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.dnn.human_bc_data import game_decisions, is_holdout, list_games  # noqa: E402

SCALE = 240  # exact for {0,1} and the k/20 grid


def _worker(args):
    """Stream-append writer: O(one game) memory, not O(chunk).

    v1 buffered the whole chunk and np.stack'd at the end — 14 workers x
    ~5.5GB peak OOM-killed the pool on the 64GB box (2026-08-28), and a
    killed child left pool.map hanging forever. Files are opened once and
    appended per game; manifests are written at the end from counters.
    """
    wid, files, variant, action_space, out_dir, holdout_pct = args
    import torch                                            # noqa: F401
    handles, counts, dims = {}, {"train": 0, "holdout": 0}, {}

    def get_handles(split):
        if split not in handles:
            d = os.path.join(out_dir, split, f"shard_{wid:03d}")
            os.makedirs(d, exist_ok=True)
            handles[split] = {k: open(os.path.join(d, f"{k}.bin"), "wb")
                              for k in ("planes", "scalars", "mask",
                                        "label", "meta")}
        return handles[split]

    n_bad = 0
    for path in files:
        split = "holdout" if is_holdout(path, holdout_pct) else "train"
        h = get_handles(split)
        for seat in range(4):
            try:
                for r in game_decisions(path, seat, variant, action_space):
                    p = r["planes"].numpy()
                    q = np.rint(p * SCALE)
                    if not np.allclose(q / SCALE, p, atol=1e-6):
                        raise ValueError("plane value off the k/20 grid")
                    dims.setdefault("planes_shape", list(q.shape))
                    dims.setdefault("scalars_dim", int(r["scalars"].shape[0]))
                    dims.setdefault("mask_dim", int(len(r["mask"])))
                    h["planes"].write(q.astype(np.uint8).tobytes())
                    h["scalars"].write(
                        r["scalars"].numpy().astype(np.float32).tobytes())
                    h["mask"].write(
                        np.asarray(r["mask"], dtype=np.uint8).tobytes())
                    h["label"].write(bytes([r["label"]]))
                    h["meta"].write(bytes(
                        [{"turn": 0, "claim": 1, "chankan": 2}[r["phase"]],
                         int(r["vs_riichi"])]))
                    counts[split] += 1
            except Exception:                                # noqa: BLE001
                n_bad += 1
    out = {}
    for split, hs in handles.items():
        for f in hs.values():
            f.close()
        d = os.path.join(out_dir, split, f"shard_{wid:03d}")
        json.dump({"rows": counts[split], **dims, "scale": SCALE,
                   "variant": variant, "action_space": action_space},
                  open(os.path.join(d, "manifest.json"), "w"))
        out[split] = counts[split]
    return out, n_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/tenhou/raw")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--action_space", default="native")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--holdout_pct", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    files = list_games(a.raw, limit=a.limit)
    chunks = [files[w::a.workers] for w in range(a.workers)]
    jobs = [(w, chunks[w], a.variant, a.action_space, a.out, a.holdout_pct)
            for w in range(a.workers) if chunks[w]]
    t0 = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        results = pool.map(_worker, jobs)
    tr = sum(r[0].get("train", 0) for r in results)
    ho = sum(r[0].get("holdout", 0) for r in results)
    bad = sum(r[1] for r in results)
    json.dump({"train_rows": tr, "holdout_rows": ho, "bad_seat_replays": bad,
               "games": len(files), "variant": a.variant,
               "action_space": a.action_space, "scale": SCALE},
              open(os.path.join(a.out, "summary.json"), "w"))
    print(f"[done] train {tr} rows / holdout {ho} rows / bad {bad} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
