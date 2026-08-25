"""Sweep rollout worker count on the training hardware.

Why this exists: the worker/GPU balance is hardware-specific and every attempt
to infer the cloud's optimum from local measurements has been wrong (the local
4080 saturates its 24 cores at ~4.4k decisions/s with the GPU only 43% busy,
while the cloud's RTX PRO 6000 has 40 idle cores and workers blocked ~83% of
the time). So measure it where it runs.

Usage:
  python scripts/phase2_dnn/bench_workers.py --arch mortal_full_xl_m46 \
      --workers 46,92,138 --games_per_worker 32 --games 2048
"""

import argparse
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


class _Gpu(threading.Thread):
    """Time-averaged utilisation: a single nvidia-smi sample is not a
    measurement (one instant read 98% on a run whose true mean was 47%)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.vals, self.stop = [], False

    def run(self):
        while not self.stop:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3).stdout
                self.vals.append(int(out.split()[0]))
            except Exception:
                pass
            time.sleep(0.25)

    def mean(self):
        return sum(self.vals) / len(self.vals) if self.vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mortal_full_xl_m46")
    ap.add_argument("--workers", default="46,92,138")
    ap.add_argument("--games_per_worker", type=int, default=32)
    ap.add_argument("--games", type=int, default=2048)
    ap.add_argument("--infer_wait_ms", type=float, default=0.0)
    ap.add_argument("--infer_max_batch", type=int, default=512)
    args = ap.parse_args()

    from src.agents.dnn.arch_zoo import ZOO
    from src.agents.dnn.parallel_rollout import collect_parallel
    from src.agents.dnn.action_space import space_of_arch

    print(f"BENCH arch={args.arch} games={args.games} "
          f"gpw={args.games_per_worker} wait={args.infer_wait_ms}", flush=True)
    print(f"BENCH cpus={os.cpu_count()}", flush=True)
    print(f"{'workers':>8} {'games/s':>9} {'dec/s':>8} {'GPU%':>6} {'load':>6}",
          flush=True)

    best = (0.0, None)
    for w in [int(x) for x in args.workers.split(",")]:
        net = ZOO[args.arch][0]()
        cfg = dict(arch=args.arch, channels=0, blocks=0, temperature=1.0,
                   gamma=0.995, games_per_worker=args.games_per_worker,
                   rollout_temps=None, shaping=False, seed=3,
                   critic_feats="none", gpu_infer=True,
                   gpu_infer_opponents=False,
                   infer_max_batch=args.infer_max_batch,
                   infer_wait_ms=args.infer_wait_ms, infer_device="cuda",
                   bf16_infer=True, action_space=space_of_arch(args.arch))
        g = _Gpu(); g.start()
        t0 = time.time()
        eps, res = collect_parallel(net, args.games, cfg, workers=w)
        dt = time.time() - t0
        g.stop = True; time.sleep(0.4)
        dec = sum(len(e["actions"]) for e in eps) / dt
        load = os.getloadavg()[0]
        print(f"{w:>8} {args.games/dt:>9.1f} {dec:>8.0f} {g.mean():>5.0f}% "
              f"{load:>6.1f}", flush=True)
        if dec > best[0]:
            best = (dec, w)
        del net

    print(f"BENCH BEST {best[0]:.0f} dec/s at workers={best[1]}", flush=True)


if __name__ == "__main__":
    main()
