"""Throughput benchmark: CPU per-worker inference vs batched GPU server.
Usage: python scripts/bench_rollout_infer.py --arch cnn_m --workers 16 --games 256 [--gpu]
"""
import argparse, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.agents.dnn.arch_zoo import ZOO, CnnPolicy
from src.agents.dnn.parallel_rollout import collect_parallel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="cnn_m")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--wait_ms", type=float, default=4.0)
    ap.add_argument("--max_batch", type=int, default=256)
    ap.add_argument("--games_per_worker", type=int, default=1)
    a = ap.parse_args()
    if a.arch.startswith("cnnbig"):           # e.g. cnnbig192x40
        ch, bl = map(int, a.arch[6:].split("x"))
        net = CnnPolicy(ch, bl)
        cfg_arch = None
    else:
        net = ZOO[a.arch][0]()
        cfg_arch = a.arch
    cfg = dict(channels=getattr(getattr(net, "stem", None), "out_channels", 64), blocks=3, arch=cfg_arch,
               temperature=1.0, gamma=0.995, shaping=False, seed=1, critic_feats="none",
               gpu_infer=a.gpu, infer_wait_ms=a.wait_ms, infer_max_batch=a.max_batch,
               games_per_worker=a.games_per_worker)
    if cfg_arch is None:
        # trainer-style cfg for MahjongPolicyNet needs channels/blocks; CnnPolicy
        # is a subclass with the same state_dict layout, so rebuild via those
        cfg["channels"], cfg["blocks"] = ch, bl
    seeds = [9_000_000 + i for i in range(a.games)]
    t0 = time.time()
    eps, res = collect_parallel(net, a.games, cfg, a.workers, seeds)
    dt = time.time() - t0
    n_dec = sum(len(e["actions"]) for e in eps)
    print(f"{a.arch} workers={a.workers} K={a.games_per_worker} gpu={a.gpu}: {a.games/dt:.1f} games/s, "
          f"{n_dec/dt:.0f} decisions/s, {dt:.1f}s")


if __name__ == "__main__":   # REQUIRED: the GPU server is spawned and re-imports __main__
    main()
