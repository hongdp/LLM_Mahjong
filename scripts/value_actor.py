"""Actor orchestrator for the value-method line (exp60).

Loop forever: seat the NEWEST promoted checkpoint at one seat of every table
(random seat), fill the other three from the pool (fixed anchors + the last K
generations, one draw per seat), play everything greedy with single-deviation
exploration on the learner seat, and append every seat's trajectory to the
game store. The learner promotes gen_NNNN.pt files into --pool_dir; this
process picks them up on its next shard. Touch <store_dir>/STOP to exit.

  python scripts/value_actor.py --pool_dir experiments/exp60_pool \
      --store_dir experiments/exp60_store --init experiments/_anchors_epoch6/bc49.pt \
      --gpu_infer --gpu_infer_opponents --workers 16
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_dir", required=True)
    ap.add_argument("--store_dir", required=True)
    ap.add_argument("--init", required=True, help="learner ckpt used until the first gen exists")
    ap.add_argument("--arch", default="convformer_m_v3r_m46")
    ap.add_argument("--anchors", default=None,
                    help="json [{name,path}] of fixed pool members (default pool_dir/anchors.json)")
    ap.add_argument("--recent_gens", type=int, default=4)
    ap.add_argument("--games_per_shard", type=int, default=512)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--games_per_worker", type=int, default=8)
    ap.add_argument("--single_dev_p", type=float, default=0.04)
    ap.add_argument("--single_dev_temp", type=float, default=1.0)
    ap.add_argument("--gpu_infer", action="store_true")
    ap.add_argument("--gpu_infer_opponents", action="store_true")
    ap.add_argument("--infer_max_batch", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=9_000_000)
    ap.add_argument("--max_shards", type=int, default=0, help="0 = run until STOP")
    return ap.parse_args()


def latest_gen(pool_dir):
    gens = sorted(glob.glob(os.path.join(pool_dir, "gen_*.pt")))
    return gens[-1] if gens else None


def main():
    args = parse_args()
    from src.agents.dnn.action_space import space_of_arch
    from src.agents.dnn.arch_zoo import ZOO
    from src.agents.dnn.net import load_compatible
    from src.agents.dnn.parallel_rollout import collect_parallel
    from src.agents.dnn.replay_store import write_shard

    os.makedirs(args.store_dir, exist_ok=True)
    anchors_path = args.anchors or os.path.join(args.pool_dir, "anchors.json")
    anchors = json.load(open(anchors_path))
    dev = torch.device(args.device)
    net = ZOO[args.arch][0]().to(dev)
    loaded, shard_no = None, 0
    t_start = time.time()
    print(f"🎬 actor: pool {args.pool_dir}, store {args.store_dir}, "
          f"{len(anchors)} anchors, recent_gens {args.recent_gens}", flush=True)
    while True:
        if os.path.exists(os.path.join(args.store_dir, "STOP")):
            print("🛑 STOP file seen, exiting", flush=True)
            return
        gens = sorted(glob.glob(os.path.join(args.pool_dir, "gen_*.pt")))
        learner_path = gens[-1] if gens else args.init
        if learner_path != loaded:
            blob = torch.load(learner_path, map_location="cpu", weights_only=False)
            load_compatible(net, blob["state_dict"])
            net.eval()
            loaded = learner_path
            print(f"🆕 learner seat now plays {os.path.basename(learner_path)}", flush=True)
        pool = list(anchors) + [{"name": os.path.basename(p)[:-3], "path": p}
                                for p in gens[-args.recent_gens:]]
        cfg = dict(channels=64, blocks=3, arch=args.arch, temperature=0.0, gamma=0.995,
                   games_per_worker=args.games_per_worker, rollout_temps=None,
                   shaping=False, seed=args.seed + shard_no, critic_feats="none",
                   gpu_infer=args.gpu_infer, gpu_infer_opponents=args.gpu_infer_opponents,
                   infer_max_batch=args.infer_max_batch, infer_wait_ms=0.0,
                   infer_device=args.device, bf16_infer=False, no_episodes=False,
                   league=pool, league_frac=1.0, league_learner_seats=1,
                   league_opp_temp=0.0, hanchan=False, hanchan_w_path=None,
                   action_space=space_of_arch(args.arch),
                   single_dev_p=args.single_dev_p, single_dev_temp=args.single_dev_temp,
                   all_seats_episodes=True)
        seeds = [args.seed + shard_no * 100003 + i for i in range(args.games_per_shard)]
        t0 = time.time()
        collect_parallel(net, len(seeds), cfg, args.workers, seeds)
        dt = time.time() - t0
        games = collect_parallel.last_games
        # tag every episode with the model that produced it; the learner seat
        # is the newest gen, the others come from the pool map
        episodes, tags, lp = [], [], []
        for g in games:
            learner = g.get("learner_seats") or []
            opp = g.get("league") or {}
            for e in g["episodes"]:
                pid = e["key"][1]
                if pid in learner:
                    tags.append(os.path.basename(learner_path)[:-3])
                else:
                    tags.append(pool[opp[pid]]["name"])
                episodes.append(e)
            if learner and g.get("points"):
                lp.append(g["points"][learner[0]] - 25000)
        meta = {"shard_no": shard_no, "games": len(games), "learner": os.path.basename(learner_path),
                "pool": [p["name"] for p in pool], "single_dev_p": args.single_dev_p,
                "learner_pts": float(np.mean(lp)) if lp else None,
                "rollout_s": round(dt, 1), "games_per_s": round(len(games) / dt, 1)}
        path = write_shard(args.store_dir, episodes, tags, meta,
                           name=f"shard_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{shard_no:05d}")
        shard_no += 1
        print(f"[{shard_no:4d}] {len(games)} games {dt:5.1f}s ({len(games)/dt:5.1f}局/s) "
              f"{len(episodes)} eps -> {os.path.basename(path)}  learner {meta['learner']} "
              f"pts {meta['learner_pts'] and round(meta['learner_pts'])}  "
              f"total {(time.time()-t_start)/60:.1f}min", flush=True)
        if args.max_shards and shard_no >= args.max_shards:
            print("✅ max_shards reached", flush=True)
            return


if __name__ == "__main__":     # REQUIRED: spawn re-imports __main__
    main()
