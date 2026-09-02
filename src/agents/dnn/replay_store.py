"""Persistent game store for the value-method line (exp60).

Actors append shards; the learner streams them. Everything goes through the
filesystem so the same code runs on a pod or the workstation.

Format (one np.savez_compressed per shard, ~512 games, all four seats):
  planes  uint8   [N, C, 34]   observation planes x 20 — v3r planes take only
                               the 21 values k/20 (measured), so this is lossless
  scalars float16 [N, S]
  mask    uint8   [N, ceil(A/8)] packbits of the legal-action mask
  actions int16   [N]
  rewards float32 [N]          per-step reward (unscaled engine units)
  returns float32 [N]          discounted return-to-go from the rollout
  lengths int32   [E]          steps per episode (episodes are contiguous)
  tags    int16   [E]          index into meta["tags"] (which model sat there)
  seeds   int64   [E]
Measured 58 bytes/step compressed (fp16 planes alone are 3808).
"""
import glob
import json
import os
import time

import numpy as np

PLANE_SCALE = 20


def encode_planes(planes) -> np.ndarray:
    return np.rint(np.asarray(planes, dtype=np.float32) * PLANE_SCALE).astype(np.uint8)


def decode_planes(u8) -> np.ndarray:
    return u8.astype(np.float32) / PLANE_SCALE


def write_shard(store_dir: str, episodes, tags, meta: dict, name: str = None) -> str:
    """episodes: list of rollout episode dicts (planes/scalars/mask/actions/
    rewards/returns/key); tags: parallel list of model-name strings."""
    os.makedirs(store_dir, exist_ok=True)
    tag_table = sorted(set(tags))
    tag_idx = {t: i for i, t in enumerate(tag_table)}
    n_act = episodes[0]["mask"].shape[1]
    arrays = {
        "planes": np.concatenate([encode_planes(e["planes"]) for e in episodes]),
        "scalars": np.concatenate([e["scalars"] for e in episodes]).astype(np.float16),
        "mask": np.packbits(np.concatenate([e["mask"] for e in episodes]).astype(bool), axis=1),
        "actions": np.concatenate([e["actions"] for e in episodes]).astype(np.int16),
        "rewards": np.concatenate([e["rewards"] for e in episodes]).astype(np.float32),
        "returns": np.concatenate([e["returns"] for e in episodes]).astype(np.float32),
        "lengths": np.array([len(e["actions"]) for e in episodes], dtype=np.int32),
        "tags": np.array([tag_idx[t] for t in tags], dtype=np.int16),
        "seeds": np.array([int(e["key"][0]) for e in episodes], dtype=np.int64),
    }
    name = name or f"shard_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    tmp = os.path.join(store_dir, name + ".npz.tmp")
    path = os.path.join(store_dir, name + ".npz")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    meta = dict(meta, tags=tag_table, n_act=int(n_act), n_steps=int(len(arrays["actions"])),
                n_episodes=int(len(episodes)), written=time.time())
    with open(os.path.join(store_dir, name + ".json"), "w") as f:
        json.dump(meta, f)
    os.replace(tmp, path)         # readers never see a half-written shard
    return path


def read_shard(path: str):
    """Returns (episodes as list of dicts with float32 planes / bool mask, meta)."""
    z = np.load(path)
    meta = json.load(open(path[:-4] + ".json"))
    n_act = meta["n_act"]
    planes = decode_planes(z["planes"])
    mask = np.unpackbits(z["mask"], axis=1)[:, :n_act].astype(bool)
    scal = z["scalars"].astype(np.float32)
    acts = z["actions"].astype(np.int64)
    rew, ret = z["rewards"], z["returns"]
    eps, o = [], 0
    for L, tag, seed in zip(z["lengths"], z["tags"], z["seeds"]):
        s = slice(o, o + int(L))
        eps.append({"planes": planes[s], "scalars": scal[s], "mask": mask[s],
                    "actions": acts[s], "rewards": rew[s], "returns": ret[s],
                    "key": (int(seed), 0), "tag": meta["tags"][int(tag)]})
        o += int(L)
    return eps, meta


class StoreReader:
    """Incremental scan of a store directory (only complete .npz + .json pairs)."""

    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        self.seen = set()

    def new_shards(self):
        paths = []
        for p in sorted(glob.glob(os.path.join(self.store_dir, "shard_*.npz")),
                        key=os.path.getmtime):
            if p in self.seen or not os.path.exists(p[:-4] + ".json"):
                continue
            self.seen.add(p)
            paths.append(p)
        return paths

    def all_shards(self):
        return sorted(glob.glob(os.path.join(self.store_dir, "shard_*.npz")),
                      key=os.path.getmtime)
