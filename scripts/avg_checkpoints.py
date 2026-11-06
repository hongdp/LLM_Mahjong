"""Uniform weight average of consecutive trainer checkpoints (exp123, SWA-style).

Averages every floating tensor in `state_dict` across the given checkpoints
(same arch required), copies the metadata (arch/channels/blocks/critic_feats/
games/iter) from the LAST input and drops the optimizer state. Integer /
bool buffers are taken from the last input unchanged.

Usage:
  python scripts/avg_checkpoints.py --out W3.pt \
      experiments/exp113_C/games_368000000.pt experiments/exp113_C/games_376000000.pt \
      experiments/exp113_C/games_384000000.pt
"""

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    blobs = [torch.load(p, map_location="cpu", weights_only=False) for p in args.ckpts]
    archs = {str(b.get("arch")) for b in blobs}
    if len(archs) != 1:
        raise SystemExit(f"refusing to average different archs: {archs}")
    keys = list(blobs[-1]["state_dict"].keys())
    for b in blobs:
        if list(b["state_dict"].keys()) != keys:
            raise SystemExit("state_dict key sets differ between checkpoints")
    avg = {}
    n_avg = 0
    for k in keys:
        ts = [b["state_dict"][k] for b in blobs]
        if ts[0].is_floating_point():
            avg[k] = torch.stack([t.float() for t in ts]).mean(0).to(ts[-1].dtype)
            n_avg += 1
        else:
            avg[k] = ts[-1].clone()
    out = {k: v for k, v in blobs[-1].items() if k not in ("state_dict", "optimizer")}
    out["state_dict"] = avg
    out["averaged_from"] = [str(p) for p in args.ckpts]
    torch.save(out, args.out)
    print(f"averaged {n_avg}/{len(keys)} tensors over {len(blobs)} checkpoints "
          f"({[b.get('games') for b in blobs]}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
