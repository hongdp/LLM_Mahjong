"""exp113: function-preserving widen + deepen of a stem + ResBlock CnnPolicy checkpoint (net2net style).

Widening C -> C': every NEW channel gets random incoming weights (PyTorch default init, so it computes a real
feature and receives gradient) and ZERO outgoing weights (every consumer of a new channel -- the next conv's
input columns feeding OLD output channels, and the policy/value heads' input columns -- is zero), so the old
channels and the logits are unchanged. Deepening B -> B': old blocks keep their (widened) weights at evenly
spaced positions; each inserted block has random convs and its LAST BatchNorm at gamma = beta = 0, so it
outputs 0 and relu(h + 0) = h (h >= 0 after the previous relu): an exact identity whose gamma still gets
gradient. Adam state is not carried (shapes changed); `games`/`iter`/`entropy_alpha` are.

usage: python scripts/net2net_widen.py --src exp107_P/games_256000000.pt --arch cnn_c384b12_v3r --out ...pt
       [--check N]  (eval-mode max |logit diff| on N real self-play states; must be ~0)
"""
from __future__ import annotations

import argparse, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agents.dnn.encoder import TILE_TYPES  # noqa: E402


def widen_state(src: dict, dst_net: torch.nn.Module, c_old: int, b_old: int, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    new = {k: v.clone() for k, v in dst_net.state_dict().items()}      # default-initialised target
    c_new = dst_net.stem.out_channels; b_new = len(dst_net.blocks)
    assert c_new >= c_old and b_new >= b_old
    # stem: old rows copied; new rows keep their random init (incoming)
    new["stem.weight"][:c_old] = src["stem.weight"]; new["stem.bias"][:c_old] = src["stem.bias"]
    # old blocks at evenly spaced positions, ending on the last block
    pos = [round((i + 1) * b_new / b_old) - 1 for i in range(b_old)]
    assert len(set(pos)) == b_old and pos[-1] == b_new - 1
    for j in range(b_new):
        p = f"blocks.{j}."
        if j in pos:
            i = pos.index(j); q = f"blocks.{i}."
            for conv in ("c1", "c2"):
                w = new[p + conv + ".weight"]                               # [C', C', 3]
                w[:c_old, :c_old] = src[q + conv + ".weight"]
                w[:c_old, c_old:] = 0.0                                     # old outputs never read new channels
                new[p + conv + ".bias"][:c_old] = src[q + conv + ".bias"]
            for bn in ("n1", "n2"):
                for t in ("weight", "bias", "running_mean", "running_var"):
                    new[p + bn + "." + t][:c_old] = src[q + bn + "." + t]
                new[p + bn + ".num_batches_tracked"] = src[q + bn + ".num_batches_tracked"].clone()
        else:
            # inserted identity block: convs keep their random init; n2 at gamma = beta = 0 zeroes its output
            new[p + "n2.weight"].zero_(); new[p + "n2.bias"].zero_()
    # heads: trunk features are [C*34 in (C, L) order | scalar_fc 64]
    for head in ("head.0", "value.0"):
        w_old = src[head + ".weight"]; w = new[head + ".weight"]
        w.zero_()
        w[:, :c_old * TILE_TYPES] = w_old[:, :c_old * TILE_TYPES]
        w[:, c_new * TILE_TYPES:] = w_old[:, c_old * TILE_TYPES:]
        new[head + ".bias"] = src[head + ".bias"].clone()
    for k, v in src.items():
        if k.startswith(("scalar_fc.", "head.2.", "value.2.")):
            new[k] = v.clone()
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--arch", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--check", type=int, default=256)
    a = ap.parse_args()
    from src.agents.dnn.arch_zoo import ZOO
    blob = torch.load(a.src, map_location="cpu")
    src = blob["state_dict"]
    c_old = src["stem.weight"].shape[0]; b_old = len({k.split(".")[1] for k in src if k.startswith("blocks.")})
    dst = ZOO[a.arch][0]()
    new = widen_state(src, dst, c_old, b_old, a.seed)
    dst.load_state_dict(new)
    n_old = sum(v.numel() for k, v in src.items() if v.dtype.is_floating_point and "running" not in k)
    n_new = sum(p.numel() for p in dst.parameters())
    out = {k: v for k, v in blob.items() if k not in ("state_dict", "optimizer", "arch")}
    out.update(state_dict=new, arch=a.arch, widened_from=a.src, widen_seed=a.seed)
    torch.save(out, a.out)
    print(f"{a.src} ({c_old}x{b_old}, {n_old/1e6:.1f}M) -> {a.arch} ({dst.stem.out_channels}x{len(dst.blocks)}, {n_new/1e6:.1f}M) -> {a.out}")
    if a.check:
        from scripts.run_elo_league import load_dnn
        from src.agents.dnn.rust_rollout import collect_rust, widen_planes
        import numpy as np
        old_net = load_dnn(a.src, "cuda").eval(); new_net = load_dnn(a.out, "cuda").eval()
        cfg = {"temperature": 1.0, "gamma": 0.995, "games_per_worker": a.check, "encoder_variant": old_net.encoder_variant, "shaping": False}
        eps, _ = collect_rust(old_net, a.check, cfg, 1, [91_000 + i for i in range(a.check)], device="cuda")
        cat = lambda k: np.concatenate([e[k] for e in eps])
        P = widen_planes(torch.from_numpy(cat("planes")).cuda()); S = torch.from_numpy(cat("scalars")).cuda(); M = torch.from_numpy(cat("mask")).cuda()
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        with torch.no_grad():
            d = 0.0; dv = 0.0
            for i in range(0, len(P), 4096):
                lo, vo = old_net.forward_with_value(P[i:i+4096], S[i:i+4096], M[i:i+4096])
                ln, vn = new_net.forward_with_value(P[i:i+4096], S[i:i+4096], M[i:i+4096])
                fin = torch.isfinite(lo)
                d = max(d, float((lo[fin] - ln[fin]).abs().max())); dv = max(dv, float((vo - vn).abs().max()))
        print(f"function check on {len(P)} real states: max |logit diff| {d:.2e}, max |value diff| {dv:.2e}")


if __name__ == "__main__":
    main()
