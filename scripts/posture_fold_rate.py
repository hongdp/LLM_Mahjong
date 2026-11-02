"""exp116 diagnostic: on identical threatened far-hand states (opponent riichi, self not, own shanten >= 1, own turn,
>= 1 genbutsu and >= 1 non-genbutsu discard legal), how often does each v3rg checkpoint's greedy choice discard a
genbutsu tile? Detects whether training compensates the posture prior. States come from Rust self-play of --gen.
usage: posture_fold_rate.py --gen G0.pt --games 4096 [--start exp113_C/games_final.pt] ckpt1.pt ckpt2.pt ...
"""
from __future__ import annotations

import argparse, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True); ap.add_argument("--games", type=int, default=4096)
    ap.add_argument("--start", default=None, help="v3r checkpoint widened twice: beta=0 and beta=(0,8,16) reference rows")
    ap.add_argument("ckpts", nargs="*")
    a = ap.parse_args()
    import riichi_rs
    from scripts.run_elo_league import load_dnn
    from src.agents.dnn.arch_zoo import ZOO
    from src.agents.dnn.net import load_compatible
    from src.agents.dnn.encoder import TYPE_TO_ID, TILE_TYPES
    dev = "cuda"; q = float(riichi_rs.PLANE_Q)
    D, D0 = TYPE_TO_ID["discard"] * TILE_TYPES, TYPE_TO_ID["discard0"] * TILE_TYPES
    gen = load_dnn(a.gen, dev).eval()
    env = riichi_rs.VecEnv(list(range(9_100_000, 9_100_000 + a.games)), a.games, 0.995, False, 1.0, True, "v3rg")
    Ps, Ss, Ms = [], [], []
    with torch.no_grad():
        while not env.done():
            planes, scalars, mask, seats, gids = env.observe()
            n = planes.shape[0]
            if n == 0:
                env.step([], []); env.drain_finished(); continue
            P = torch.from_numpy(planes).to(dev).view(n, -1, 34).float().div_(q)
            S = torch.from_numpy(scalars).to(dev); M = torch.from_numpy(mask).to(dev)
            g = P[:, -1, :] > 0.5; far = S[:, -2:].sum(1) > 0.5                     # genbutsu plane; 1 / 2+ shanten
            legal_d = M[:, D:D + 34] | M[:, D0:D0 + 34]
            sel = far & (legal_d & g).any(1) & (legal_d & ~g).any(1)
            if sel.any():
                Ps.append(P[sel].cpu()); Ss.append(S[sel].cpu()); Ms.append(M[sel].cpu())
            idx = gen(P, S, M).argmax(1)
            env.step(idx.cpu().numpy().astype(np.int64).tolist(), [0.0] * n); env.drain_finished()
    P, S, M = torch.cat(Ps).to(dev), torch.cat(Ss).to(dev), torch.cat(Ms).to(dev)
    print(f"{len(P)} threatened far-hand decisions from {a.games} self-play games of {os.path.basename(a.gen)}")
    nets = []
    if a.start:
        for tag, beta in (("start beta=0", (0, 0, 0)), ("start beta=0,8,16", (0, 8, 16))):
            n_ = ZOO["cnn_l_v3rg"][0]().to(dev).eval(); load_compatible(n_, torch.load(a.start, map_location=dev)["state_dict"])
            with torch.no_grad():
                n_.posture_beta.copy_(torch.tensor(beta, dtype=torch.float32, device=dev))
            nets.append((tag, n_))
    for ck in a.ckpts:
        nets.append((ck.split("experiments/")[-1], load_dnn(ck, dev).eval()))
    sh1 = S[:, -2] > 0.5
    for tag, n_ in nets:
        with torch.no_grad():
            idx = torch.cat([n_(P[i:i + 4096], S[i:i + 4096], M[i:i + 4096]).argmax(1) for i in range(0, len(P), 4096)])
        t = torch.where(idx < D0, idx - D, idx - D0)                   # tile index for discard / discard0 slots
        is_d = ((idx >= D) & (idx < D + 34)) | ((idx >= D0) & (idx < D0 + 34))
        gb = is_d & P[torch.arange(len(P), device=dev), -1, t.clamp(0, 33)].gt(0.5)
        beta = [round(float(x), 2) for x in n_.posture_beta.tolist()] if getattr(n_, "posture_beta", None) is not None else None
        print(f"{tag:58s} beta {beta}  genbutsu rate: all {gb.float().mean():.3f} | 1-shanten {gb[sh1].float().mean():.3f} | 2+ {gb[~sh1].float().mean():.3f}")


if __name__ == "__main__":
    main()
