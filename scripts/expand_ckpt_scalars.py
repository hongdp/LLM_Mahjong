"""bc49 -> v3rh warm-start surgery (exp55-D): widen the scalar projection
by zero-initialized columns so the v3rh net is bit-equivalent to bc49
whenever the 3 new match-context scalars are zero."""
import sys

import torch

sys.path.insert(0, ".")

SRC = "experiments/exp49_20260827_205132/B/bc_convformer_m_v3r_m46_best.pt"
DST = "experiments/placement_value/bc49_v3rh_init.pt"


def main():
    blob = torch.load(SRC, map_location="cpu")
    sd = blob["state_dict"]
    w = sd["global_proj.weight"]                      # [d, 29]
    from src.agents.dnn.encoder import N_SCALARS_V3H
    neww = torch.zeros(w.shape[0], N_SCALARS_V3H)
    neww[:, :w.shape[1]] = w
    sd["global_proj.weight"] = neww
    blob["arch"] = "convformer_m_v3rh_m46"
    blob["encoder_variant"] = "v3rh"
    torch.save(blob, DST)

    # equivalence check: v3rh(zeros-context) == v3r on random input
    from src.agents.dnn.arch_zoo import ZOO
    a = ZOO["convformer_m_v3r_m46"][0]()
    b = ZOO["convformer_m_v3rh_m46"][0]()
    from src.agents.dnn.net import load_compatible
    orig = torch.load(SRC, map_location="cpu")["state_dict"]
    load_compatible(a, orig)
    load_compatible(b, sd)
    a.eval(); b.eval()
    p = torch.rand(3, a.in_planes, 34)
    s29 = torch.rand(3, 29)
    s32 = torch.cat([s29, torch.zeros(3, 3)], 1)
    m = torch.zeros(3, a.action_dim, dtype=torch.bool); m[:, :10] = True
    with torch.no_grad():
        la = a(p, s29, m); lb = b(p, s32, m)
    d = (la[m] - lb[m]).abs().max().item()          # legal slots only
    # zero columns contribute exactly 0 mathematically; the residual is
    # GEMM reduction-order float noise from the different in_features
    print(f"max logit delta (float noise bound 1e-4): {d:.2e}")
    assert d < 1e-4
    print(f"saved {DST}")


if __name__ == "__main__":
    main()
