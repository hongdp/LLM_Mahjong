"""Checkpoint -> policy net, the one loader every evaluator shares.

Moved verbatim from scripts/run_arena_dnn.py (2026-11-01 structure clean-up);
that script keeps `load_dnn(path, device)` as a thin re-export because
resident evaluators and the Majsoul service import it from there.
"""

import torch

from src.agents.dnn.net import MahjongPolicyNet, load_compatible


def load_dnn(path, device, verbose: bool = True):
    """Rebuild the net from the checkpoint's own arch tag, load the policy
    keys (critic keys may be absent or shape-mismatched) and apply the
    test-time suit symmetrisation the checkpoint asks for (exp61)."""
    blob = torch.load(path, map_location=device)
    # .get(): the trainer has always written "arch" (None for the default
    # CNN) since exp10 — `"arch" in blob` sent every post-exp10 default-CNN
    # checkpoint into the zoo branch and KeyError'd on None (exp12 arenas).
    if blob.get("arch"):
        from src.agents.dnn.arch_zoo import ZOO
        net = ZOO[blob["arch"]][0]().to(device)
    else:
        net = MahjongPolicyNet(channels=blob.get("channels", 64),
                               blocks=blob.get("blocks", 3)).to(device)
    # Policy keys must load fully; critic keys (value*/hazard_head*) may be
    # absent (pre-critic) or shape-mismatched (exp11 privileged variants).
    # The value path is unused at play time, so both are harmless —
    # load_compatible enforces exactly that split and raises otherwise.
    skipped = load_compatible(net, blob["state_dict"])
    if skipped and verbose:
        print(f"[load] {path}: {len(skipped)} critic key(s) not loaded; "
              f"policy loaded fully", flush=True)
    net.eval()
    if blob.get("symmetrize"):                 # exp61: test-time suit symmetrisation
        from src.agents.dnn.symmetry import maybe_symmetrize
        net = maybe_symmetrize(net, blob["symmetrize"]).to(device).eval()
    return net
