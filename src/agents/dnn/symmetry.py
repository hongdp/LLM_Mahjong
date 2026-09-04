"""Rule symmetries of riichi mahjong as tensor permutations (exp61, 2026-09-03).

The three suits (m/p/s) are interchangeable under every Tenhou/Majsoul rule
(yaku, dora, one red five per suit) with a single exception — ryuuiisou
(all green: 2/3/4/6/8s + hatsu) — so a policy should be invariant under the
6 suit permutations. This module provides:

* tile / plane / action-slot permutations for a given suit permutation,
* `SuitSymmetrized`: a drop-in wrapper that averages a net's logits over all
  6 permutations (test-time symmetrisation; the wrapper proxies the
  attributes the rollout stack reads: encoder_variant, action_space, ...),
* `green_count`: the ryuuiisou guard (skip symmetrising states that are
  plausibly heading for it).

Every encoder variant handled here (v1, v1r, v3, v3r, v3rh, v3r2) indexes
ALL planes by the 34-tile axis and carries no suit-specific scalar, so
permuting the last axis is an exact symmetry of the observation. (v4's
event buffer and Mortal's 934-plane obs are NOT plane-per-tile; refuse them.)

Number reflection (1<->9) is a symmetry of hand structure but not of the dora
rule (indicator + 1); it is deliberately not implemented here.
"""
from __future__ import annotations

import itertools
import re
from typing import List, Sequence

import numpy as np
import torch
from torch import nn

from src.agents.dnn import encoder as _enc
from src.agents.dnn import mortal_action as _ma

TILE_TYPES = _enc.TILE_TYPES
SUIT_PERMS: List[tuple] = [tuple(p) for p in itertools.permutations(range(3))]   # identity first
PLANE_PER_TILE_VARIANTS = ("v1", "v1r", "v3", "v3r", "v3rh", "v3r2")
_SUIT_CH = "mps"
GREEN_TILES = ("2s", "3s", "4s", "6s", "8s", "6z")


def tile_perm(sp: Sequence[int]) -> np.ndarray:
    """[34] int: tile index i -> its index after moving suit s to slot sp[s]."""
    p = np.arange(TILE_TYPES)
    for i in range(27):
        s, r = divmod(i, 9)
        p[i] = sp[s] * 9 + r
    return p


def slot_perm(space_name: str, sp: Sequence[int]) -> np.ndarray:
    """Action-slot permutation matching `tile_perm` for an action space."""
    tp = tile_perm(sp)
    if space_name == "mortal46":
        p = np.arange(_ma.MORTAL_ACTION_DIM)
        p[:TILE_TYPES] = tp                                  # discard / kan-select tiles
        for s in range(3):                                   # red fives 34..36
            p[34 + s] = 34 + sp[s]
        return p                                             # 37..45: semantic slots, fixed
    if space_name == "native":
        # slot = type * 34 + key_tile; tile-less types (skip / kyuushu) always
        # sit at key 0 and must NOT follow tile 0 (1m) into another suit
        p = np.arange(_enc.ACTION_DIM)
        for t, name in enumerate(_enc.ACTION_TYPES):
            if name in ("skip", "kyuushu"):
                continue
            p[t * TILE_TYPES:(t + 1) * TILE_TYPES] = t * TILE_TYPES + tp
        return p
    raise NotImplementedError(f"no slot permutation for action space {space_name!r}")


_IDX_CACHE = {}


def apply_perm(x: torch.Tensor, p: np.ndarray) -> torch.Tensor:
    """out[..., p[i]] = x[..., i]  (permute the LAST axis). Index tensors are
    cached per (perm, device) so no host->device copy happens per call (the
    inference server's CUDA-graph capture rejects unpinned H2D copies)."""
    key = (p.tobytes(), str(x.device))
    idx = _IDX_CACHE.get(key)
    if idx is None:
        idx = torch.as_tensor(np.argsort(p), device=x.device, dtype=torch.long)
        _IDX_CACHE[key] = idx
    return x.index_select(-1, idx)


def rename_tile(tok: str, sp: Sequence[int]) -> str:
    """'5m' / '0p' / '3s*' -> same tile in the permuted suit; honors unchanged."""
    m = re.fullmatch(r"([0-9])([mps])(\*?)", tok)
    if not m:
        return tok
    return f"{m.group(1)}{_SUIT_CH[sp[_SUIT_CH.index(m.group(2))]]}{m.group(3)}"


_TILE_TOKEN = re.compile(r"\b([0-9][mps]\*?)")


def rename_action(action_xml: str, sp: Sequence[int]) -> str:
    """Rewrite every tile token inside an engine action XML string."""
    return _TILE_TOKEN.sub(lambda m: rename_tile(m.group(1), sp), action_xml)


def green_count(hand: Sequence[str], melds) -> int:
    """Ryuuiisou guard: green tiles in hand + own melds (red 5s is never green)."""
    n = sum(1 for t in hand if t.replace("*", "") in GREEN_TILES)
    for m in melds or ():
        n += sum(1 for t in m["tiles"] if t.replace("*", "") in GREEN_TILES)
    return n


class SuitSymmetrized(nn.Module):
    """Average a policy net's logits over the 6 suit permutations.

    forward(planes, scalars, mask) -> logits, same contract as the wrapped
    net, so the GPU inference server and the arena can use it unchanged.
    The 6 views are stacked into one batch (6B) for a single forward.
    """

    def __init__(self, net: nn.Module, perms: Sequence[Sequence[int]] = SUIT_PERMS):
        super().__init__()
        variant = getattr(net, "encoder_variant", "v1")
        if variant not in PLANE_PER_TILE_VARIANTS:
            raise ValueError(f"encoder variant {variant!r} is not plane-per-tile; cannot symmetrise")
        self.net = net
        self.perms = [tuple(p) for p in perms]
        space = getattr(net, "action_space", None) or "native"
        self._tile_p = [tile_perm(p) for p in self.perms]
        self._slot_p = [slot_perm(space, p) for p in self.perms]
        # proxied metadata (the rollout stack reads these off the policy object)
        self.encoder_variant = variant
        self.action_space = space
        self.arch_name = getattr(net, "arch_name", None)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.net, name)

    def state_dict(self, *args, **kwargs):
        # the rollout stack ships model-0 weights to the inference server as
        # a plain state dict and rebuilds the bare arch there (then re-wraps
        # via cfg["symmetrize"]); expose the inner net's keys unprefixed
        return self.net.state_dict(*args, **kwargs)

    def _views(self, planes, mask):
        P = torch.cat([apply_perm(planes, p) for p in self._tile_p], 0)
        M = torch.cat([apply_perm(mask, p) for p in self._slot_p], 0)
        return P, M

    def _unview(self, logits, B):
        outs = []
        for k, p in enumerate(self._slot_p):
            outs.append(apply_perm(logits[k * B:(k + 1) * B], np.argsort(p)))
        return torch.stack(outs, 0)                          # [K, B, A]

    def forward(self, planes, scalars, mask):
        B = planes.shape[0]
        P, M = self._views(planes, mask)
        S = scalars.repeat(len(self.perms), 1)
        logits = self.net(P, S, M).float()
        return self._unview(logits, B).mean(0)

    def forward_with_value(self, planes, scalars, mask, cfeats=None):
        B = planes.shape[0]
        P, M = self._views(planes, mask)
        S = scalars.repeat(len(self.perms), 1)
        cf = cfeats.repeat(len(self.perms), 1) if cfeats is not None else None
        logits, v = self.net.forward_with_value(P, S, M, cf)
        return self._unview(logits.float(), B).mean(0), v[:B]

    def per_view_logits(self, planes, scalars, mask):
        """[K, B, A] logits of every view mapped back to the identity frame
        (diagnostics: how much the views disagree)."""
        B = planes.shape[0]
        P, M = self._views(planes, mask)
        S = scalars.repeat(len(self.perms), 1)
        return self._unview(self.net(P, S, M).float(), B)


_GREEN_IDX = [19, 20, 21, 23, 25, 32]          # 2s 3s 4s 6s 8s 6z on the 34 axis


def make_batch_augmenter(variant: str, space_name: str, device, green_max: int = 7,
                         perms: Sequence[Sequence[int]] = SUIT_PERMS):
    """Training-time suit augmentation (exp62) as a batch transform.

    Returns f(planes, mask, label) -> (planes, mask, label): the batch is cut
    into len(perms) chunks and chunk k is rewritten in the k-th suit
    permutation (planes on the tile axis, mask and label on the action-slot
    axis). The DataLoader already shuffles, so the chunk assignment is a
    uniform random permutation per sample. Samples whose own hand + melds
    hold more than `green_max` green tiles keep the identity (ryuuiisou is
    the one suit-asymmetric yaku). Planes 0-3 are hand-count>=k and plane 4
    own-meld presence in every plane-per-tile variant, so the guard reads
    them directly.
    """
    if variant not in PLANE_PER_TILE_VARIANTS:
        raise ValueError(f"encoder variant {variant!r} is not plane-per-tile; cannot augment")
    tile_idx = [torch.as_tensor(np.argsort(tile_perm(p)), device=device, dtype=torch.long) for p in perms]
    slot_fwd = [torch.as_tensor(slot_perm(space_name, p), device=device, dtype=torch.long) for p in perms]
    slot_idx = [torch.as_tensor(np.argsort(slot_perm(space_name, p)), device=device, dtype=torch.long)
                for p in perms]
    green = torch.as_tensor(_GREEN_IDX, device=device, dtype=torch.long)
    K = len(perms)

    def augment(planes, mask, label):
        B = planes.shape[0]
        hand = planes[:, 0:4][:, :, green].sum((1, 2))            # own hand green count
        meld = planes[:, 4][:, green].sum(1) * 3                   # own meld presence ~ 3 tiles each
        guard = (hand + meld) > green_max                          # keep identity
        p_out, m_out, y_out = planes.clone(), mask.clone(), label.clone()
        bounds = torch.linspace(0, B, K + 1).long().tolist()
        for k in range(1, K):                                      # chunk 0 = identity
            lo, hi = bounds[k], bounds[k + 1]
            if hi <= lo:
                continue
            sel = ~guard[lo:hi]
            if not bool(sel.any()):
                continue
            rows = torch.arange(lo, hi, device=planes.device)[sel]
            p_out[rows] = planes[rows].index_select(-1, tile_idx[k])
            m_out[rows] = mask[rows].index_select(-1, slot_idx[k])
            y_out[rows] = slot_fwd[k][label[rows]]
        return p_out, m_out, y_out

    return augment


def maybe_symmetrize(net: nn.Module, tag) -> nn.Module:
    """Wrap `net` when a checkpoint blob / rollout cfg carries
    `symmetrize="suit6"` (the only mode so far). Identity otherwise."""
    if not tag:
        return net
    if tag != "suit6":
        raise ValueError(f"unknown symmetrize mode {tag!r}")
    return SuitSymmetrized(net)
