"""Commitment-style fold option (exp103, pure line).

One extra policy slot, ENTER_FOLD (index 374), legal on the seat's own TURN
decision when at least one opponent is in riichi, the seat itself is not,
and it is not already folding. Choosing it is a two-step decision like
Mortal's riichi: the policy is queried again with the discard mask
restricted to the SAFE set, and the seat stays in fold mode for the rest of
the deal. In fold mode every later turn decision is restricted the same way
and riichi / chi / pon / kan / kyuushu are removed (wins stay legal).

Purity boundary (user 2026-09-30): the safe set is the GENBUTSU set only —
tiles that cannot deal into a riichi seat because they are in that seat's
own river or were discarded by anyone after its riichi declaration. That is
the furiten rule, an engine fact; suji / kabe / honor counts are human
heuristics and deliberately NOT used. The sub-policy inside the option is
the network itself (its own logits over the safe set), so what is imposed is
the option's SHAPE (one commitment covers several turns), not its content.

Three callers share this module so train and play agree bit for bit:
  rust_rollout.collect_rust   (training; VecEnv masks are 374 wide and get
                               widened/restricted here, episodes rewritten)
  selfplay._choose            (probes)
  parallel_rollout            (arena / rating; follow-up round protocol)
"""
from __future__ import annotations
from typing import Iterable, Optional, Set
import numpy as np
import torch
from src.agents.dnn import encoder as _enc

FOLD_SLOT = _enc.ACTION_DIM            # 374
FOLD_ACTION_DIM = _enc.ACTION_DIM + 1  # 375
FOLD_MODE = "fold_discard"             # follow-up mode name handed back by the space
T = _enc.TYPE_TO_ID
_DISCARD_TYPES = (T["discard"], T["discard0"])
_KEEP_TYPES = (T["discard"], T["discard0"], T["ron"], T["tsumo"], T["skip"])
_DROP_TYPES = tuple(i for i in range(len(_enc.ACTION_TYPES)) if i not in _KEEP_TYPES)
_TT = _enc.TILE_TYPES


def _norm(t: str) -> str:
    t = t.rstrip("*")
    return "5" + t[1] if t[0] == "0" else t


def genbutsu_34(table, pid: int) -> Optional[np.ndarray]:
    """[34] bool: tiles safe against EVERY opponent in riichi (Python engine
    table). None when no opponent is in riichi. Same rule as VecEnv.safe_info."""
    opp = [o for o in range(4) if o != pid and table.riichi[o]]
    if not opp:
        return None
    safe = np.ones(_TT, dtype=np.bool_)
    for o in opp:
        s: Set[str] = {_norm(x) for x in table.discards[o]}
        ev = table.river_events[o]
        ridx = next((e[4] for e in ev if e[2]), None)
        if ridx is not None:
            for p in range(4):
                if p == o:
                    continue
                s |= {_norm(e[0]) for e in table.river_events[p] if e[4] > ridx}
        m = np.zeros(_TT, dtype=np.bool_)
        for t in s:
            m[_enc.tile_to_34(t)] = True
        safe &= m
    return safe


def fold_flags(table) -> list:
    f = getattr(table, "fold_mode", None)
    if f is None:
        f = [False, False, False, False]
        try:
            table.fold_mode = f
        except Exception:
            pass
    return f


def enter_legal(mask374: np.ndarray, n_opp_riichi: int, in_fold: bool) -> bool:
    """ENTER_FOLD legality from rule facts only: a turn decision (some discard
    legal), >= 1 opponent riichi, not already folding, and >= 2 legal discards
    (a seat in riichi has exactly one: the drawn tile)."""
    if in_fold or n_opp_riichi <= 0:
        return False
    d = 0
    for t in _DISCARD_TYPES:
        d += int(mask374[t * _TT:(t + 1) * _TT].sum())
    return d >= 2


def restrict_mask(mask374: np.ndarray, genb: Optional[np.ndarray]) -> np.ndarray:
    """Fold-mode mask: discards ∩ genbutsu (falls back to all legal discards
    when none is safe), wins and skip kept, riichi/calls/kyuushu dropped.
    Returns a NEW [374] bool array."""
    m = mask374.copy()
    for t in _DROP_TYPES:
        m[t * _TT:(t + 1) * _TT] = False
    if genb is not None:
        safe_any = False
        for t in _DISCARD_TYPES:
            if (m[t * _TT:(t + 1) * _TT] & genb).any():
                safe_any = True
        if safe_any:
            for t in _DISCARD_TYPES:
                m[t * _TT:(t + 1) * _TT] &= genb
    if not m.any():            # never hand back an empty row (e.g. interrupt with only a call offered)
        m = mask374.copy()
        for t in _DROP_TYPES:
            if t != T["skip"]:
                m[t * _TT:(t + 1) * _TT] = False
        if not m.any():
            return mask374.copy()
    return m


def widen(mask374: np.ndarray, enter: bool) -> np.ndarray:
    out = np.zeros(FOLD_ACTION_DIM, dtype=np.bool_)
    out[:_enc.ACTION_DIM] = mask374
    out[FOLD_SLOT] = enter
    return out


def batch_masks(mask374: np.ndarray, genb: np.ndarray, n_opp: np.ndarray, in_fold: np.ndarray):
    """Vectorised version for the Rust rollout. mask374 [n,374], genb [n,34],
    n_opp [n] (opponents in riichi), in_fold [n] bool. Returns
    (mask375 [n,375] for the policy query, enter [n] bool)."""
    n = mask374.shape[0]
    out = np.zeros((n, FOLD_ACTION_DIM), dtype=np.bool_)
    enter = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if in_fold[i]:
            out[i, :_enc.ACTION_DIM] = restrict_mask(mask374[i], genb[i] if n_opp[i] > 0 else None)
        else:
            out[i, :_enc.ACTION_DIM] = mask374[i]
            enter[i] = enter_legal(mask374[i], int(n_opp[i]), False)
            out[i, FOLD_SLOT] = enter[i]
    return out, enter


def fold_discard_mask(mask374: np.ndarray, genb: Optional[np.ndarray]) -> np.ndarray:
    """Second-step mask after ENTER_FOLD: safe discards only (same rule as
    restrict_mask, wins/skip excluded because the row IS a turn decision
    and the commitment means 'discard a safe tile now')."""
    m = restrict_mask(mask374, genb)
    for t in (T["ron"], T["tsumo"], T["skip"]):
        m[t * _TT:(t + 1) * _TT] = False
    if not m.any():
        m = restrict_mask(mask374, genb)
    return widen(m, False)
