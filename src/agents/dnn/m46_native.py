"""exp124: Mortal's 46-slot action space on top of the Rust VecEnv's native 374 masks.

The VecEnv only speaks the native (type x key_tile) space. `MortalActionSpace`
(action_space.py) builds its masks from the Python engine's action XML; this
module derives the SAME masks from a batch of native masks, vectorised, so a
46-slot net (ConvFormer46, i.e. bc49's architecture) can be trained from
scratch on the Rust engine. Equivalence with `MortalActionSpace` on the
Python engine is asserted decision by decision in tests/test_m46_native.py.

Native layout (encoder.action_to_index): index = TYPE_TO_ID[type] * 34 + key,
key = the action's tile (normalised; discard0/riichi0 = the red five of that
suit, key 5x), except chi whose key is the FIRST consumed tile -- the lower
of the two tiles that leave the hand. The only context the 46-slot chi
needs on top of that is the called tile, which is the last discard and is
one-hot in v1r/v3r plane 13 (`called_from_planes`).

46-slot layout (mortal_action): 0..33 discard (plain), 34..36 discard red
5m/5p/5s, 37 riichi (declare), 38/39/40 chi low/mid/high, 41 pon, 42 kan
(declare), 43 agari (ron or tsumo), 44 ryukyoku (kyuushu), 45 pass.
Second steps reuse the tile slots: riichi-select (0..33 plain riichi
discard, 34..36 red), kan-select (0..33 kan tile; only when >1 kan tile).
"""
from __future__ import annotations

import numpy as np

from src.agents.dnn import encoder as _enc
from src.agents.dnn import mortal_action as _ma

T = _enc.TILE_TYPES
_ID = _enc.TYPE_TO_ID
DISCARD, RIICHI, CHI, PON, KAN, RON, TSUMO, SKIP = (
    _ID[k] for k in ("discard", "riichi", "chi", "pon", "kan", "ron", "tsumo", "skip"))
DISCARD0, RIICHI0, KYUUSHU = _ID["discard0"], _ID["riichi0"], _ID["kyuushu"]
FIVES = np.array([4, 13, 22], dtype=np.int64)          # 5m, 5p, 5s key tiles
DIM = _ma.MORTAL_ACTION_DIM
LAST_DISCARD_PLANE = 13

RIICHI_MODE, KAN_MODE = 1, 2


def _blk(M: np.ndarray, t: int) -> np.ndarray:
    return M[:, t * T:(t + 1) * T]


def called_from_planes(planes_u8: np.ndarray) -> np.ndarray:
    """[n, C*34] (or [n, C, 34]) encoder planes -> [n] last-discard tile (-1 if none)."""
    p = planes_u8.reshape(planes_u8.shape[0], -1, T)[:, LAST_DISCARD_PLANE, :] > 0
    return np.where(p.any(1), p.argmax(1), -1).astype(np.int64)


def _chi_keys(called: np.ndarray):
    """[(key, valid)] for chi (low, mid, high) given the called tile c: the lower consumed tile
    is c+1 (we hold c+1,c+2), c-1 (c-1,c+1), c-2 (c-2,c-1). `valid` = the key exists in c's suit
    (a clipped / cross-suit key would alias another shape's slot); key is clipped for indexing."""
    c = called
    out = []
    for d in (1, -1, -2):
        k = c + d
        ok = (c >= 0) & (c < 27) & (k >= 0) & (k < 27) & (k // 9 == c // 9)
        out.append((np.clip(k, 0, T - 1), ok))
    return out


def first_mask(M: np.ndarray, called: np.ndarray) -> np.ndarray:
    """native [n,374] bool -> 46-slot first-step mask [n,46]."""
    n = M.shape[0]
    out = np.zeros((n, DIM), dtype=bool)
    out[:, :T] = _blk(M, DISCARD)
    out[:, 34:37] = _blk(M, DISCARD0)[:, FIVES]
    out[:, _ma.IDX_RIICHI] = _blk(M, RIICHI).any(1) | _blk(M, RIICHI0).any(1)
    chi = _blk(M, CHI)
    if chi.any():
        r = np.arange(n)
        for slot, (key, ok) in zip((_ma.IDX_CHI_LOW, _ma.IDX_CHI_MID, _ma.IDX_CHI_HIGH), _chi_keys(called)):
            out[:, slot] = chi[r, key] & ok
    out[:, _ma.IDX_PON] = _blk(M, PON).any(1)
    out[:, _ma.IDX_KAN] = _blk(M, KAN).any(1)
    out[:, _ma.IDX_AGARI] = _blk(M, RON).any(1) | _blk(M, TSUMO).any(1)
    out[:, _ma.IDX_RYUKYOKU] = _blk(M, KYUUSHU).any(1)
    out[:, _ma.IDX_PASS] = _blk(M, SKIP).any(1)
    return out


def riichi_mask(M: np.ndarray) -> np.ndarray:
    out = np.zeros((M.shape[0], DIM), dtype=bool)
    out[:, :T] = _blk(M, RIICHI)
    out[:, 34:37] = _blk(M, RIICHI0)[:, FIVES]
    return out


def kan_mask(M: np.ndarray) -> np.ndarray:
    out = np.zeros((M.shape[0], DIM), dtype=bool)
    out[:, :T] = _blk(M, KAN)
    return out


def follow_up_mode(slot1: np.ndarray, M: np.ndarray) -> np.ndarray:
    """0 none / RIICHI_MODE / KAN_MODE (kan only when >1 kan tile, as MortalActionSpace.follow_up)."""
    mode = np.zeros(len(slot1), dtype=np.int64)
    mode[slot1 == _ma.IDX_RIICHI] = RIICHI_MODE
    kan = slot1 == _ma.IDX_KAN
    if kan.any():
        mode[kan & (_blk(M, KAN).sum(1) > 1)] = KAN_MODE
    return mode


def second_mask(M: np.ndarray, mode: np.ndarray) -> np.ndarray:
    """rows with mode != 0 only: [m, 46]."""
    out = np.zeros((M.shape[0], DIM), dtype=bool)
    r = mode == RIICHI_MODE
    k = mode == KAN_MODE
    if r.any():
        out[r] = riichi_mask(M[r])
    if k.any():
        out[k] = kan_mask(M[k])
    return out


def to_native(slot1: np.ndarray, slot2: np.ndarray, M: np.ndarray, called: np.ndarray) -> np.ndarray:
    """(first slot, second slot or -1) -> native action index [n]; raises if the result is not legal."""
    n = len(slot1)
    r = np.arange(n)
    out = np.full(n, -1, dtype=np.int64)
    s = slot1
    d = s < T
    out[d] = DISCARD * T + s[d]
    d0 = (s >= 34) & (s < 37)
    out[d0] = DISCARD0 * T + FIVES[s[d0] - 34]
    keys = _chi_keys(called)
    for slot, (key, _ok) in zip((_ma.IDX_CHI_LOW, _ma.IDX_CHI_MID, _ma.IDX_CHI_HIGH), keys):
        c = s == slot
        out[c] = CHI * T + key[c]

    def first_legal(t, sel):
        if sel.any():
            out[sel] = t * T + _blk(M, t)[sel].argmax(1)

    first_legal(PON, s == _ma.IDX_PON)
    first_legal(KYUUSHU, s == _ma.IDX_RYUKYOKU)
    first_legal(SKIP, s == _ma.IDX_PASS)
    ag = s == _ma.IDX_AGARI
    if ag.any():
        ron = _blk(M, RON).any(1)
        first_legal(RON, ag & ron)
        first_legal(TSUMO, ag & ~ron)
    kan = s == _ma.IDX_KAN
    if kan.any():
        single = kan & (slot2 < 0)
        first_legal(KAN, single)
        sel = kan & (slot2 >= 0)
        out[sel] = KAN * T + slot2[sel]
    ri = s == _ma.IDX_RIICHI
    if ri.any():
        s2 = slot2
        p = ri & (s2 >= 0) & (s2 < T)
        out[p] = RIICHI * T + s2[p]
        q = ri & (s2 >= 34)
        out[q] = RIICHI0 * T + FIVES[s2[q] - 34]
    bad = (out < 0) | ~M[r, np.clip(out, 0, M.shape[1] - 1)]
    if bad.any():
        i = int(np.nonzero(bad)[0][0])
        raise RuntimeError(f"m46_native.to_native: illegal native action for row {i}: slot1={int(slot1[i])} "
                           f"slot2={int(slot2[i])} called={int(called[i])} -> {int(out[i])}; "
                           f"legal={np.nonzero(M[i])[0].tolist()}")
    return out
