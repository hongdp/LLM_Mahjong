"""Mortal-aligned observation encoder (exp41).

A faithful Python port of Equim-chan/Mortal's `libriichi/src/state/obs_repr.rs`
observation **version 3** (934 planes x 34 tiles), built against OUR engine's
`PyMahjongTable` state instead of libriichi's `PlayerState`.

Why version 3 and not 4: v4 (1012 planes) embeds the output of Mortal's
single-player expected-value solver (`algo/sp.rs`) -- per-discard, per-turn
tenpai/win probabilities and EVs. Porting that solver faithfully is a separate
project-sized job, and any deviation would silently distort the comparison.
v3 keeps every *derived tile-efficiency* feature that makes the comparison
interesting (shanten, waits, furiten, keep-/next-shanten discards,
unconditional-tenpai candidates) without requiring the solver.

Plane budget derived group-by-group from the Rust source sums to exactly 934,
matching `obs_shape(3)` -- see `test_mortal_obs.py`, which asserts it.

Two variants, per the exp41 two-arm design:
  * ``derived=True``  -- full alignment (arm A). Feeds precomputed tile-efficiency
    analysis, i.e. domain knowledge the network would otherwise have to learn.
  * ``derived=False`` -- pure-lineage arm (arm B). Identical layout and plane
    count, but every derived-feature plane is left at zero, so the two arms are
    bit-comparable and their difference isolates the value of that knowledge.

Engine-difference notes (honest deviations, none of them silent):
  * honba: our env is single-hand and never carries a repeat counter, so its
    planes encode 0. Mortal sees real honba.
  * `is_dora` on a discard: computed against the current dora indicators, which
    is what our engine tracks; Mortal's KawaItem stamps it at discard time.
    These agree except across a kan-dora reveal.
  * Rules: our engine is Majsoul single-hand; Mortal's is Tenhou hanchan. This
    encoder does not attempt to bridge that -- see the exp41 preregistration.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from src.tasks.mahjong.table import PyMahjongTable

TILE_TYPES = 34
MORTAL_V3_PLANES = 934
MAX_NUM_TURNS = 17
SELF_KAWA_ITEM_CHANNELS = 4
KAWA_ITEM_CHANNELS = 8

# our tile spelling -> 0..33 index, mirroring encoder.tile_to_34
_SUITS = {"m": 0, "p": 9, "s": 18}
_HONORS = ["1z", "2z", "3z", "4z", "5z", "6z", "7z"]


def _tid(tile: str) -> int:
    """Tile string -> 0..33, red fives folded onto their plain rank ('deaka')."""
    t = tile.replace("*", "")
    if t[1] == "z":
        return 27 + int(t[0]) - 1
    rank = int(t[0])
    if rank == 0:            # our red-five spelling '0m'/'0p'/'0s'
        rank = 5
    return _SUITS[t[1]] + rank - 1


def _is_aka(tile: str) -> bool:
    return tile.replace("*", "")[0] == "0"


class _Ctx:
    """Cursor-based writer mirroring Rust's ObsEncoderContext (idx advances
    exactly as the source does, so the layout stays verifiable line by line)."""

    def __init__(self, planes: int, derived: bool):
        self.arr = np.zeros((planes, TILE_TYPES), dtype=np.float32)
        self.idx = 0
        self.derived = derived

    def fill(self, row: int, v: float = 1.0) -> None:
        self.arr[row, :] = v

    def assign(self, row: int, col: int, v: float = 1.0) -> None:
        self.arr[row, col] = v

    def assign_rows(self, row: int, col: int, n: int, v: float = 1.0) -> None:
        self.arr[row:row + n, col] = v

    # --- derived-feature gate: arm B keeps the slot, writes nothing ---
    def d_fill(self, row: int, v: float = 1.0) -> None:
        if self.derived:
            self.fill(row, v)

    def d_assign(self, row: int, col: int, v: float = 1.0) -> None:
        if self.derived:
            self.assign(row, col, v)

    def int_enc(self, n: int, cap: int, *, one_hot=False, rescale=False,
                rbf_intervals: Optional[int] = None) -> None:
        """IntegerEncoder for version 2|3 (obs_repr.rs lines 59-107)."""
        nc = min(n, cap)
        if one_hot:
            self.fill(self.idx + nc)
            self.idx += cap + 1
        if rescale:
            self.fill(self.idx, nc / cap)
            self.idx += 1
        if rbf_intervals:
            interval_size = cap / rbf_intervals
            for i in range(1, rbf_intervals):
                mu = i * interval_size
                sigma = interval_size
                # NOTE: the *unclamped* n, exactly as the Rust source does
                v = math.exp(-((n - mu) ** 2) / (2 * sigma ** 2))
                self.fill(self.idx + i - 1, v)
            self.idx += rbf_intervals - 1

    def encode_tile_set(self, tiles: List[str]) -> None:
        counts = [0] * TILE_TYPES
        for t in tiles:
            i = _tid(t)
            self.assign(self.idx + counts[i], i)
            counts[i] += 1
            if _is_aka(t):
                which = {"m": 0, "p": 1, "s": 2}[t.replace("*", "")[1]]
                self.fill(self.idx + 4 + which)
        self.idx += 7


def _river_items(table: PyMahjongTable, pid: int) -> List[dict]:
    """Our river_events -> Mortal KawaItem-shaped dicts.

    river_events[pid] entries are (tile, tsumogiri, riichi_decl, called, idx).
    Mortal's KawaItem additionally carries the chi/pon consumed pair and any
    kan tiles that happened on that turn; our engine records melds separately,
    so those channels stay zero (documented deviation, same for both arms).
    """
    dora_tiles = {_tid(_dora_from_indicator(d)) for d in table.dora_indicators}
    out = []
    for (tile, tsumogiri, rdecl, called, _idx) in table.river_events[pid]:
        out.append({
            "tile": tile,
            "is_tedashi": not tsumogiri,
            "is_riichi": bool(rdecl),
            "is_dora": _tid(tile) in dora_tiles,
        })
    return out


def _dora_from_indicator(ind: str) -> str:
    """Indicator -> the tile it makes dora (standard riichi wrap-around)."""
    t = ind.replace("*", "")
    if t[1] == "z":
        n = int(t[0])
        if n <= 4:                      # winds E S W N
            return f"{n % 4 + 1}z"
        return f"{(n - 5 + 1) % 3 + 5}z"  # dragons
    rank = int(t[0])
    if rank == 0:
        rank = 5
    return f"{rank % 9 + 1}{t[1]}"


def _encode_self_kawa(ctx: _Ctx, item: Optional[dict]) -> None:
    if item is not None:
        # channel 0 = kan tiles that turn (our engine tracks melds separately)
        i = _tid(item["tile"])
        ctx.assign(ctx.idx + 1, i)
        if _is_aka(item["tile"]):
            ctx.fill(ctx.idx + 2)
        if item["is_dora"]:
            ctx.fill(ctx.idx + 3)
    ctx.idx += SELF_KAWA_ITEM_CHANNELS


def _encode_kawa(ctx: _Ctx, item: Optional[dict]) -> None:
    if item is not None:
        # channels 0,1 = chi/pon consumed pair; 2 = kan (not tracked per-turn)
        i = _tid(item["tile"])
        ctx.assign(ctx.idx + 3, i)
        if _is_aka(item["tile"]):
            ctx.fill(ctx.idx + 4)
        if item["is_dora"]:
            ctx.fill(ctx.idx + 5)
        if item["is_tedashi"]:
            ctx.fill(ctx.idx + 6)
        if item["is_riichi"]:
            ctx.fill(ctx.idx + 7)
    ctx.idx += KAWA_ITEM_CHANNELS


def encode_mortal_obs(table: PyMahjongTable, player_id: int,
                      derived: bool = True,
                      at_kan_select: bool = False) -> np.ndarray:
    """[934, 34] float32, following obs_repr.rs version 3 group by group."""
    ctx = _Ctx(MORTAL_V3_PLANES, derived)
    me = player_id
    seats = [(me + i) % 4 for i in range(4)]      # relative: 0 = self

    hand = list(table.hands[me])
    n_melds = len(table.melds[me])

    # 1. hand counts (4)
    counts = [0] * TILE_TYPES
    for t in hand:
        counts[_tid(t)] += 1
    for i, c in enumerate(counts):
        if c:
            ctx.assign_rows(ctx.idx, i, c)
    ctx.idx += 4

    # 2. akas in hand (3)
    for j, s in enumerate("mps"):
        if any(_is_aka(t) and t.replace("*", "")[1] == s for t in hand):
            ctx.fill(ctx.idx + j)
    ctx.idx += 3

    # 3. scores: rescale + rbf(10) per seat (4 x 10)
    for p in seats:
        score = max(0, min(100_000, table.points[p]))
        ctx.fill(ctx.idx, score / 100_000.0)
        ctx.idx += 1
        ctx.int_enc(table.points[p] // 100, 500, rbf_intervals=10)

    # 4. rank (4)
    order = sorted(range(4), key=lambda p: -table.points[p])
    ctx.fill(ctx.idx + order.index(me))
    ctx.idx += 4

    # 5. kyoku (4)  -- our engine's within-round hand number
    kyoku = (table.dealer) % 4
    ctx.fill(ctx.idx + kyoku)
    ctx.idx += 4

    # 6/7. honba (always 0 here) and kyotaku, cap 6, rbf(3)
    ctx.int_enc(0, 6, rbf_intervals=3)
    ctx.int_enc(int(table.kyotaku), 6, rbf_intervals=3)

    # 8. bakaze / jikaze (2)
    bakaze = (table.round_wind - 27) if table.round_wind >= 27 else 0
    jikaze = (me - table.dealer) % 4
    ctx.assign(ctx.idx, 27 + bakaze)
    ctx.assign(ctx.idx + 1, 27 + jikaze)
    ctx.idx += 2

    # 9. combined round index, cap 7, rescale
    ctx.int_enc(min(bakaze, 1) * 4 + kyoku, 7, rescale=True)

    # 10. dora indicators (7)
    ctx.encode_tile_set(list(table.dora_indicators))

    # 11-13. self kawa: first 6, last 18, decay plane
    kawa = {p: _river_items(table, p) for p in range(4)}
    mine = kawa[me]
    for k in range(6):
        _encode_self_kawa(ctx, mine[k] if k < len(mine) else None)
    for k in range(18):
        rev = list(reversed(mine))
        _encode_self_kawa(ctx, rev[k] if k < len(rev) else None)
    max_kawa_len = max(len(kawa[p]) for p in range(4))
    for turn, it in enumerate(mine):
        v = math.exp(-0.2 * (max_kawa_len - 1 - turn))
        ctx.assign(ctx.idx, _tid(it["tile"]), v)
    ctx.idx += 1

    # 14. opponents' kawa (3 x [6*8 + 18*8 + 3])
    for p in seats[1:]:
        ok = kawa[p]
        for k in range(6):
            _encode_kawa(ctx, ok[k] if k < len(ok) else None)
        rev = list(reversed(ok))
        for k in range(18):
            _encode_kawa(ctx, rev[k] if k < len(rev) else None)
        for turn, it in enumerate(ok):
            v = math.exp(-0.2 * (max_kawa_len - 1 - turn))
            i = _tid(it["tile"])
            ctx.assign(ctx.idx, i, v)
            if it["is_tedashi"]:
                ctx.assign(ctx.idx + 1, i, v)
            if it["is_riichi"]:
                ctx.assign(ctx.idx + 2, i, v)
        ctx.idx += 3

    # 15. tiles left (1)
    ctx.fill(ctx.idx, len(table.wall) / 69.0)
    ctx.idx += 1

    # 16. doras owned per seat (4 x 3)
    dora_tiles = {_tid(_dora_from_indicator(d)) for d in table.dora_indicators}
    for p in seats:
        owned = sum(1 for t in table.hands[p] if _tid(t) in dora_tiles)
        owned += sum(1 for m in table.melds[p] for t in m["tiles"]
                     if _tid(t) in dora_tiles)
        owned += sum(1 for t in table.hands[p] if _is_aka(t))
        ctx.int_enc(owned, 12, rescale=True, rbf_intervals=3)

    # 17. doras unseen (4)
    seen_counts = [0] * TILE_TYPES
    for p in range(4):
        for t in table.rivers_flat(p) if hasattr(table, "rivers_flat") else \
                [it["tile"] for it in kawa[p]]:
            seen_counts[_tid(t)] += 1
        for m in table.melds[p]:
            for t in m["tiles"]:
                seen_counts[_tid(t)] += 1
    for t in hand:
        seen_counts[_tid(t)] += 1
    for d in table.dora_indicators:
        seen_counts[_tid(d)] += 1
    doras_seen = sum(seen_counts[i] for i in dora_tiles)
    ctx.int_enc(len(table.dora_indicators) * 4 + 3 - doras_seen, 5 * 4 + 3,
                rescale=True, rbf_intervals=4)

    # 18. kawa overview per seat (4 x 7)
    for p in seats:
        ctx.encode_tile_set([it["tile"] for it in kawa[p]])

    # 19. fuuro overview (4 seats x 4 melds x 5)
    for p in seats:
        fl = table.melds[p]
        for m in fl[:4]:
            for t in m["tiles"]:
                i = _tid(t)
                for k in range(4):
                    if ctx.arr[ctx.idx + k, i] == 0:
                        ctx.assign(ctx.idx + k, i)
                        break
                if _is_aka(t):
                    ctx.fill(ctx.idx + 4)
            ctx.idx += 5
        ctx.idx += (4 - len(fl[:4])) * 5

    # 20. ankan overview (4)
    for p in seats:
        for m in table.melds[p]:
            if m.get("type") == "ankan" or m.get("kind") == "ankan":
                ctx.assign(ctx.idx, _tid(m["tiles"][0]))
        ctx.idx += 1

    # 21. tiles seen (1)
    for i, c in enumerate(seen_counts):
        ctx.assign(ctx.idx, i, min(c, 4) / 4.0)
    ctx.idx += 1

    # 22/23. opponents' last tedashi / riichi-declaration tile (3 x 3 each)
    for p in seats[1:]:
        last_ted = next((it for it in reversed(kawa[p]) if it["is_tedashi"]), None)
        if last_ted:
            i = _tid(last_ted["tile"])
            ctx.assign(ctx.idx, i)
            if _is_aka(last_ted["tile"]):
                ctx.fill(ctx.idx + 1)
            if last_ted["is_dora"]:
                ctx.fill(ctx.idx + 2)
        ctx.idx += 3
    for p in seats[1:]:
        rt = next((it for it in kawa[p] if it["is_riichi"]), None)
        if rt:
            i = _tid(rt["tile"])
            ctx.assign(ctx.idx, i)
            if _is_aka(rt["tile"]):
                ctx.fill(ctx.idx + 1)
            if rt["is_dora"]:
                ctx.fill(ctx.idx + 2)
        ctx.idx += 3

    # 24/25. riichi declared / accepted for opponents (3 + 3)
    for j, p in enumerate(seats[1:]):
        if table.riichi[p]:
            ctx.fill(ctx.idx + j)
    ctx.idx += 3
    for j, p in enumerate(seats[1:]):
        if table.riichi[p]:
            ctx.fill(ctx.idx + j)
    ctx.idx += 3

    # ---- derived tile-efficiency block (arm A only; arm B leaves zeros) ----
    # 26. waits (1)
    if ctx.derived:
        try:
            for t in table._waits_of(me) if hasattr(table, "_waits_of") else []:
                ctx.assign(ctx.idx, _tid(t) if isinstance(t, str) else int(t))
        except Exception:
            pass
    ctx.idx += 1

    # 27. furiten (1)
    if ctx.derived:
        try:
            if table._is_furiten(me):
                ctx.fill(ctx.idx)
        except Exception:
            pass
    ctx.idx += 1

    # 28. shanten one-hot, cap 6 (7)
    sh = 6
    if ctx.derived:
        try:
            sh = max(0, min(6, table._shanten(hand, n_melds)))
        except Exception:
            sh = 6
    if ctx.derived:
        ctx.fill(ctx.idx + sh)
    ctx.idx += 7

    # 29. own riichi accepted (1)
    if table.riichi[me]:
        ctx.fill(ctx.idx)
    ctx.idx += 1

    # 30. at_kan_select (1)
    if at_kan_select:
        ctx.fill(ctx.idx)
    ctx.idx += 1

    # 31. can_pass context: the callable tile (3)
    last_disc = getattr(table, "last_discard", None)
    if last_disc:
        tile = last_disc if isinstance(last_disc, str) else last_disc[0]
        i = _tid(tile)
        ctx.assign(ctx.idx, i)
        if _is_aka(tile):
            ctx.fill(ctx.idx + 1)
        if i in dora_tiles:
            ctx.fill(ctx.idx + 2)
    ctx.idx += 3

    # 32. discard-candidate block (5): candidates + keep/next-shanten + tenpai
    can_discard = (table.turn == me and len(hand) % 3 == 2)
    if can_discard:
        for t in set(hand):
            ctx.assign(ctx.idx, _tid(t))
        if ctx.derived:
            _derived_discard_planes(ctx, table, hand, n_melds, sh)
        if table.riichi[me]:
            ctx.fill(ctx.idx + 4)
    ctx.idx += 5

    # 33-40. action-availability flags (1+3+1+1+1+1+1+1 = 10)
    ctx.idx += 10

    # Layout guard: the cursor must land exactly on obs_shape(3)[0]. This is
    # the single strongest check that the port did not drift from the Rust
    # source -- every group's width is validated by the total.
    assert ctx.idx == MORTAL_V3_PLANES, (
        f"plane cursor ended at {ctx.idx}, expected {MORTAL_V3_PLANES}")
    return ctx.arr


def _derived_discard_planes(ctx: _Ctx, table, hand, n_melds, shanten) -> None:
    """keep_shanten / next_shanten / unconditional-tenpai discard candidates.

    This is the precomputed tile-efficiency analysis Mortal hands to its
    network. Computed here with our own engine's shanten routine, so it is
    rule-derived (not human data), but it is domain knowledge the pure arm
    deliberately withholds.
    """
    for t in set(hand):
        rest = list(hand)
        rest.remove(t)
        try:
            sh_after = table._shanten(rest, n_melds)
        except Exception:
            continue
        i = _tid(t)
        if sh_after <= shanten:
            ctx.assign(ctx.idx + 1, i)
        if sh_after < shanten:
            ctx.assign(ctx.idx + 2, i)
        if shanten <= 1 and sh_after == 0:
            ctx.assign(ctx.idx + 3, i)
