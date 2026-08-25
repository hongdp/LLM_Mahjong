"""Mortal-aligned action space (exp41): 46 slots instead of our 374.

Mortal's layout (`libriichi/src/consts.rs`)::

    0..37   discard | kan choice   (34 tile types + 3 red fives)
    37      riichi
    38,39,40  chi low / mid / high
    41      pon
    42      kan (decide)
    43      agari (ron or tsumo -- the winning tile is context-determined)
    44      ryukyoku (kyuushu)
    45      pass

The compression comes from one observation: for every *claim* action the tile
is already determined by context -- when someone discards 5m and you may pon,
the only ponnable tile is that 5m. So pon needs no tile argument at all, and
chi needs only three (which of the sequences containing the called tile).

Our own space is 11 action types x 34 tiles = 374, of which a measured 29%
(109/374) is ever legal in real play -- the rest are structurally unreachable.
The redundancy is harmless for correctness (illegal slots are masked to -inf)
but costs *sample efficiency*: our `pon-5m` and `pon-5p` are separate output
units that each learn from only the pon decisions involving that tile, while
Mortal's single `pon` unit learns from every pon decision at the table.

Testing that sample-efficiency claim is the point of exp41's action arm.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

MORTAL_ACTION_DIM = 46

IDX_RIICHI = 37
IDX_CHI_LOW, IDX_CHI_MID, IDX_CHI_HIGH = 38, 39, 40
IDX_PON = 41
IDX_KAN = 42
IDX_AGARI = 43
IDX_RYUKYOKU = 44
IDX_PASS = 45

_SUITS = {"m": 0, "p": 9, "s": 18}
_AKA_SLOT = {"m": 34, "p": 35, "s": 36}      # 5mr, 5pr, 5sr

# our engine emits `<action type="discard" tile="4p" />` -- the kind lives
# in the type attribute, not the tag name
ACTION_RE = re.compile(r'type="(\w+)"')
TILE_RE = re.compile(r"tile=\"([^\"]+)\"")
# the two tiles we contribute to a chi live in a separate attribute:
# `<action type="chi" tile="7s" with="5s 6s" />`
WITH_RE = re.compile(r'with="([^"]+)"')


def tile_to_slot(tile: str) -> int:
    """Tile string -> 0..36 (34 plain types, then the three red fives)."""
    t = tile.replace("*", "")
    if t[0] == "0":                     # our red-five spelling
        return _AKA_SLOT[t[1]]
    if t[1] == "z":
        return 27 + int(t[0]) - 1
    return _SUITS[t[1]] + int(t[0]) - 1


def _chi_slot(called: str, consumed: List[str]) -> int:
    """Which of chi low/mid/high this is, from the called tile's position.

    low  = called tile is the lowest of the run (we hold the two above)
    mid  = called tile sits in the middle
    high = called tile is the highest
    """
    def rank(x: str) -> int:
        x = x.replace("*", "")
        return 5 if x[0] == "0" else int(x[0])

    c = rank(called)
    others = sorted(rank(x) for x in consumed)
    if not others or len(others) < 2:
        return IDX_CHI_MID
    if c < others[0]:
        return IDX_CHI_LOW
    if c > others[1]:
        return IDX_CHI_HIGH
    return IDX_CHI_MID


def action_to_slot(action_xml: str, *, at_kan_select: bool = False,
                   at_riichi_select: bool = False) -> Optional[int]:
    """Our engine's action XML -> Mortal slot, or None if unmappable.

    Two of Mortal's decisions are two-step where ours are bundled, and both
    flags below select the second step:

    * `at_kan_select` -- the *decision* to kan is slot 42; the follow-up choice
      of which tile to kan reuses the 0..36 tile slots.
    * `at_riichi_select` -- the *declaration* is slot 37; Mortal then picks the
      discard in a separate query (with `riichi_declared[0]` set in the obs).
      Our engine emits `<action type="riichi" tile="3m"/>` with both fused, so
      without this flag every riichi option collides on slot 37.
    """
    m = ACTION_RE.search(action_xml)
    if not m:
        return None
    kind = m.group(1)
    tm = TILE_RE.search(action_xml)
    tile = tm.group(1) if tm else None

    if kind in ("discard", "discard0"):
        return tile_to_slot(tile) if tile else None
    if kind in ("riichi", "riichi0"):
        return tile_to_slot(tile) if (at_riichi_select and tile) else IDX_RIICHI
    if kind == "pon":
        return IDX_PON
    if kind in ("kan", "ankan", "kakan", "daiminkan"):
        return tile_to_slot(tile) if (at_kan_select and tile) else IDX_KAN
    if kind in ("ron", "tsumo"):
        return IDX_AGARI
    if kind == "kyuushu":
        return IDX_RYUKYOKU
    if kind == "skip":
        return IDX_PASS
    if kind == "chi":
        wm = WITH_RE.search(action_xml)
        consumed = wm.group(1).split() if wm else []
        if tile and len(consumed) >= 2:
            return _chi_slot(tile, consumed[:2])
        return IDX_CHI_MID
    return None


def legal_mask_46(actions: List[str], *, at_kan_select: bool = False,
                  at_riichi_select: bool = False
                  ) -> Tuple[List[bool], Dict[int, str]]:
    """(mask, slot -> action XML) for one decision.

    Collisions are real and expected: several of our XML actions can map to one
    Mortal slot (e.g. two different ankan candidates both land on slot 42 when
    not at_kan_select). We keep the FIRST and report the rest through
    `collisions()` so the caller can decide -- exp41 uses the kan-select
    two-step to resolve the only case that matters in practice.
    """
    mask = [False] * MORTAL_ACTION_DIM
    lookup: Dict[int, str] = {}
    for a in actions:
        s = action_to_slot(a, at_kan_select=at_kan_select,
                           at_riichi_select=at_riichi_select)
        if s is None:
            continue
        if not mask[s]:
            mask[s] = True
            lookup[s] = a
    return mask, lookup


def collisions(actions: List[str], *, at_kan_select: bool = False,
               at_riichi_select: bool = False) -> Dict[int, List[str]]:
    """Slots that more than one of `actions` maps to (diagnostic for tests)."""
    by: Dict[int, List[str]] = {}
    for a in actions:
        s = action_to_slot(a, at_kan_select=at_kan_select,
                           at_riichi_select=at_riichi_select)
        if s is None:
            continue
        by.setdefault(s, []).append(a)
    return {k: v for k, v in by.items() if len(v) > 1}
