"""Tensor encoding of the mahjong table for the conventional-DNN baseline.

Design constraint — INFORMATION FAIRNESS with the LLM agent: this encoder
exposes exactly what the LLM's text prompt exposes (own hand, own melds,
every player's visible river and melds, dora indicators, points, winds,
wall count, riichi flags, last discard). It deliberately does NOT feed
engine-computed shanten/ukeire, which the prompt also lacks — otherwise
the comparison would measure feature engineering rather than the agent.

Action space: the engine hands us a list of legal action XML strings, so
actions are mapped to a fixed index (type, key_tile) and everything else
is masked out.  key_tile disambiguates the three chi variants because
each uses a different pair of tiles from hand.
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.tasks.mahjong.shanten import dora_from_indicator

TILE_TYPES = 34
SUITS = "mps"

ACTION_TYPES = ["discard", "riichi", "chi", "pon", "kan", "ron", "tsumo", "skip"]
TYPE_TO_ID = {t: i for i, t in enumerate(ACTION_TYPES)}
ACTION_DIM = len(ACTION_TYPES) * TILE_TYPES      # 8 * 34 = 272

# 15 board planes over the 34 tile types (see class docstring for the list)
N_PLANES = 15
# +4 optional order planes: each seat's river with values = discard index/20
# (later discards score higher). The LLM prompt shows rivers as ORDERED text,
# so adding order restores information parity rather than exceeding it.
N_PLANES_V2 = 19
N_SCALARS = 20
# encoder v3 (2026-08-22, exp23): the COMPLETE public record, zero derived
# features. v1's 15 planes + per-seat river facts (order / tsumogiri /
# riichi-declaration tile / called-away) x4 + per-seat meld type planes
# (chi/pon/kan) x4 + "this opponent's open meld was fed by me" x3 +
# visible-tile count >= k (k=1..4, union of everything on the table).
N_PLANES_V3 = 15 + 16 + 12 + 3 + 4          # = 50
N_SCALARS_V3 = N_SCALARS + 4 + 4 + 1        # + riichi turn x4, discard count x4, wall-turn

_ACT_RE = re.compile(r'type="(\w+)"(?:[^>]*?tile="([^"]+)")?(?:[^>]*?with="([^"]+)")?')


def _tile_to_34_slow(tile: str) -> int:
    tile = tile.replace("*", "").strip()
    val, suit = int(tile[:-1]), tile[-1]
    if suit == "z":
        return 27 + (val - 1)
    return SUITS.index(suit) * 9 + (val - 1)


_T34 = {}
for _v in range(1, 10):
    for _s in SUITS:
        _T34[f"{_v}{_s}"] = _tile_to_34_slow(f"{_v}{_s}")
        _T34[f"{_v}{_s}*"] = _T34[f"{_v}{_s}"]
for _v in range(1, 8):
    _T34[f"{_v}z"] = _tile_to_34_slow(f"{_v}z")
    _T34[f"{_v}z*"] = _T34[f"{_v}z"]


def tile_to_34(tile: str) -> int:
    """'1m'->0 .. '9m'->8, '1p'->9 .., '1s'->18 .., '1z'->27 .. '7z'->33.
    Dict lookup for the ~160k calls/30 games (perf 2026-08-22); falls back
    to parsing for any unusual spelling."""
    try:
        return _T34[tile]
    except KeyError:
        return _tile_to_34_slow(tile)


def action_to_index(action_xml: str) -> Optional[int]:
    """(type, key_tile) -> flat index; None if unparseable."""
    m = _ACT_RE.search(action_xml)
    if not m:
        return None
    a_type, tile, with_tiles = m.group(1), m.group(2), m.group(3)
    if a_type not in TYPE_TO_ID:
        return None
    if a_type == "chi" and with_tiles:
        # the three chi shapes differ in which pair leaves the hand
        key = tile_to_34(with_tiles.split()[0])
    elif tile:
        key = tile_to_34(tile)
    else:
        key = 0
    return TYPE_TO_ID[a_type] * TILE_TYPES + key


def legal_mask(actions: List[str]) -> Tuple[torch.Tensor, Dict[int, str]]:
    """Boolean mask over ACTION_DIM plus index->action-string lookup.

    On a (type, key_tile) collision the FIRST action wins and the clash is
    reported by the caller-visible size mismatch (len(lookup) < len(actions)).
    """
    mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
    lookup: Dict[int, str] = {}
    for a in actions:
        idx = action_to_index(a)
        if idx is None or idx in lookup:
            continue
        mask[idx] = True
        lookup[idx] = a
    return mask, lookup


def _counts_plane(tiles: List[str]) -> torch.Tensor:
    c = [0.0] * TILE_TYPES
    for t in tiles:
        c[tile_to_34(t)] += 1.0
    return torch.tensor(c)


def _counts_list(tiles) -> List[float]:
    c = [0.0] * TILE_TYPES
    for t in tiles:
        c[tile_to_34(t)] += 1.0
    return c


def _presence_row(tiles) -> np.ndarray:
    row = np.zeros(TILE_TYPES, dtype=np.float32)
    for t in tiles:
        row[tile_to_34(t)] = 1.0
    return row


def encode_state(table, player_id: int,
                 with_order: bool = False,
                 variant: str = "v1") -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (planes [N_PLANES, 34], scalars [N_SCALARS]).

    Everything is written from `player_id`'s point of view: opponents are
    indexed by seat offset 1..3 downstream of the player, so the network
    never has to learn absolute seat identities.

    numpy build + one from_numpy (perf 2026-08-22): the per-plane torch
    ops were ~20% of rollout time; values are bit-identical.
    """
    if variant == "v3":
        return _encode_v3(table, player_id)
    planes = np.zeros((N_PLANES_V2 if with_order else N_PLANES, TILE_TYPES),
                      dtype=np.float32)

    hand = table.hands[player_id]
    counts = np.asarray(_counts_list(hand), dtype=np.float32)
    for k in range(4):                       # 0-3: hand count >= k+1
        planes[k] = counts >= (k + 1)

    planes[4] = _presence_row(t for m in table.melds[player_id] for t in m["tiles"])
    for off in range(1, 4):                  # 5-7: opponents' melds
        pid = (player_id + off) % 4
        planes[4 + off] = _presence_row(t for m in table.melds[pid] for t in m["tiles"])

    planes[8] = _presence_row(table.discards[player_id])
    for off in range(1, 4):                  # 9-11: opponents' rivers
        pid = (player_id + off) % 4
        planes[8 + off] = _presence_row(table.discards[pid])

    planes[12] = _presence_row(dora_from_indicator(i) for i in table.dora_indicators)

    if table.last_discard:
        planes[13][tile_to_34(table.last_discard)] = 1.0

    planes[14] = _presence_row(table.furiten_river[player_id])

    if with_order:
        for off in range(4):             # 15-18: rivers with discard order
            pid = (player_id + off) % 4
            for j, t in enumerate(table.discards[pid]):
                planes[15 + off][tile_to_34(t)] = min((j + 1) / 20.0, 1.0)

    s = np.zeros(N_SCALARS, dtype=np.float32)
    for off in range(4):                     # 0-3: points, own seat first
        pid = (player_id + off) % 4
        s[off] = (table.points[pid] - 25000) / 25000.0
    for off in range(4):                     # 4-7: riichi flags
        pid = (player_id + off) % 4
        s[4 + off] = 1.0 if table.riichi[pid] else 0.0
    s[8] = len(table.wall) / 70.0
    s[9] = table.kyotaku / 4.0
    s[10] = len(hand) / 14.0
    s[11] = len(table.melds[player_id]) / 4.0
    s[12 + table.round_wind_idx] = 1.0       # 12-13: round wind
    seat_wind = (player_id - table.dealer) % 4
    s[14 + seat_wind] = 1.0                  # 14-17: seat wind
    s[18] = 1.0 if table.turn == player_id else 0.0
    s[19] = 1.0 if table.last_discarder == (player_id - 1) % 4 else 0.0
    planes = torch.from_numpy(planes)
    s = torch.from_numpy(s)
    return planes, s


def _encode_v3(table, player_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Complete public record (see N_PLANES_V3). Seats are relative: offset 0
    is the player, 1..3 the opponents downstream."""
    P = np.zeros((N_PLANES_V3, TILE_TYPES), dtype=np.float32)
    hand = table.hands[player_id]
    counts = np.asarray(_counts_list(hand), dtype=np.float32)
    for k in range(4):
        P[k] = counts >= (k + 1)
    seen = counts.copy()                       # visible tiles: hand + rivers + melds + dora
    for off in range(4):
        pid = (player_id + off) % 4
        meld_tiles = [t for m in table.melds[pid] for t in m["tiles"]]
        P[4 + off] = _presence_row(meld_tiles)
        P[8 + off] = _presence_row(table.discards[pid])
        if off:
            seen += np.asarray(_counts_list(meld_tiles), dtype=np.float32)
        else:
            seen += np.asarray(_counts_list(meld_tiles), dtype=np.float32)
        seen += np.asarray(_counts_list(t.replace("*", "") for t in table.furiten_river[pid]),
                           dtype=np.float32)
    dora = [dora_from_indicator(i) for i in table.dora_indicators]
    P[12] = _presence_row(dora)
    seen += np.asarray(_counts_list(table.dora_indicators), dtype=np.float32)
    if table.last_discard:
        P[13][tile_to_34(table.last_discard)] = 1.0
    P[14] = _presence_row(table.furiten_river[player_id])
    # 15..30: per-seat river facts (order, tsumogiri, riichi-decl, called)
    for off in range(4):
        pid = (player_id + off) % 4
        base = 15 + 4 * off
        for j, (tile, tsumogiri, rdecl, called, _idx) in enumerate(table.river_events[pid]):
            t = tile_to_34(tile)
            P[base][t] = min((j + 1) / 20.0, 1.0)
            if tsumogiri:
                P[base + 1][t] = 1.0
            if rdecl:
                P[base + 2][t] = 1.0
            if called:
                P[base + 3][t] = 1.0
    # 31..42: per-seat meld types; 43..45: opponent melds fed by me
    for off in range(4):
        pid = (player_id + off) % 4
        for m in table.melds[pid]:
            kind = {"chi": 0, "pon": 1}.get(m["type"], 2)   # kan/ankan/shouminkan -> 2
            for t in m["tiles"]:
                P[31 + 3 * off + kind][tile_to_34(t)] = 1.0
            if off and m.get("from") == player_id:
                for t in m["tiles"]:
                    P[43 + off - 1][tile_to_34(t)] = 1.0
    # 46..49: visible count >= k
    for k in range(4):
        P[46 + k] = seen >= (k + 1)

    s = np.zeros(N_SCALARS_V3, dtype=np.float32)
    for off in range(4):
        pid = (player_id + off) % 4
        s[off] = (table.points[pid] - 25000) / 25000.0
        s[4 + off] = 1.0 if table.riichi[pid] else 0.0
    s[8] = len(table.wall) / 70.0
    s[9] = table.kyotaku / 4.0
    s[10] = len(hand) / 14.0
    s[11] = len(table.melds[player_id]) / 4.0
    s[12 + table.round_wind_idx] = 1.0
    s[14 + (player_id - table.dealer) % 4] = 1.0
    s[18] = 1.0 if table.turn == player_id else 0.0
    s[19] = 1.0 if table.last_discarder == (player_id - 1) % 4 else 0.0
    for off in range(4):
        pid = (player_id + off) % 4
        rt = table.riichi_turn[pid]
        s[20 + off] = 0.0 if rt is None else min((rt + 1) / 20.0, 1.0)
        s[24 + off] = min(table.discard_count[pid] / 20.0, 1.0)
    s[28] = min(sum(table.discard_count) / 70.0, 1.0)
    return torch.from_numpy(P), torch.from_numpy(s)
