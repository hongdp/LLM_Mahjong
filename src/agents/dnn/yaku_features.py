"""Value-distance profile of a hand (exp11's shared feature).

User's key design insight: parameterize yaku features by VALUE, not by
identity. A feature like "shanten-ish distance to any hand worth >= X"
lets a critic (or a potential) learn the mapping value x P(complete|d)
from the yaku families the policy already completes (toitoi/honitsu at
2-4 han) and TRANSFER it to rarer, richer families (chinitsu at 6 han),
because the rare hand activates the same continuous feature dimension
instead of a never-seen one-hot.

Distances here are cheap monotone PROXIES (tiles-to-replace / pairs
missing), not exact yaku shanten — sufficient for a feature, documented
as such. Values are rough average completed-hand points for the family
alone; dora is not modelled.
"""

from typing import Dict, List, Tuple

from src.tasks.mahjong.shanten import TileEfficiency, pad_for_melds

_TE = TileEfficiency()

TERMINAL_HONOR = {f"1{s}" for s in "mps"} | {f"9{s}" for s in "mps"} | \
                 {f"{v}z" for v in range(1, 8)}
DRAGONS = {"5z", "6z", "7z"}

# family -> rough standalone completed value (points)
FAMILY_VALUE = {
    "riichi_menzen": 5000,   # riichi + typical adds (tsumo/ippatsu/ura)
    "tanyao": 2000,
    "yakuhai": 2000,
    "chiitoi": 3200,
    "toitoi": 4000,
    "honitsu": 5000,
    "chinitsu": 9000,
}

VALUE_BUCKETS = (2000, 4000, 5000, 8000)
MAX_D = 8.0


def family_distances(hand: List[str], n_melds: int,
                     closed: bool) -> Dict[str, float]:
    """Monotone distance proxies per family; smaller = closer."""
    counts: Dict[str, int] = {}
    for t in hand:
        counts[t] = counts.get(t, 0) + 1
    d: Dict[str, float] = {}

    try:
        base_sh = _TE.calculate_shanten(pad_for_melds(hand, n_melds))
    except Exception:
        base_sh = 6
    # riichi needs a closed hand
    d["riichi_menzen"] = base_sh if closed else MAX_D

    off_tanyao = sum(1 for t in hand if t in TERMINAL_HONOR)
    d["tanyao"] = max(base_sh, off_tanyao / 2.0)

    best_dragon = max((counts.get(t, 0) for t in DRAGONS), default=0)
    d["yakuhai"] = max(base_sh, 3 - best_dragon if best_dragon < 3 else 0)

    pairs = sum(1 for v in counts.values() if v >= 2)
    d["chiitoi"] = (6 - min(pairs, 6)) if (closed and n_melds == 0) else MAX_D

    trip = sum(1 for v in counts.values() if v >= 3) + n_melds
    d["toitoi"] = max(4 - trip - min(pairs, 1), 0) + max(base_sh - 1, 0) * 0.5

    for fam, keep_honors in (("honitsu", True), ("chinitsu", False)):
        best = MAX_D
        for suit in "mps":
            off = sum(1 for t in hand
                      if t[-1] != suit and not (keep_honors and t[-1] == "z"))
            best = min(best, float(off))
        d[fam] = best

    return d


def value_distance_profile(hand: List[str], n_melds: int,
                           closed: bool) -> List[float]:
    """For each value bucket X: min distance to any family worth >= X,
    normalized to [0,1] (1 = far/unreachable). Monotone non-decreasing
    in X by construction."""
    dists = family_distances(hand, n_melds, closed)
    out = []
    for x in VALUE_BUCKETS:
        best = MAX_D
        for fam, val in FAMILY_VALUE.items():
            if val >= x:
                best = min(best, dists[fam])
        out.append(min(best, MAX_D) / MAX_D)
    return out


# ---------------------------------------------------------------------------
# Hazard-critic feature extractors (exp11+): per-family EXACT distance and
# ukeire, so a SHARED completion-hazard head can generalize across families
# by dynamics (d, u, resources) instead of identity. Kokushi and suuankou are
# both worth 32000 — what tells them apart is the SHAPE of (d, u): kokushi
# runs on a narrow fixed set of missing orphan types; suuankou runs on
# pairs-to-concealed-triplets conversion.
# ---------------------------------------------------------------------------

ORPHANS = sorted(TERMINAL_HONOR)          # the 13 kokushi tile types


def kokushi_distance(hand: List[str],
                     visible_counts: Dict[str, int] = None) -> Tuple[float, float]:
    """(shanten, ukeire_tiles) for kokushi musou.

    shanten = 13 - distinct_orphan_types - (1 if any orphan pair) ; -1 = win.
    ukeire counts REMAINING copies of missing orphan types (4 minus what we
    hold and what is visible elsewhere), plus pair-forming copies when no
    pair is held yet.
    """
    counts: Dict[str, int] = {}
    for t in hand:
        counts[t] = counts.get(t, 0) + 1
    vis = visible_counts or {}
    types = [t for t in ORPHANS if counts.get(t, 0) >= 1]
    has_pair = any(counts.get(t, 0) >= 2 for t in ORPHANS)
    shanten = 13 - len(types) - (1 if has_pair else 0)
    missing = [t for t in ORPHANS if counts.get(t, 0) == 0]
    ukeire = sum(max(4 - vis.get(t, 0), 0) for t in missing)
    if not has_pair:
        ukeire += sum(max(4 - counts.get(t, 0) - vis.get(t, 0), 0)
                      for t in types)
    return float(shanten), float(ukeire)


def suuankou_distance(hand: List[str], n_melds: int,
                      visible_counts: Dict[str, int] = None) -> Tuple[float, float]:
    """(distance, ukeire_tiles) for suuankou (four CONCEALED triplets).

    Any open meld kills the family (returns MAX_D, 0): the categorical
    "call -> this yakuman dies instantly" fact lives HERE as a feature,
    which is what lets gamma*V(s')-V(s) punish the call immediately.
    distance = 4 - concealed_triplets - min(pairs,needed) heuristic on the
    standard toitoi-like ladder; ukeire = third copies of held pairs.
    """
    if n_melds > 0:
        return MAX_D, 0.0
    counts: Dict[str, int] = {}
    for t in hand:
        counts[t] = counts.get(t, 0) + 1
    vis = visible_counts or {}
    trips = sum(1 for v in counts.values() if v >= 3)
    pairs = sum(1 for v in counts.values() if v == 2)
    need = 4 - trips
    # each missing triplet must come from upgrading a pair (or drawing pairs)
    dist = float(max(need - pairs, 0) + need)
    ukeire = sum(max(2 - vis.get(t, 0), 0)
                 for t, v in counts.items() if v == 2)
    return dist, float(ukeire)


HAZARD_FAMILIES = list(FAMILY_VALUE) + ["kokushi", "suuankou"]
HAZARD_VALUE = dict(FAMILY_VALUE, kokushi=32000, suuankou=32000)

# ---------------------------------------------------------------------------
# Completion labels (exp11 A2): which hazard families did each seat actually
# complete this game? Parsed from PyMahjongTable.result_summary, whose yaku
# names are the scoring lib's str(yaku). These are supervised BCE targets —
# whether a hand completed is settled fact, exempt from on-policy limits.
# ---------------------------------------------------------------------------

import re as _re

_WIN_SEAT_RE = _re.compile(r"玩家(\d)\s*(?:自摸|荣和|抢杠)")

# normalized (lowercase, spaces stripped) substring -> family
_NAME_TO_FAMILY = {
    "riichi": "riichi_menzen",       # covers Daburu Riichi too
    "tanyao": "tanyao",
    "yakuhai": "yakuhai",
    "chiitoitsu": "chiitoi",
    "toitoi": "toitoi",
    "honitsu": "honitsu",
    "chinitsu": "chinitsu",
    "kokushi": "kokushi",            # "Kokushi Musou"
    "suuankou": "suuankou",          # "Suu Ankou" / tanki variant
}


def completion_labels(result_summary: str) -> Dict[int, List[float]]:
    """Per-seat binary vector over HAZARD_FAMILIES for one finished game.

    Winners get 1.0 for every family named in THEIR yaku segment (double
    ron: segments are ';'-joined, each parsed separately); losers and all
    seats of a draw get zeros. The 放铳:玩家N mention never matches because
    the seat regex requires a win verb right after the seat number.
    """
    idx = {f: i for i, f in enumerate(HAZARD_FAMILIES)}
    out = {p: [0.0] * len(HAZARD_FAMILIES) for p in range(4)}
    if not result_summary:
        return out
    for part in result_summary.split(";"):
        m = _WIN_SEAT_RE.search(part)
        if not m:
            continue
        pid = int(m.group(1))
        norm = part.lower().replace(" ", "")
        for pat, fam in _NAME_TO_FAMILY.items():
            if pat in norm:
                out[pid][idx[fam]] = 1.0
    return out


def hazard_features(hand: List[str], n_melds: int, closed: bool,
                    turns_left: float,
                    visible_counts: Dict[str, int] = None) -> List[List[float]]:
    """Per-family [d/MAX_D, u/40, closed_ok, value/32000, turns_left/18].

    One row per family, all through the SAME hazard head downstream. The
    family's VALUE is an injected feature (rule knowledge), never a
    regression target — a yakuman's 32000 never has to survive advantage
    clipping to reach the critic.
    """
    base = family_distances(hand, n_melds, closed)
    kd, ku = kokushi_distance(hand, visible_counts)
    sd, su = suuankou_distance(hand, n_melds, visible_counts)
    rows = []
    for fam in HAZARD_FAMILIES:
        if fam == "kokushi":
            d, u, ok = kd, ku, closed and n_melds == 0
        elif fam == "suuankou":
            d, u, ok = sd, su, closed and n_melds == 0
        else:
            d, ok = base[fam], base[fam] < MAX_D
            u = 8.0        # generic families: ukeire proxy TBD, constant
        rows.append([min(max(d, -1.0), MAX_D) / MAX_D,
                     min(u, 40.0) / 40.0,
                     1.0 if ok else 0.0,
                     HAZARD_VALUE[fam] / 32000.0,
                     min(turns_left, 18.0) / 18.0])
    return rows
