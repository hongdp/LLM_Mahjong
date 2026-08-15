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

from typing import Dict, List

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
