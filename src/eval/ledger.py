"""Append-only match ledger (design D3/D4).

One row per HANCHAN (or per deal), never per pair: the aggregation unit
is a fitting choice, and v1 lost that choice by archiving only the
duplicate-pair sum. Rows carry every surface's fingerprint, so a rule
change EXCLUDES rows at fit time instead of invalidating history.

Match ids are deterministic in (driver, unit, seed, seats, temps), so
re-running the same configuration is idempotent — replaying a shard after
a crash can never double-count it.
"""

import hashlib
import json
import os
import time

from src.eval.fingerprints import all_fingerprints

ROOT = "experiments/rating"
LEDGER = f"{ROOT}/matches.jsonl"


def match_id(driver, unit, seed, seats, temps):
    key = json.dumps([driver, unit, int(seed), list(seats),
                      [round(float(t), 4) for t in temps]], sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def row(driver, unit, seed, wall, seats, temps, points, uma, placements,
        n_deals, busted, fp=None, extra=None):
    r = {"mid": match_id(driver, unit, seed, seats, temps),
         "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "driver": driver,
         "unit": unit, "seed": int(seed), "wall": int(wall),
         "seats": list(seats), "temps": [float(t) for t in temps],
         "points": [int(p) for p in points],
         "uma": [int(u) for u in uma],
         "placements": [int(p) for p in placements],
         "n_deals": int(n_deals), "busted": bool(busted),
         "fp": fp or all_fingerprints()}
    if extra:
        r.update(extra)
    return r


def existing_ids(path=LEDGER):
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path) as f:
        for line in f:
            try:
                out.add(json.loads(line)["mid"])
            except Exception:
                continue
    return out


def append(rows, path=LEDGER, seen=None):
    """Returns (written, skipped_duplicates)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seen = existing_ids(path) if seen is None else seen
    w = s = 0
    with open(path, "a") as f:
        for r in rows:
            if r["mid"] in seen:
                s += 1
                continue
            seen.add(r["mid"])
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
            w += 1
    return w, s


def read(path=LEDGER, require_fp=None, unit=None, entities=None):
    """Load rows, filtered by fingerprint set / unit / participating
    entities. `require_fp` maps surface -> expected value; a surface left
    out is not required to match (so an encoder-only change need not
    exclude engine-identical history)."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if unit and r["unit"] != unit:
                continue
            if require_fp and any(r["fp"].get(k) != v
                                  for k, v in require_fp.items()):
                continue
            if entities and not set(r["seats"]) & set(entities):
                continue
            out.append(r)
    return out


def summary(rows):
    ent, pairs = {}, {}
    for r in rows:
        for s in r["seats"]:
            ent[s] = ent.get(s, 0) + 1
        uniq = sorted(set(r["seats"]))
        for i, a in enumerate(uniq):
            for b in uniq[i + 1:]:
                pairs[(a, b)] = pairs.get((a, b), 0) + 1
    return {"matches": len(rows), "entities": len(ent),
            "seat_slots": ent, "pairs": len(pairs), "pair_counts": pairs}
