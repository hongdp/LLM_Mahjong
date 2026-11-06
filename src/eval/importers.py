"""Bring v1 archives into the v2 ledger.

A v1 2v2 duplicate match is just a table whose four seats hold two
entities twice — a perfectly good observation for the Plackett-Luce and
pt fits, so 30k+ archived hanchan do not have to be replayed. Only
archives carrying per-game primary records can come across; the older
aggregate-only files (duplicate-pair sums) are unrecoverable by
construction, which is the whole reason D3 exists.
"""

import glob
import json
import os

from src.eval import ledger, registry
from src.eval.fingerprints import all_fingerprints

UMA_RANK = [15000, 5000, -5000, -15000]


def _name_map(pool_files, history_files):
    """path -> human name, from the anchor pools and the rating history."""
    out = {}
    for fn in pool_files:
        if not os.path.exists(fn):
            continue
        for n, a in json.load(open(fn))["anchors"].items():
            out.setdefault(a["path"], n)
    for fn in history_files:
        if not os.path.exists(fn):
            continue
        for line in open(fn):
            try:
                r = json.loads(line)
            except Exception:
                continue
            lbl = (r.get("label") or "").replace("H6T0_", "").replace("H6_", "")
            lbl = lbl.replace("E6_", "").replace("_T0", "")
            if r.get("ckpt") and lbl:
                out.setdefault(r["ckpt"], lbl)
    return out


def import_v1_matches(pattern, pool_files=(), history_files=(), lineage=None,
                      dry_run=False):
    names = _name_map(list(pool_files), list(history_files))
    fp = all_fingerprints()
    rows, skipped, files = [], 0, 0
    for fn in sorted(glob.glob(pattern)):
        blob = json.load(open(fn))
        games = blob.get("games")
        if not games or blob.get("unit") != "hanchan":
            skipped += 1
            continue
        files += 1
        pa, pb = blob["path_a"], blob["path_b"]
        na = names.get(pa) or os.path.basename(pa).split(".")[0]
        nb = names.get(pb) or os.path.basename(pb).split(".")[0]
        ta, tb = float(blob.get("temp_a", 1.0)), float(blob.get("temp_b", 1.0))
        if dry_run:
            continue
        ea = registry.add(pa, na, temperatures=(ta,), lineage=lineage)[0]
        eb = registry.add(pb, nb, temperatures=(tb,), lineage=lineage)[0]
        for g in games:
            aseats = set(g["a_seats"])
            seats = [ea if s in aseats else eb for s in range(4)]
            temps = [ta if s in aseats else tb for s in range(4)]
            uma, plc = g["uma"], g["placements"]
            pts = [uma[s] + 25000 - UMA_RANK[plc[s] - 1] for s in range(4)]
            rows.append(ledger.row("v1-import/" + blob.get("driver", "vec-pair"),
                                   "hanchan", g["seed"], g["wall"], seats,
                                   temps, pts, uma, plc, g["n_deals"],
                                   g.get("busted", False), fp,
                                   {"src": os.path.basename(fn)}))
    return rows, {"files_imported": files, "files_skipped_no_primary": skipped}
