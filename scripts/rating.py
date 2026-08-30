"""Rating system v2 CLI (design_rating_system_v2.md).

  rating register --path P --name N [--temps 0,1] [--lineage L]
  rating import   [--pattern GLOB]        # v1 archives -> ledger
  rating schedule [--budget SECONDS] [--targets a:b,...]
  rating play     [--plan FILE | --table a,b,c,d --walls N]
  rating fit      [--pin LABEL]
  rating board    [--sort pl|pt|top]

Everything derives from experiments/rating/matches.jsonl. Delete every
table and the board rebuilds; that is the point of D3.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import arena, fits, importers, ledger, registry, schedule  # noqa: E402
from src.eval.fingerprints import all_fingerprints                       # noqa: E402

ROOT = "experiments/rating"
CURSOR = f"{ROOT}/seed_cursor.json"
TABLES = f"{ROOT}/tables.json"


def next_seed(n):
    """Hand-picked seed0 values were a v1 footgun (collisions replay the
    same match, which the ledger then dedupes away silently). One cursor."""
    os.makedirs(ROOT, exist_ok=True)
    cur = json.load(open(CURSOR))["next"] if os.path.exists(CURSOR) else 60000000
    json.dump({"next": cur + int(n)}, open(CURSOR, "w"))
    return cur


def cmd_register(a):
    eids = registry.add(a.path, a.name,
                        temperatures=[float(t) for t in a.temps.split(",")],
                        kind=a.kind, lineage=a.lineage)
    ents = registry.entities()
    for e in eids:
        print(f"{e}  {ents[e]['label']}")


def cmd_import(a):
    rows, info = importers.import_v1_matches(
        a.pattern,
        pool_files=["experiments/elo_league/hanchan/anchors.json",
                    "experiments/elo_league/hanchan/anchors_T0.json"],
        history_files=["experiments/elo_league/hanchan/history.jsonl"])
    w, s = ledger.append(rows)
    print(json.dumps({**info, "rows_built": len(rows), "written": w,
                      "skipped_dupes": s}, indent=1))


def _current_ratings(rows, pin):
    try:
        return fits.fit_pl(rows, pin=pin)
    except Exception:
        return {}


def cmd_schedule(a):
    ents = list(registry.entities())
    rows = ledger.read(require_fp=all_fingerprints(), unit="hanchan")
    targets = []
    for t in (a.targets or "").split(",") if a.targets else []:
        if ":" in t:
            x, y = t.split(":")
            targets.append((registry.resolve(x)["id"], registry.resolve(y)["id"]))
    slow = {}
    for eid, e in registry.entities().items():
        m = registry.models()[e["model"]]
        if m["kind"] != "dnn":
            slow[eid] = 6.7            # measured Mortal-bridge throughput
    plan, info = schedule.plan(ents, _current_ratings(rows, a.pin), rows=rows,
                               targets=targets, budget_seconds=a.budget,
                               walls_per_table=a.walls,
                               throughput_per_min=a.throughput,
                               cost_per_min=slow)
    lbl = registry.entities()
    for p in plan:
        print(f"{'FORCED' if p.get('forced') else 'info  '} walls={p['walls']:>3} "
              + " | ".join(lbl[e]["label"] for e in p["table"]))
    print(json.dumps(info, indent=1))
    json.dump(plan, open(a.out, "w"), indent=1)
    print(f"saved {a.out}")


def cmd_play(a):
    if a.plan:
        plan = json.load(open(a.plan))
    else:
        plan = [{"table": [registry.resolve(x)["id"]
                           for x in a.table.split(",")], "walls": a.walls}]
    lbl = registry.entities()
    seen = ledger.existing_ids()
    total = 0
    for i, p in enumerate(plan):
        s0 = next_seed(p["walls"])
        t0 = time.time()
        rows, rate = arena.play_table(list(p["table"]), p["walls"], s0,
                                      device=a.device, parallel=a.parallel)
        w, sk = ledger.append(rows, seen=seen)
        total += w
        print(f"[{i+1}/{len(plan)}] {' | '.join(lbl[e]['label'] for e in p['table'])}"
              f"  {len(rows)} matches @ {rate}/min  (+{w} rows, {sk} dup, "
              f"{time.time()-t0:.0f}s)", flush=True)
    print(f"ledger +{total} rows")


def cmd_fit(a):
    rows = ledger.read(require_fp=all_fingerprints(), unit="hanchan")
    if not rows:
        raise SystemExit("ledger has no rows matching the current fingerprints")
    ents = registry.entities()
    pin = None
    if a.pin:
        pin = registry.resolve(a.pin)["id"]
    else:
        for e in ents:
            if ents[e]["label"] in ("bc_cnn@T0", "bc49@T0"):
                pin = e
                break
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "fingerprints": all_fingerprints(), "unit": "hanchan",
           "n_matches": len(rows), "pin": ents[pin]["label"] if pin else None,
           "pl": fits.fit_pl(rows, pin=pin),
           "pt": fits.fit_pt(rows, pin=pin),
           "placement": fits.placement_table(rows),
           "h2h": {f"{x}|{y}": v for (x, y), v in fits.head_to_head(rows).items()}}
    json.dump(out, open(TABLES, "w"), indent=1)
    print(f"fitted {len(rows)} matches over {len(out['pl'])} entities "
          f"-> {TABLES}")
    _print_board(out, "pl")


def _print_board(t, sort):
    ents = registry.entities()
    key = {"pl": lambda e: -t["pl"][e]["rating"],
           "pt": lambda e: -t["pt"][e]["rating"],
           "top": lambda e: -t["placement"][e]["top_rate"]}[sort]
    order = sorted(t["pl"], key=key)
    print(f"\n{'#':>2} {'entity':<22} {'Elo(PL)':>14} {'pt/半庄':>13} "
          f"{'首位率':>8} {'四位率':>8} {'平均顺位':>9} {'n':>6}")
    for i, e in enumerate(order, 1):
        pl, pt, pc = t["pl"][e], t["pt"][e], t["placement"][e]
        print(f"{i:>2} {ents[e]['label']:<22} {pl['rating']:8.1f}±{pl['se']:<5.1f} "
              f"{pt['rating']:+8.2f}±{pt['se']:<4.2f} {pc['top_rate']*100:7.1f}% "
              f"{pc['last_rate']*100:7.1f}% {pc['mean_plc']:9.3f} {pc['n']:>6}")


def cmd_board(a):
    if not os.path.exists(TABLES):
        raise SystemExit("no fit yet — run `rating fit`")
    _print_board(json.load(open(TABLES)), a.sort)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--path", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--temps", default="0")
    r.add_argument("--kind", default="dnn")
    r.add_argument("--lineage", default=None)
    r.set_defaults(fn=cmd_register)
    i = sub.add_parser("import")
    i.add_argument("--pattern", default="experiments/elo_league/hanchan/matches/*.json")
    i.set_defaults(fn=cmd_import)
    s = sub.add_parser("schedule")
    s.add_argument("--budget", type=float, default=1800)
    s.add_argument("--walls", type=int, default=25)
    s.add_argument("--targets", default=None, help="labelA:labelB,...")
    s.add_argument("--throughput", type=float, default=600.0)
    s.add_argument("--pin", default=None)
    s.add_argument("--out", default=f"{ROOT}/plan.json")
    s.set_defaults(fn=cmd_schedule)
    p = sub.add_parser("play")
    p.add_argument("--plan", default=None)
    p.add_argument("--table", default=None, help="four labels, comma separated")
    p.add_argument("--walls", type=int, default=25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--parallel", type=int, default=20)
    p.set_defaults(fn=cmd_play)
    f = sub.add_parser("fit")
    f.add_argument("--pin", default=None)
    f.set_defaults(fn=cmd_fit)
    b = sub.add_parser("board")
    b.add_argument("--sort", default="pl", choices=("pl", "pt", "top"))
    b.set_defaults(fn=cmd_board)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
