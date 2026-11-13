"""Terminal-heavy opening probe: deals where some seat starts with K >= 8 kinds of terminals/honors (rejection
sampling on the exp129 deal generator, rules untouched), four seats the same net (probe_loop.play_probe_game),
recording only the qualifying seats: kyuushu declared, yaochuu share of the first 6 discards, kokushi shanten
after each own discard, kokushi tenpai / win, overall win and point delta.
usage (repo root):  kokushi_probe.py scan SEEDS.json
                    kokushi_probe.py play SEEDS.json OUT.json name=path[@T[@nokyu]] ...   (T defaults to 0.0)
@nokyu (diagnostic only, not a training rule): the kyuushu kyuuhai option is removed from every legal-action list,
to see whether a net that always declares the abortive draw would otherwise play toward kokushi."""
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.getcwd())
from src.tasks.mahjong.table import PyMahjongTable  # noqa: E402

SEED0 = 70_000_000
TARGET = {8: 1500, 9: 1500, 10: 600}         # 10 = ">= 10"
TILE_RE = re.compile(r'tile="([^"]+)"')
TYPE_RE = re.compile(r'type="(\w+)"')
WIN_SEG_RE = re.compile(r"玩家(\d) (?:荣和|自摸|抢杠)[^|]*\| \d+番\d+符 \| ([^|;]*)")


def yao(x):
    return x[1] == "z" or x[0] in "19"


def kinds(h):
    return len({x for x in h if yao(x)})


def kokushi_shanten(h):
    ks = {x for x in h if yao(x)}
    pair = any(sum(1 for y in h if y == x) >= 2 for x in ks)
    return 13 - len(ks) - (1 if pair else 0)


def scan(out):
    got, s, res = {k: 0 for k in TARGET}, SEED0, []
    while any(got[k] < TARGET[k] for k in TARGET):
        random.seed(s)
        t = PyMahjongTable(randomize_round=True)
        seats = []
        for p in range(4):
            k = min(kinds(t.hands[p]), 10)
            if k >= 8 and got[k] < TARGET[k]:
                seats.append([p, kinds(t.hands[p])])
                got[k] += 1
        if seats:
            res.append({"seed": s, "seats": seats})
        s += 1
    json.dump({"scanned": s - SEED0, "deals": res}, open(out, "w"))
    print(f"scanned {s - SEED0} deals -> {len(res)} deals, seat counts {got}")


def play_slice(args):
    path, temp, deals, dev, nokyu = args
    import torch
    torch.set_num_threads(1)
    if nokyu:
        _orig = PyMahjongTable.get_legal_actions

        def _no_kyuushu(self, pid):
            acts = _orig(self, pid)
            kept = [a for a in acts if 'type="kyuushu"' not in a]
            return kept or acts
        PyMahjongTable.get_legal_actions = _no_kyuushu
    from src.agents.dnn.checkpoint import load_dnn
    from src.eval.probe_loop import play_probe_game
    net = load_dnn(path, dev).eval()
    rows = []
    for d in deals:
        watch = {p: {"K": k, "disc": [], "ks": [], "kyuushu_legal": False, "kyuushu": False, "first": True}
                 for p, k in d["seats"]}
        start = {}

        def before(table, pid, actions):
            if not start:
                start["pts"] = list(table.points)
            w = watch.get(pid)
            if w is not None and w["first"]:
                w["first"] = False
                w["kyuushu_legal"] = any('type="kyuushu"' in a for a in actions)
                w["K_draw"] = kinds(table.hands[pid])
            return None

        def after(ctx, table, pid, actions, a):
            w = watch.get(pid)
            if w is None:
                return
            ty = (TYPE_RE.search(a) or [None, None])[1]
            if ty == "kyuushu":
                w["kyuushu"] = True
            m = TILE_RE.search(a)
            if ty in ("discard", "riichi") and m:
                tile = m.group(1)
                h = list(table.hands[pid])
                if tile in h:
                    h.remove(tile)
                w["disc"].append(tile)
                w["ks"].append(kokushi_shanten(h))

        with torch.no_grad():
            table = play_probe_game(net, d["seed"], temp, device=dev, before_turn=before, after_turn=after)
        r = table.result_summary or ""
        wins = {int(m.group(1)): m.group(2) for m in WIN_SEG_RE.finditer(r)}
        for p, w in watch.items():
            d6 = w["disc"][:6]
            rows.append({"seed": d["seed"], "seat": p, "K": w["K"], "K_draw": w.get("K_draw"),
                         "kyuushu_legal": w["kyuushu_legal"],
                         "kyuushu": w["kyuushu"], "yao6": sum(yao(x) for x in d6), "n6": len(d6),
                         "ks_min": min(w["ks"]) if w["ks"] else None, "ks6": w["ks"][5] if len(w["ks"]) >= 6 else None,
                         "won": p in wins, "kokushi_win": p in wins and "Kokushi" in wins[p],
                         "delta": table.points[p] - start.get("pts", table.points)[p]})
    return rows


def play(seeds_path, out, specs):
    import torch.multiprocessing as mp
    deals = json.load(open(seeds_path))["deals"]
    res = json.load(open(out)) if os.path.exists(out) else {}
    nw = int(os.environ.get("PROBE_WORKERS", "8"))
    ctx = mp.get_context("spawn")
    for spec in specs:
        name, rest = spec.split("=", 1)
        path, *opt = rest.split("@")
        t = opt[0] if opt else ""
        nokyu = "nokyu" in opt[1:]
        if name in res:
            continue
        t0 = time.time()
        parts = [(path, float(t or 0.0), deals[i::nw], "cuda", nokyu) for i in range(nw)]
        with ctx.Pool(nw) as pool:
            rows = [r for part in pool.map(play_slice, parts) for r in part]
        res[name] = {"path": path, "temperature": float(t or 0.0), "nokyu": nokyu, "rows": rows,
                     "sec": round(time.time() - t0)}
        json.dump(res, open(out + ".tmp", "w"))
        os.replace(out + ".tmp", out)
        print(f"{name}: {len(rows)} seat-deals in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "scan":
        scan(sys.argv[2])
    else:
        play(sys.argv[2], sys.argv[3], sys.argv[4:])
