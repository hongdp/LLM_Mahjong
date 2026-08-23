"""Score a Majsoul session log written by scripts/serve_mjai_bot.py.

The log is JSONL: {"t", "kind": start|in|in_batch|out|error, "data"}.
Per game (a `start` record = MahjongCopilot's init_bot) we collect the
kyoku stream (`start_kyoku` scores), our decisions (`out` reactions) and,
if the installer's game_state.py patch is in place, the liqi round results
(`end_kyoku` with hules / delta scores) and the final table (`end_game`).

Reported: games, avg placement, per-kyoku win / deal-in / riichi / call
rates, mean score delta per kyoku, and a per-game table — the same style
statistics the arena uses, so Majsoul results can be compared with
self-play (riichi 0.24 / calls ~0.4 for exp17-C).

Usage:
  python scripts/analyze_majsoul_session.py experiments/majsoul_sessions/mjai_session.jsonl
"""

import argparse
import json
import math
import statistics
from collections import Counter


def g(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def analyze(path):
    games, game, kyoku = [], None, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind, data = rec["kind"], rec["data"]
            if kind == "start":
                game = {"seat": int(data["seat"]), "kyokus": [], "decisions": Counter(),
                        "final": None, "rank": None, "t": rec["t"]}
                games.append(game)
                kyoku = None
                continue
            if game is None:
                continue
            msgs = data if kind == "in_batch" else ([data] if kind == "in" else [])
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                t = m.get("type")
                if t == "start_kyoku":
                    kyoku = {"bakaze": m.get("bakaze"), "kyoku": m.get("kyoku"), "honba": m.get("honba"),
                             "scores": list(m.get("scores", [])), "result": None, "delta": None,
                             "won": False, "dealt_in": False, "tsumo": False, "riichi": False,
                             "called": False}
                    game["kyokus"].append(kyoku)
                elif t == "reach" and kyoku and m.get("actor") == game["seat"]:
                    kyoku["riichi"] = True
                elif t in ("chi", "pon", "daiminkan") and kyoku and m.get("actor") == game["seat"]:
                    kyoku["called"] = True
                elif t == "end_kyoku" and kyoku and m.get("liqi_data"):
                    ld = m["liqi_data"]
                    name = m.get("liqi_name", "")
                    me = game["seat"]
                    deltas = g(ld, "delta_scores", "deltaScores")
                    if deltas is None and "scores" in ld and isinstance(ld["scores"], list) \
                            and ld["scores"] and isinstance(ld["scores"][0], dict):      # NoTile
                        deltas = [0] * 4
                        for s in ld["scores"]:
                            seat = g(s, "seat", default=0)
                            ds = g(s, "delta_scores", "deltaScores", default=[])
                            if ds:
                                deltas = ds if len(ds) == 4 else deltas
                    hules = g(ld, "hules", default=[]) or []
                    winners = [g(h, "seat", default=-1) for h in hules]
                    kyoku["result"] = ("hora" if hules else
                                       "ryukyoku" if "NoTile" in name else "abort")
                    kyoku["won"] = me in winners
                    kyoku["tsumo"] = any(g(h, "zimo", default=False) for h in hules if g(h, "seat") == me)
                    if deltas and len(deltas) == 4:
                        kyoku["delta"] = int(deltas[me])
                        if hules and not any(g(h, "zimo", default=False) for h in hules) \
                                and me not in winners and int(deltas[me]) < 0:
                            # ron: exactly one payer -> the one with the negative delta
                            payers = [i for i in range(4) if int(deltas[i]) < 0]
                            kyoku["dealt_in"] = payers == [me]
                elif t == "end_game":
                    res = g(m.get("liqi_data") or {}, "result", default={}) or {}
                    players = g(res, "players", default=[]) or []
                    if players:
                        game["final"] = {int(g(p, "seat", default=i)): g(p, "total_point", "totalPoint",
                                                                           "part_point_1", "partPoint1", default=0)
                                         for i, p in enumerate(players)}
                        order = sorted(game["final"], key=lambda s: -game["final"][s])
                        if game["seat"] in order:
                            game["rank"] = order.index(game["seat"]) + 1
            if kind == "out" and isinstance(data, dict):
                game["decisions"][data.get("type")] += 1
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--json", default=None, help="also dump per-game records here")
    a = ap.parse_args()
    games = analyze(a.log)
    kyokus = [k for gm in games for k in gm["kyokus"]]
    scored = [k for k in kyokus if k["result"] is not None]
    ranks = [gm["rank"] for gm in games if gm["rank"]]
    dec = Counter()
    for gm in games:
        dec.update(gm["decisions"])

    def rate(key, pool):
        return (sum(1 for k in pool if k[key]) / len(pool)) if pool else float("nan")

    print(f"log: {a.log}")
    print(f"games: {len(games)}   kyokus: {len(kyokus)}   kyokus with result: {len(scored)}")
    if ranks:
        mu = statistics.mean(ranks)
        se = (statistics.pstdev(ranks) / math.sqrt(len(ranks))) if len(ranks) > 1 else float("nan")
        print(f"avg placement: {mu:.3f} ± {se:.3f} (n={len(ranks)}; 2.5 = neutral)  "
              f"dist: {dict(sorted(Counter(ranks).items()))}")
    if scored:
        deltas = [k["delta"] for k in scored if k["delta"] is not None]
        print(f"per-kyoku  win {rate('won', scored):.3f}  (tsumo {rate('tsumo', scored):.3f})  "
              f"deal-in {rate('dealt_in', scored):.3f}  riichi {rate('riichi', kyokus):.3f}  "
              f"called {rate('called', kyokus):.3f}")
        if deltas:
            print(f"mean score delta / kyoku: {statistics.mean(deltas):+.0f}  (n={len(deltas)})")
    print(f"decisions: {dict(dec)}")
    print()
    print("game  seat  kyokus  rank  final_points")
    for i, gm in enumerate(games):
        fp = gm["final"].get(gm["seat"]) if gm["final"] else None
        print(f"{i:>4}  {gm['seat']:>4}  {len(gm['kyokus']):>6}  {gm['rank'] or '-':>4}  {fp if fp is not None else '-'}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(games, f, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
