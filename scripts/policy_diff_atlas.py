"""Frame-by-frame policy comparison (BC vs pure RL) on shared states, Python engine, all T=0.

Mixed tables: 2 seats BC + 2 seats RL (seats swap with the wall's low bit). At EVERY decision of every seat both
policies are queried on the identical state; the seat owner's choice is played. Each record keeps the decision kind
(own turn / claim), phase (own discards so far), situation (own shanten, opponents in riichi, own riichi, dealer), the
option types on offer, and both policies' greedy choices (type + tile, tile class, genbutsu vs riichi opponents).
Game outcomes are recorded per seat (win / deal-in / riichi / meld / point delta) for the owner's style profile.

usage: policy_diff_atlas.py --bc bc49.pt --rl rl.pt --games N --seed0 S --out part.json      (collect)
       policy_diff_atlas.py --report part1.json part2.json ...  [--md out.md]                   (aggregate)
"""
from __future__ import annotations

import argparse, collections, json, os, random, re, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TYPE_RE = re.compile(r'type="(\w+)"')
TILE_RE = re.compile(r'tile="(\w+)"')
WIN_RE = re.compile(r"玩家(\d) (自摸|荣和|抢杠)(?:\(放铳:玩家(\d)\))?")


def _type(a):
    m = TYPE_RE.search(a or ""); return m.group(1) if m else None


def _tile(a):
    m = TILE_RE.search(a or ""); return m.group(1) if m else None


def _base(t):
    return None if t is None else (("5" + t[1]) if t[0] == "0" else t)


def _cls(t):
    if t is None:
        return None
    t = _base(t)
    return "honor" if t[1] == "z" else ("term" if t[0] in "19" else "simple")


def collect(args):
    from src.agents.dnn.selfplay import play_game_gen, _choose
    from scripts.run_elo_league import load_dnn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nets = {"bc": load_dnn(args.bc, dev).eval(), "rl": load_dnn(args.rl, dev).eval()}
    if args.fast and hasattr(nets["rl"], "blocks"):
        from src.agents.dnn.net import enable_fast_trunk
        enable_fast_trunk(nets["rl"])

    def greedy(who, table, pid, actions):
        with torch.no_grad():
            _, a = _choose(nets[who], table, pid, actions, 0.0, dev)
        return a

    recs, outs = [], []
    for gi in range(args.games):
        wall = args.seed0 + gi
        owner = {p: ("bc" if (p + wall) % 2 == 0 else "rl") for p in range(4)}
        gen = play_game_gen(deal_seed=wall, randomize_round=True)
        try:
            table, reqs = next(gen)
            while True:
                replies = []
                for pid, actions in reqs:
                    k = "own" if pid == table.turn and any(_type(a) == "discard" or _type(a) == "tsumo" for a in actions) else "claim"
                    ch = {w: greedy(w, table, pid, actions) for w in ("bc", "rl")}
                    opp_r = [o for o in range(4) if o != pid and table.riichi[o]]
                    rec = {"g": gi, "pid": pid, "own": owner[pid], "kind": k,
                           "ph": len(table.discards[pid]), "dealer": int(table.dealer == pid),
                           "self_r": int(bool(table.riichi[pid])), "n_opp_r": len(opp_r),
                           "sh": int(table._shanten(table.hands[pid], len(table.melds[pid]))),
                           "opts": sorted({_type(a) for a in actions}), "na": len(actions)}
                    for w in ("bc", "rl"):
                        t = _tile(ch[w]); ty = _type(ch[w])
                        rec[w] = ty; rec[w + "_t"] = _base(t); rec[w + "_c"] = _cls(t)
                        if ty == "discard" and opp_r:
                            rec[w + "_gb"] = int(all(_base(t) in {_base(x) for x in table.discards[o]} for o in opp_r))
                    recs.append(rec)
                    replies.append(ch[owner[pid]])
                table, reqs = gen.send([(_DummyStep(), r) for r in replies])
        except StopIteration as stop:
            game = stop.value
            summ = game.result or ""
            wins = {int(m.group(1)): (int(m.group(3)) if m.group(3) else None) for m in WIN_RE.finditer(summ)}
            houjuu = {d for d in wins.values() if d is not None}
            for p in range(4):
                outs.append({"g": gi, "pid": p, "own": owner[p], "win": int(p in wins), "tsumo": int(p in wins and wins[p] is None),
                             "houjuu": int(p in houjuu), "riichi": int(game.riichi[p]), "meld": int(game.n_melds[p] > 0),
                             "delta": game.points[p] - game.start_points[p], "draw": int("流局" in summ)})
        if (gi + 1) % 100 == 0:
            print(f"{gi + 1}/{args.games} games, {len(recs)} decisions", flush=True)
    json.dump({"bc": args.bc, "rl": args.rl, "recs": recs, "outs": outs}, open(args.out, "w"))
    print("->", args.out)


class _DummyStep:
    """play_game_gen only stores the step object and sets reward/phi on it."""
    reward = 0.0; is_terminal = False; phi = 0.0


def bucket(ph):
    return "early(0-5)" if ph <= 5 else ("mid(6-11)" if ph <= 11 else "late(12+)")


def report(paths, md_out=None):
    recs, outs = [], []
    meta = None
    for p in paths:
        d = json.load(open(p)); recs += d["recs"]; outs += d["outs"]; meta = (d["bc"], d["rl"])
    L = []
    pr = lambda s="": L.append(s)
    pr(f"# BC vs RL frame atlas\n\nBC = `{meta[0]}`\nRL = `{meta[1]}`\n\n{len(recs)} decisions, {len(outs)//4} games (2 BC + 2 RL seats, all T=0)\n")
    # outcomes
    pr("## Outcomes per seat (mixed table)\n\n| policy | n | win | tsumo share | deal-in | riichi | meld | mean Δ points |\n|---|---|---|---|---|---|---|---|")
    for w in ("bc", "rl"):
        o = [x for x in outs if x["own"] == w]; n = len(o)
        wins = sum(x["win"] for x in o)
        pr(f"| {w} | {n} | {wins/n:.3f} | {sum(x['tsumo'] for x in o)/max(wins,1):.2f} | {sum(x['houjuu'] for x in o)/n:.3f} | "
           f"{sum(x['riichi'] for x in o)/n:.3f} | {sum(x['meld'] for x in o)/n:.3f} | {np.mean([x['delta'] for x in o]):+.0f} ± {np.std([x['delta'] for x in o])/np.sqrt(n):.0f} |")
    # agreement
    pr("\n## Top-1 agreement on identical states (multi-option decisions)\n\n| kind | phase | n | agree |\n|---|---|---|---|")
    multi = [r for r in recs if r.get("na", 2) > 1]
    for k in ("own", "claim"):
        for b in ("early(0-5)", "mid(6-11)", "late(12+)"):
            s = [r for r in multi if r["kind"] == k and bucket(r["ph"]) == b]
            if s:
                pr(f"| {k} | {b} | {len(s)} | {np.mean([(r['bc'], r.get('bc_t')) == (r['rl'], r.get('rl_t')) for r in s]):.3f} |")
    # riichi decision
    pr("\n## Riichi when available (own turn, not yet in riichi)\n\n| phase | own shanten | n | BC riichi | RL riichi |\n|---|---|---|---|---|")
    for b in ("early(0-5)", "mid(6-11)", "late(12+)"):
        s = [r for r in recs if r["kind"] == "own" and "riichi" in r["opts"] and not r["self_r"] and bucket(r["ph"]) == b]
        if s:
            pr(f"| {b} | 0 | {len(s)} | {np.mean([r['bc'] == 'riichi' for r in s]):.3f} | {np.mean([r['rl'] == 'riichi' for r in s]):.3f} |")
    # claims
    pr("\n## Claims offered (accept = anything but skip)\n\n| option | phase | n | BC accept | RL accept |\n|---|---|---|---|---|")
    for opt in ("chi", "pon", "kan", "ron"):
        for b in ("early(0-5)", "mid(6-11)", "late(12+)"):
            s = [r for r in recs if r["kind"] == "claim" and opt in r["opts"] and bucket(r["ph"]) == b]
            if len(s) >= 30:
                pr(f"| {opt} | {b} | {len(s)} | {np.mean([r['bc'] not in ('skip', 'pass', None) for r in s]):.3f} | {np.mean([r['rl'] not in ('skip', 'pass', None) for r in s]):.3f} |")
    # discard composition, no threat
    pr("\n## Discard tile class, no riichi threat (own turn, both discard)\n\n| phase | n | BC honor/term/simple | RL honor/term/simple | same tile |\n|---|---|---|---|---|")
    for b in ("early(0-5)", "mid(6-11)", "late(12+)"):
        s = [r for r in recs if r["kind"] == "own" and r["n_opp_r"] == 0 and r["bc"] == "discard" and r["rl"] == "discard" and bucket(r["ph"]) == b]
        if s:
            f = lambda w: "/".join(f"{np.mean([r[w + '_c'] == c for r in s]):.2f}" for c in ("honor", "term", "simple"))
            pr(f"| {b} | {len(s)} | {f('bc')} | {f('rl')} | {np.mean([r['bc_t'] == r['rl_t'] for r in s]):.3f} |")
    # under threat
    pr("\n## Under riichi threat (opponent in riichi, self not): genbutsu discard rate\n\n| own shanten | phase | n | BC genbutsu | RL genbutsu | same tile |\n|---|---|---|---|---|---|")
    for sh in ("0", "1", "2+"):
        for b in ("early(0-5)", "mid(6-11)", "late(12+)"):
            s = [r for r in recs if r["kind"] == "own" and r["n_opp_r"] > 0 and not r["self_r"] and r["bc"] == "discard" and r["rl"] == "discard"
                 and bucket(r["ph"]) == b and (str(min(r["sh"], 2)) if r["sh"] < 2 else "2+") == sh]
            if len(s) >= 30:
                pr(f"| {sh} | {b} | {len(s)} | {np.mean([r.get('bc_gb', 0) for r in s]):.3f} | {np.mean([r.get('rl_gb', 0) for r in s]):.3f} | {np.mean([r['bc_t'] == r['rl_t'] for r in s]):.3f} |")
    # disagreement cross-tab
    pr("\n## Disagreement types (multi-option decisions where the choices differ)\n\n| BC chooses | RL chooses | n | share |\n|---|---|---|---|")
    dis = [r for r in multi if (r["bc"], r.get("bc_t")) != (r["rl"], r.get("rl_t"))]
    cnt = collections.Counter((r["bc"], r["rl"]) for r in dis)
    for (a, b), c in cnt.most_common(12):
        pr(f"| {a} | {b} | {c} | {c/len(dis):.3f} |")
    txt = "\n".join(L); print(txt)
    if md_out:
        open(md_out, "w").write(txt + "\n"); print("->", md_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc"); ap.add_argument("--rl"); ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=990000); ap.add_argument("--out")
    ap.add_argument("--fast", action="store_true"); ap.add_argument("--report", nargs="*"); ap.add_argument("--md")
    a = ap.parse_args()
    if a.report:
        report(a.report, a.md)
    else:
        collect(a)


if __name__ == "__main__":
    main()
