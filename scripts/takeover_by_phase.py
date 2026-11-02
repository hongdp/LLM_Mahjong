"""Where does BC's play earn points in the RL population, by phase? (Python engine, all T=0, paired same wall.)

Table: 4 RL seats (the population the RL policy is trained in). In each game one own-turn decision of seat 0 is
sampled at a random target phase (own discards so far). From a deep copy of the table at that decision, three
continuations are played to the end of the hand (identical wall, deterministic policies):
  A  RL plays seat 0 throughout                      (baseline)
  S  BC makes this ONE decision, RL continues        (single-step value of BC's advice)
  T  BC takes over seat 0 until the end of the hand  (multi-step value of BC's continuation)
Other seats are RL in every branch. Reported per phase bucket / situation: mean(S-A), mean(T-A) in points (one sample
per game => plain SE is the clustered SE), plus how often BC's first choice differs from RL's.

usage: takeover_by_phase.py --bc bc49.pt --rl rl.pt --games N --seed0 S --out part.json
       takeover_by_phase.py --report part*.json [--md out.md]
"""
from __future__ import annotations

import argparse, copy, json, os, random, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.policy_diff_atlas import _DummyStep, _type, _tile, _base, bucket  # noqa: E402


def collect(a):
    from src.agents.dnn.selfplay import play_game_gen, _choose
    from scripts.run_elo_league import load_dnn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nets = {"bc": load_dnn(a.bc, dev).eval(), "rl": load_dnn(a.rl, dev).eval()}

    def greedy(who, table, pid, actions):
        with torch.no_grad():
            _, s = _choose(nets[who], table, pid, actions, 0.0, dev)
        return s

    def play(table, seat0, first=None):
        """Continue a cloned table to the end; seat 0 uses `seat0` ('rl'|'bc'), its FIRST decision may be forced."""
        gen = play_game_gen(table=table)
        forced = first
        try:
            t, reqs = next(gen)
            while True:
                rep = []
                for pid, acts in reqs:
                    if pid == 0 and forced is not None:
                        rep.append(forced); forced = None
                    else:
                        rep.append(greedy(seat0 if pid == 0 else "rl", t, pid, acts))
                t, reqs = gen.send([(_DummyStep(), r) for r in rep])
        except StopIteration as stop:
            g = stop.value
            return g.points[0]

    rng = random.Random(a.seed0)
    out = []
    for gi in range(a.games):
        wall = a.seed0 + gi
        target = rng.randint(0, 17)
        gen = play_game_gen(deal_seed=wall, randomize_round=True)
        sample = None
        try:
            t, reqs = next(gen)
            while True:
                rep = []
                for pid, acts in reqs:
                    cand = (sample is None and pid == 0 and pid == t.turn and len(acts) > 1 and not t.riichi[0]
                            and (a.cond != "any" or len(t.discards[0]) >= target))
                    if cand and a.cond != "any":
                        rl_a = greedy("rl", t, 0, acts); bc_a = greedy("bc", t, 0, acts)
                        opp = [o for o in (1, 2, 3) if t.riichi[o]]
                        sh_ = int(t._shanten(t.hands[0], len(t.melds[0])))
                        if a.cond == "dama":            # riichi on offer, RL declares, BC stays dama
                            cand = _type(rl_a) == "riichi" and _type(bc_a) == "discard"
                        elif a.cond == "fold":          # opponent riichi, self >= 1-shanten, BC genbutsu, RL not
                            gb = lambda x: _type(x) == "discard" and all(_base(_tile(x)) in {_base(y) for y in t.discards[o]} for o in opp)
                            cand = bool(opp) and sh_ >= 1 and gb(bc_a) and not gb(rl_a)
                    if cand:
                        rl_a = greedy("rl", t, 0, acts); bc_a = greedy("bc", t, 0, acts)
                        opp_r = sum(1 for o in (1, 2, 3) if t.riichi[o])
                        sample = {"g": gi, "ph": len(t.discards[0]), "n_opp_r": opp_r,
                                  "sh": int(t._shanten(t.hands[0], len(t.melds[0]))),
                                  "rl": _type(rl_a), "rl_t": _base(_tile(rl_a)), "bc": _type(bc_a), "bc_t": _base(_tile(bc_a)),
                                  "differ": int(rl_a != bc_a), "start": t.points[0]}
                        st = random.getstate()
                        pA = play(copy.deepcopy(t), "rl")
                        pS = play(copy.deepcopy(t), "rl", first=bc_a) if rl_a != bc_a else pA
                        pT = play(copy.deepcopy(t), "bc")
                        random.setstate(st)
                        sample.update(A=pA, S=pS, T=pT)
                        rep.append(rl_a)
                    else:
                        rep.append(greedy("rl", t, pid, acts))
                t, reqs = gen.send([(_DummyStep(), r) for r in rep])
        except StopIteration:
            pass
        if sample:
            out.append(sample)
        if (gi + 1) % 100 == 0:
            print(f"{gi + 1}/{a.games} games, {len(out)} samples", flush=True)
    json.dump({"bc": a.bc, "rl": a.rl, "samples": out}, open(a.out, "w")); print("->", a.out)


def report(paths, md=None):
    S = []; meta = None
    for p in paths:
        d = json.load(open(p)); S += d["samples"]; meta = (d["bc"], d["rl"])
    L = []; pr = L.append
    pr(f"# Takeover by phase (4 RL seats; seat 0 branches)\n\nBC = `{meta[0]}`\nRL = `{meta[1]}`\n{len(S)} sampled decisions (one per game)\n")
    pr("| slice | n | BC first choice differs | single step S−A | takeover T−A |\n|---|---|---|---|---|")

    def row(name, s):
        if len(s) < 30:
            return
        d1 = np.array([x["S"] - x["A"] for x in s], float); d2 = np.array([x["T"] - x["A"] for x in s], float)
        pr(f"| {name} | {len(s)} | {np.mean([x['differ'] for x in s]):.2f} | {d1.mean():+.0f} ± {d1.std(ddof=1)/np.sqrt(len(s)):.0f} | "
           f"{d2.mean():+.0f} ± {d2.std(ddof=1)/np.sqrt(len(s)):.0f} |")
    row("all", S)
    for b in ("early(0-5)", "mid(6-11)", "late(12+)"):
        row(b, [x for x in S if bucket(x["ph"]) == b])
    for b in ("early(0-5)", "mid(6-11)", "late(12+)"):
        row(b + ", no threat", [x for x in S if bucket(x["ph"]) == b and x["n_opp_r"] == 0])
        row(b + ", under riichi", [x for x in S if bucket(x["ph"]) == b and x["n_opp_r"] > 0])
    for sh in (0, 1, 2):
        row(f"own shanten {'2+' if sh == 2 else sh}", [x for x in S if (min(x["sh"], 2)) == sh])
    txt = "\n".join(L); print(txt)
    if md:
        open(md, "w").write(txt + "\n"); print("->", md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc"); ap.add_argument("--rl"); ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=995000); ap.add_argument("--out")
    ap.add_argument("--cond", default="any", choices=["any", "dama", "fold"],
                    help="any = random decision at a random target phase; dama / fold = first decision of that kind")
    ap.add_argument("--report", nargs="*"); ap.add_argument("--md")
    a = ap.parse_args()
    report(a.report, a.md) if a.report else collect(a)


if __name__ == "__main__":
    main()
