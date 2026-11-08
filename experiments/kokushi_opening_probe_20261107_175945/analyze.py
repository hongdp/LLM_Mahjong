"""Summarize the terminal-heavy opening probe by stratum. usage: analyze.py RESULTS.json HUMAN.json"""
import json
import math
import sys

R = json.load(open(sys.argv[1]))
H = json.load(open(sys.argv[2]))


def strat(r):
    return "≥10" if r["K"] >= 10 else str(r["K"])


def pct(n, d):
    return f"{100 * n / d:5.1f}%" if d else "   - "


def table(name, rows, human=False):
    out = []
    for s in ("8", "9", "≥10"):
        rs = [r for r in rows if strat(r) == s]
        if not rs:
            continue
        leg = [r for r in rs if (r["K_draw"] or 0) >= 9] if human else [r for r in rs if r["kyuushu_legal"]]
        ky = sum(r["kyuushu"] for r in leg)
        n6 = sum(r["n6"] for r in rs)
        y6 = sum(r["yao6"] for r in rs)
        tp = sum(1 for r in rs if r.get("ks_min") == 0) if not human else None
        kw = sum(r["kokushi_win"] for r in rs)
        won = sum(r["won"] for r in rs)
        dl = sum(r.get("delta", 0) for r in rs) / len(rs) if not human else None
        out.append(f"{name:16s} K={s:3s} n={len(rs):6d} | kyuushu {pct(ky, len(leg))} of {len(leg):5d} legal | "
                   f"yao in first6 {pct(y6, n6)} | kokushi tenpai {pct(tp, len(rs)) if tp is not None else '   - '} | "
                   f"kokushi win {kw:4d} ({pct(kw, len(rs))}) | win {pct(won, len(rs))}"
                   + (f" | Δpts {dl:+7.0f}" if dl is not None else ""))
    return out


for name, v in R.items():
    print("\n".join(table(name, v["rows"])))
print("\n".join(table("human(Tenhou)", H, human=True)))
