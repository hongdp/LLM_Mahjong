"""Refit every ladder from the archived primary records (exp56).

One set of games, several rulers. The match archives now carry per-game
rows (per-seat uma, placements, length), so the choice of scoring unit is
a FITTING decision, not something that has to be replayed:

  sign_match : each hanchan is one 0/1 observation — "who wins a match",
               the natural unit, comparable to run_hanchan_arena and to
               external top-rate numbers.
  sign_pair  : the duplicate PAIR is one observation (A's uma summed over
               both orientations). Same target, luck-cancelled, so the
               same skill gap reads as a LARGER Elo — a measurement trick,
               not a different quantity. Never mix the two on one table.
  pt         : Majsoul's ladder currency, pt = placement points +
               (score-25000)/1000, per player per hanchan. Fitted as a
               LINEAR paired-comparison model m_ij = theta_i - theta_j
               (weighted least squares, same pin as Elo), the additive
               analogue of the logistic Elo fit. Its residuals are a real
               diagnostic: large ones mean margin is not additive for that
               model (e.g. it crushes weak opponents but only ties strong
               ones), which is itself a finding.

  python scripts/fit_ladders.py --hanchan [--tag T0] [--out report.json]
"""

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_elo_league import (anchors_path, expected, fit_ratings,   # noqa: E402
                                    league_dir, rating_se, residuals, PINNED)

UMA_RANK = [15000, 5000, -5000, -15000]


def load_matches(d, history):
    """Every archived pair -> (name_a, name_b, games). Candidate files are
    all written under the label 'cand', so they are re-identified by
    (path, seed0) against history.jsonl."""
    by_key = {}
    for rec in history:
        by_key[(rec.get("ckpt"), rec.get("seed0"))] = rec.get("label")
    out = []
    for fn in sorted(glob.glob(f"{d}/matches/*.json")):
        blob = json.load(open(fn))
        if not blob.get("games"):
            continue                      # pre-exp56 archive: aggregate only
        na, nb = blob["a"], blob["b"]
        if na == "cand":
            na = by_key.get((blob["path_a"], blob["seed0"])) or f"cand@{blob['seed0']}"
        out.append((na, nb, blob["games"], blob))
    return out


def sign_obs(games, per_pair):
    if not per_pair:
        return [1.0 if g["a_pts"] > g["b_pts"] else
                0.0 if g["a_pts"] < g["b_pts"] else 0.5 for g in games]
    walls = {}
    for g in games:
        walls[g["wall"]] = walls.get(g["wall"], 0.0) + g["a_pts"]
    return [1.0 if v > 0 else 0.0 if v < 0 else 0.5 for v in walls.values()]


def pt_obs(games):
    """Per-player pt margin for side A: team uma / 2 seats / 1000."""
    return [g["a_pts"] / 2000.0 for g in games]


def fit_margin(obs, names, pinned, free):
    """Weighted least squares on m_ij = theta_i - theta_j, solved by
    Gauss-Seidel (the design is a connected graph, so it converges)."""
    th = {n: 0.0 for n in names}
    th[pinned[0]] = pinned[1]
    for _ in range(2000):
        step = 0.0
        for n in free:
            num = den = 0.0
            for a, b, m, w in obs:
                if a == n:
                    num += w * (m + th[b])
                    den += w
                elif b == n:
                    num += w * (th[a] - m)
                    den += w
            if den:
                new = num / den
                step = max(step, abs(new - th[n]))
                th[n] = new
        if step < 1e-6:
            break
    return th


def margin_se(obs, name):
    w = sum(o[3] for o in obs if name in (o[0], o[1]))
    return float("inf") if w == 0 else 1.0 / math.sqrt(w)


def margin_resid(obs, th, name):
    per = {}
    for a, b, m, _ in obs:
        if a == name:
            per.setdefault(b, []).append(m - (th[a] - th[b]))
        elif b == name:
            per.setdefault(a, []).append(-m - (th[b] - th[a]))
    return {o: round(sum(v) / len(v), 3) for o, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hanchan", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = league_dir(a.hanchan)
    pool = json.load(open(anchors_path(a.hanchan, a.tag)))
    anchors = set(pool["anchors"])
    hist_fn = f"{d}/history.jsonl"
    history = ([json.loads(l) for l in open(hist_fn)]
               if os.path.exists(hist_fn) else [])
    matches = load_matches(d, history)
    if not matches:
        raise SystemExit(f"no primary-record archives under {d}/matches/ — "
                         "only pre-exp56 aggregates; replay to refit")

    # keep only the pool's own condition: a T=0 pool must not be fitted
    # from matches its anchors played at T=1
    temp = float(pool.get("temperature", 1.0))
    matches = [m for m in matches
               if float(m[3].get("temp_b", 1.0)) == temp]

    tables = {}
    for scheme in ("sign_match", "sign_pair", "pt"):
        obs, names = [], set()
        for na, nb, games, blob in matches:
            names |= {na, nb}
            if scheme == "pt":
                vals = pt_obs(games)
                m = sum(vals) / len(vals)
                var = sum(v * v for v in vals) / len(vals) - m * m
                w = len(vals) / max(var, 1e-9)          # inverse-variance
                obs.append((na, nb, m, w))
            else:
                for s in sign_obs(games, scheme == "sign_pair"):
                    obs.append((na, nb, s))
        names = sorted(names)
        free = [n for n in names if n != PINNED[0]]
        if scheme == "pt":
            th = fit_margin(obs, names, PINNED if PINNED[0] in names
                            else (names[0], 0.0), free)
            th[PINNED[0]] = 0.0            # pt is an interval scale: pin 0
            th = fit_margin(obs, names, (PINNED[0], 0.0), free)
            tables[scheme] = {n: {"rating": round(th[n], 3),
                                  "se": round(margin_se(obs, n), 3),
                                  "residuals": margin_resid(obs, th, n)}
                              for n in names}
        else:
            r = {n: 1000.0 for n in names}
            r[PINNED[0]] = PINNED[1]
            fit_ratings(obs, r, free)
            tables[scheme] = {n: {"rating": round(r[n], 1),
                                  "se": round(rating_se(obs, r, n), 1),
                                  "residuals": residuals(obs, r, n)}
                              for n in names}

    order = sorted(tables["sign_pair"], key=lambda n: -tables["sign_pair"][n]["rating"])
    print(f"pool {anchors_path(a.hanchan, a.tag)}  unit={pool.get('unit','deal')}  "
          f"T={temp}  models={len(order)}  pairs={len(matches)}")
    print(f"{'model':>20} {'Elo/场':>14} {'Elo/复式对':>14} {'pt/半庄/人':>14}")
    for n in order:
        sm, sp, pt = (tables[s][n] for s in ("sign_match", "sign_pair", "pt"))
        star = "*" if n in anchors else " "
        print(f"{n:>19}{star} {sm['rating']:8.1f}±{sm['se']:<4.1f} "
              f"{sp['rating']:8.1f}±{sp['se']:<4.1f} {pt['rating']:+8.2f}±{pt['se']:<4.2f}")
    out = a.out or f"{d}/ladders{'_' + a.tag if a.tag else ''}.json"
    json.dump({"pool": anchors_path(a.hanchan, a.tag), "temperature": temp,
               "unit": pool.get("unit", "deal"), "n_pairs": len(matches),
               "anchors": sorted(anchors), "tables": tables},
              open(out, "w"), indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
