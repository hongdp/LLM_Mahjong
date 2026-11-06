"""Information-driven match scheduling (design D7).

v1 played every anchor pair the same 200 matches. On the hanchan scale
that put 24% of the compute into 0-2/200 shutouts, which carry almost no
information, while the pairs that actually decide the championship got
the same allowance. v2 spends time where it buys variance.

Objective (greedy A-optimality, stated as the approximation it is): a
pair's Fisher information about its own contrast is proportional to
p(1-p), so a table's value is the sum over its six pairs of
p(1-p) * (var_i + var_j), i.e. informative AND still-uncertain pairs
first. Divided by the table's cost in seconds, since a Mortal table
costs ~30x a net table. After each pick the picked pairs' variances are
decremented, so the greedy does not stack every match on one contrast.

Declared targets ("champion vs Mortal") get a weight multiplier: the
schedule serves the questions we are actually asking, not just the
average SE.
"""

import itertools
import math

ELO_PER_LOGIT = 400.0 / math.log(10.0)


def expected_score(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def pair_counts(rows):
    out = {}
    for r in rows:
        uniq = sorted(set(r["seats"]))
        for a, b in itertools.combinations(uniq, 2):
            out[(a, b)] = out.get((a, b), 0) + 1
    return out


def _key(a, b):
    return (a, b) if a < b else (b, a)


def plan(entities, ratings, rows=(), targets=(), budget_seconds=1800,
         walls_per_table=25, rotations=4, throughput_per_min=600.0,
         cost_per_min=None, target_weight=8.0, prior_var=400.0 ** 2,
         min_opponents=3, max_tables=200):
    """Returns [{"table": (e0,e1,e2,e3), "walls": n, "score": s}, ...].

    ratings: {eid: {"rating", "se"}} — unrated entities are given
    prior_var so a newcomer is scheduled aggressively.
    cost_per_min: {eid: matches/min} for entities whose driver is slower
    (Mortal); a table costs the min throughput of its members.
    """
    ents = list(entities)
    if len(ents) < 4:
        raise ValueError("need at least four entities to seat a table")
    var = {e: (ratings.get(e, {}).get("se") or math.sqrt(prior_var)) ** 2
           for e in ents}
    rating = {e: ratings.get(e, {}).get("rating", 1000.0) for e in ents}
    tgt = {_key(a, b): target_weight for a, b in targets}
    have = pair_counts(rows)
    cost_per_min = cost_per_min or {}

    def table_rate(t):
        return min([throughput_per_min] + [cost_per_min[e] for e in t
                                           if e in cost_per_min])

    out, spent = [], 0.0
    for _ in range(max_tables):
        best, best_val = None, 0.0
        for t in itertools.combinations(ents, 4):
            info = 0.0
            for a, b in itertools.combinations(t, 2):
                p = expected_score(rating[a], rating[b])
                w = tgt.get(_key(a, b), 1.0)
                # a pair already measured to death is worth less: damp by
                # the count we have, so the greedy spreads out
                damp = 1.0 / (1.0 + have.get(_key(a, b), 0) / 200.0)
                info += p * (1 - p) * (var[a] + var[b]) * w * damp
            matches = walls_per_table * rotations
            secs = matches / max(table_rate(t), 1e-6) * 60.0
            val = info / secs
            if val > best_val:
                best, best_val = t, val
        if best is None:
            break
        matches = walls_per_table * rotations
        secs = matches / max(table_rate(best), 1e-6) * 60.0
        if spent + secs > budget_seconds:
            break
        spent += secs
        out.append({"table": best, "walls": walls_per_table,
                    "score": round(best_val, 6), "est_seconds": round(secs, 1)})
        # bookkeeping: each pair in the table gains information, so its
        # variance drops and the next pick looks elsewhere
        for a, b in itertools.combinations(best, 2):
            k = _key(a, b)
            have[k] = have.get(k, 0) + matches
            p = expected_score(rating[a], rating[b])
            gain = matches * max(p * (1 - p), 1e-6) / (ELO_PER_LOGIT ** 2)
            for e in (a, b):
                var[e] = 1.0 / (1.0 / max(var[e], 1e-9) + gain)

    forced = _connectivity_fixups(ents, rating, have, min_opponents,
                                  walls_per_table)
    return out + forced, {"est_seconds": round(spent, 1),
                          "tables": len(out) + len(forced)}


def _connectivity_fixups(ents, rating, have, min_opponents, walls):
    """A greedy on information alone can strand an entity: if nobody near
    its strength is uncertain, it never gets scheduled and its rating
    floats free of the pinned origin. Force tables until every entity has
    `min_opponents` distinct opponents it has actually played."""
    played = {e: set() for e in ents}
    for (a, b), n in have.items():
        if n and a in played and b in played:
            played[a].add(b)
            played[b].add(a)
    out = []
    for e in ents:
        while len(played[e]) < min_opponents and len(ents) >= 4:
            near = sorted((x for x in ents if x != e and x not in played[e]),
                          key=lambda x: abs(rating[x] - rating[e]))[:3]
            if len(near) < 3:
                break
            t = (e,) + tuple(near)
            out.append({"table": t, "walls": walls, "score": None,
                        "forced": "connectivity"})
            for a in t:
                for b in t:
                    if a != b:
                        played[a].add(b)
    return out
