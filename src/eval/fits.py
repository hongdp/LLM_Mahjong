"""Ratings as pure functions of the ledger (design D6).

One set of matches, several rulers — each with its unit stated, never
mixed on one table:

  pl        : Plackett-Luce over the table's full PLACEMENT ranking. The
              right likelihood for a 4-player game: one match is ONE
              observation of a joint ranking, so the correlation between
              the six pairwise comparisons at a table is handled instead
              of being silently ignored (naive pairwise Elo on 4-seat
              tables understates SE). Reported on the Elo scale
              (400/ln10 per logit) so the numbers stay readable.
  pt        : Majsoul's ladder currency, pt = placement points +
              (score-25000)/1000 per player per hanchan. Linear model:
              E[uma_i | table] = theta_i - mean(theta at table). Interval
              scale, so a crushing win counts more than a squeaker.
  placement : top rate / 4th rate / mean placement with CIs — the view
              that made Mortal's edge visible when the win share hid it.
  sign_pair : v1's duplicate-pair sign Elo, kept for continuity with the
              epoch-6 tables. NOT comparable to per-match numbers.
"""

import math

import torch

ELO_PER_LOGIT = 400.0 / math.log(10.0)


def _index(rows):
    ents = sorted({e for r in rows for e in r["seats"]})
    return ents, {e: i for i, e in enumerate(ents)}


def _pin_index(ents, pin):
    return ents.index(pin) if pin in ents else 0


def fit_pl(rows, pin=None, pin_value=1000.0, iters=400):
    """Plackett-Luce MLE. Returns {entity: {"rating", "se"}} on Elo scale."""
    ents, idx = _index(rows)
    n = len(ents)
    if n < 2:
        return {}
    order = []
    for r in rows:
        # seats sorted best-to-worst by placement
        seq = sorted(range(4), key=lambda s: r["placements"][s])
        order.append([idx[r["seats"][s]] for s in seq])
    O = torch.tensor(order, dtype=torch.long)
    theta = torch.zeros(n, requires_grad=True)

    def nll(t):
        # log P = sum_k [ t_pi(k) - logsumexp(t_pi(k..3)) ]
        vals = t[O]                                   # (m, 4)
        ll = 0.0
        for k in range(3):
            ll = ll + vals[:, k] - torch.logsumexp(vals[:, k:], dim=1)
        return -ll.sum()

    opt = torch.optim.LBFGS([theta], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        # tiny ridge keeps a disconnected component from running to +-inf
        loss = nll(theta) + 1e-4 * (theta ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    t = theta.detach()
    H = torch.autograd.functional.hessian(nll, t.clone().requires_grad_(False))
    p = _pin_index(ents, pin)
    keep = [i for i in range(n) if i != p]
    try:
        cov = torch.linalg.inv(H[keep][:, keep] + 1e-6 * torch.eye(len(keep)))
        se = torch.zeros(n)
        se[torch.tensor(keep)] = torch.sqrt(torch.clamp(torch.diag(cov), min=0))
    except Exception:
        se = torch.full((n,), float("nan"))
    shift = pin_value - t[p].item() * ELO_PER_LOGIT
    return {e: {"rating": round(t[i].item() * ELO_PER_LOGIT + shift, 1),
                "se": round(se[i].item() * ELO_PER_LOGIT, 1)}
            for e, i in idx.items()}


def fit_pt(rows, pin=None, pin_value=0.0):
    """Least squares on table-centered uma. Returns pt per player per match."""
    ents, idx = _index(rows)
    n = len(ents)
    if n < 2:
        return {}
    A = torch.zeros(len(rows) * 4, n)
    y = torch.zeros(len(rows) * 4)
    for m, r in enumerate(rows):
        for s in range(4):
            k = m * 4 + s
            A[k, idx[r["seats"][s]]] += 1.0
            for s2 in range(4):
                A[k, idx[r["seats"][s2]]] -= 0.25
            y[k] = r["uma"][s] / 1000.0
    sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
    resid = (A @ sol - y)
    dof = max(len(y) - n + 1, 1)
    s2 = (resid ** 2).sum().item() / dof
    try:
        cov = torch.linalg.pinv(A.T @ A) * s2
        se = torch.sqrt(torch.clamp(torch.diag(cov), min=0))
    except Exception:
        se = torch.full((n,), float("nan"))
    p = _pin_index(ents, pin)
    shift = pin_value - sol[p].item()
    return {e: {"rating": round(sol[i].item() + shift, 3),
                "se": round(se[i].item(), 3)}
            for e, i in idx.items()}


def placement_table(rows):
    out = {}
    for r in rows:
        for s in range(4):
            e = r["seats"][s]
            d = out.setdefault(e, {"n": 0, "plc": [0, 0, 0, 0], "uma": 0.0,
                                   "busted": 0, "deals": 0})
            d["n"] += 1
            d["plc"][r["placements"][s] - 1] += 1
            d["uma"] += r["uma"][s] / 1000.0
            d["deals"] += r["n_deals"]
            d["busted"] += bool(r["busted"] and r["points"][s] < 0)
    for e, d in out.items():
        n = d["n"]
        d["top_rate"] = round(d["plc"][0] / n, 4)
        d["last_rate"] = round(d["plc"][3] / n, 4)
        d["mean_plc"] = round(sum((i + 1) * c for i, c in enumerate(d["plc"])) / n, 3)
        d["mean_pt"] = round(d["uma"] / n, 3)
        d["se_plc"] = round(math.sqrt(max(sum(
            ((i + 1) - d["mean_plc"]) ** 2 * c for i, c in enumerate(d["plc"]))
            / max(n - 1, 1), 0) / n), 3)
        d["bust_rate"] = round(d["busted"] / n, 4)
        d["mean_deals"] = round(d["deals"] / n, 2)
    return out


def head_to_head(rows):
    """Per-seat pairwise outcomes by uma, with a table-correlation caveat:
    counts are seat-pair comparisons, so treat CIs as optimistic when a
    pair meets many times at the same table."""
    hh = {}
    for r in rows:
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = r["seats"][i], r["seats"][j]
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                ua, ub = (r["uma"][i], r["uma"][j]) if a < b else (r["uma"][j], r["uma"][i])
                d = hh.setdefault(key, {"n": 0, "w": 0.0, "margin": 0.0})
                d["n"] += 1
                d["w"] += 1.0 if ua > ub else 0.0 if ua < ub else 0.5
                d["margin"] += (ua - ub) / 1000.0
    for k, d in hh.items():
        d["share"] = round(d["w"] / d["n"], 4)
        d["mean_pt_diff"] = round(d["margin"] / d["n"], 3)
        d["se"] = round(math.sqrt(max(d["share"] * (1 - d["share"]), 1e-9) / d["n"]), 4)
    return hh
