"""Arena: pit two policies (LoRA adapters on a shared base) at one table.

Protocol
--------
- 2v2 diagonal seating: policy "A" holds seats {0,2}, policy "B" holds {1,3}.
- Duplicate deals: every deal seed is played in BOTH orientations (A on 0/2,
  then A on 1/3) with the identical wall, so deal luck cancels pairwise.
- Strength readout uses RAW final table points (not training rewards):
  per-deal paired differential = sum over both orientations of
  (A-team points − 50000). Positive ⇒ A stronger on that deal.

Implementation notes
--------------------
- One bf16 base model + two named PEFT adapters; per decision round the
  pending requests are grouped by policy and generated in two batched
  calls with model.set_adapter() in between.
- Reuses batch_rollout's game generator (_drive_game) and request dataclass;
  no logprob capture (evaluation only).
"""

import random
from typing import Dict, List, Optional

import torch

from src.tasks.mahjong.table import PyMahjongTable
from src.tasks.mahjong.batch_rollout import _drive_game, _batch_generate, _Req


def run_match(model, tokenizer, deal_seeds: List[int],
              value_facts: bool = False, parallel: int = 12,
              policy_names=("A", "B"), log_path: Optional[str] = None):
    """Returns per-deal paired results:
    [{"seed": s, "diff": paired_point_diff, "wins_a": int, "wins_b": int,
      "orient": [{"a_seats": [...], "points": [...], "result": str}, ...]}]
    """
    jobs = []   # (seed, orientation) — orientation 0: A on {0,2}; 1: A on {1,3}
    for s in deal_seeds:
        jobs.append((s, 0))
        jobs.append((s, 1))

    results: Dict[tuple, dict] = {}
    next_job = 0
    active: Dict[tuple, dict] = {}

    def a_seats(orient):
        return (0, 2) if orient == 0 else (1, 3)

    def start(job):
        seed, orient = job
        random.seed(seed)                     # deterministic wall per seed
        table = PyMahjongTable(value_facts=value_facts)
        trajectories = {i: [] for i in range(4)}
        gen = _drive_game(table, trajectories, model, tokenizer)
        try:
            pending = next(gen)
        except StopIteration:
            pending = []
        active[job] = {"gen": gen, "table": table, "pending": pending}

    def finalize(job):
        st = active.pop(job)
        table = st["table"]
        seed, orient = job
        results[job] = {
            "points": list(table.points),
            "result": table.result_summary,
            "a_seats": list(a_seats(orient)),
        }

    while next_job < len(jobs) or active:
        while next_job < len(jobs) and len(active) < parallel:
            start(jobs[next_job])
            next_job += 1
        if not active:
            break

        # group pending requests by policy
        by_policy: Dict[str, List[_Req]] = {policy_names[0]: [],
                                            policy_names[1]: []}
        for job, st in active.items():
            _, orient = job
            aset = set(a_seats(orient))
            for r in st["pending"]:
                pol = policy_names[0] if r.player_id in aset else policy_names[1]
                by_policy[pol].append(r)

        for pol, reqs in by_policy.items():
            if not reqs:
                continue
            if model is not None:
                model.set_adapter(pol)
            _batch_generate(model, tokenizer, reqs, capture=False,
                            max_batch=max(24, parallel))

        for job in list(active.keys()):
            st = active[job]
            try:
                st["pending"] = st["gen"].send(st["pending"])
            except StopIteration:
                finalize(job)

    # pair up orientations
    out = []
    for s in deal_seeds:
        pair = [results.get((s, 0)), results.get((s, 1))]
        if not all(pair):
            continue
        diff = 0.0
        wins_a = wins_b = 0
        orient_rows = []
        for orient, r in enumerate(pair):
            a = set(r["a_seats"])
            a_pts = sum(p for i, p in enumerate(r["points"]) if i in a)
            diff += a_pts - 50000
            winner = None
            if "荣和" in r["result"] or "自摸" in r["result"]:
                import re as _re
                m = _re.search(r'玩家(\d)', r["result"])
                if m:
                    winner = int(m.group(1))
            if winner is not None:
                if winner in a:
                    wins_a += 1
                else:
                    wins_b += 1
            orient_rows.append({"a_seats": r["a_seats"],
                                "points": r["points"],
                                "result": r["result"]})
        out.append({"seed": s, "diff": diff, "wins_a": wins_a,
                    "wins_b": wins_b, "orient": orient_rows})
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"seed={s} diff={diff:+.0f} "
                        f"wins A/B={wins_a}/{wins_b}\n")
    return out
