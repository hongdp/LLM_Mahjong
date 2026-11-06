"""Batch-1 arena driver for DNN-vs-DNN matches on the Python engine.

Duplicate-deal, both-orientation, paired-differential protocol
(src/tasks/mahjong/arena.run_match). The vectorised GPU path
(scripts/run_elo_league.play_pair_vector) is ~50x faster and is what every
verdict uses; this driver remains for CPU-only boxes and for the rule
override diagnostics (src/agents/dnn/overrides.py).

`load_dnn(path, device)` lives in src/agents/dnn/checkpoint.py since the
2026-11-01 clean-up; it is re-exported here because resident evaluators,
the probes and the Majsoul service import it from this module.
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.dnn.checkpoint import load_dnn as _load_dnn   # noqa: E402
from src.tasks.mahjong.arena import run_match                 # noqa: E402


def load_dnn(path, device):
    """Stable entry point: checkpoint -> eval-mode policy net on `device`."""
    return _load_dnn(path, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dnn_a", required=True, help="checkpoint for policy A")
    ap.add_argument("--dnn_b", required=True, help="checkpoint for policy B")
    ap.add_argument("--deals", type=int, default=64)
    ap.add_argument("--seed0", type=int, default=20260802)
    ap.add_argument("--parallel", type=int, default=12)
    ap.add_argument("--dnn_temperature", type=float, default=1.0)
    ap.add_argument("--dnn_temperature_a", type=float, default=None,
                    help="override for side A (0 = greedy/argmax)")
    ap.add_argument("--dnn_temperature_b", type=float, default=None)
    ap.add_argument("--override_a", default=None, choices=["kan", "ankan", "nodaiminkan"],
                    help="diagnostic rule override wrapped around A (src/agents/dnn/overrides.py)")
    ap.add_argument("--override_b", default=None, choices=["kan", "ankan", "nodaiminkan"])
    ap.add_argument("--out", default="arena_dnn_result.json")
    ap.add_argument("--transcript", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dnn_policies = {}
    for side, path, override in (("A", args.dnn_a, args.override_a),
                                 ("B", args.dnn_b, args.override_b)):
        dnn_policies[side] = load_dnn(path, device)
        if override:
            from src.agents.dnn.overrides import KanOverride
            dnn_policies[side] = KanOverride(dnn_policies[side],
                                             ankan=override != "nodaiminkan",
                                             no_daiminkan=override != "ankan")

    temp_a = args.dnn_temperature if args.dnn_temperature_a is None else args.dnn_temperature_a
    temp_b = args.dnn_temperature if args.dnn_temperature_b is None else args.dnn_temperature_b
    seeds = [args.seed0 + i for i in range(args.deals)]
    rows = run_match(None, None, seeds, parallel=args.parallel,
                     transcript_path=args.transcript,
                     dnn_policies=dnn_policies, dnn_device=device,
                     dnn_temperature={"A": temp_a, "B": temp_b})

    diffs = [r["diff"] for r in rows]
    n = len(diffs)
    mean = sum(diffs) / max(n, 1)
    var = sum((d - mean) ** 2 for d in diffs) / max(n - 1, 1)
    ci = 1.96 * math.sqrt(var / max(n, 1))
    wins_a = sum(r["wins_a"] for r in rows)
    wins_b = sum(r["wins_b"] for r in rows)
    verdict = ("A stronger" if mean - ci > 0 else
               "B stronger" if mean + ci < 0 else "no significant difference")
    print(f"A = {args.dnn_a}  (T={temp_a})")
    print(f"B = {args.dnn_b}  (T={temp_b})")
    print(f"deals={n}  paired diff = {mean:+.1f} +/- {ci:.1f}  "
          f"wins {wins_a}:{wins_b}  => {verdict}")
    for side, net in dnn_policies.items():
        if getattr(net, "stats", None):
            print(f"override stats {side}: {dict(net.stats)}")
    json.dump({"a": args.dnn_a, "b": args.dnn_b, "deals": n, "mean_diff": mean,
               "ci95": ci, "wins_a": wins_a, "wins_b": wins_b,
               "verdict": verdict, "rows": rows},
              open(args.out, "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
