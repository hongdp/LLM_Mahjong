"""Arena driver for DNN-vs-LLM (or DNN-vs-DNN) matches.

Uses the same duplicate-deal, both-orientation, paired-differential
protocol as scripts/run_arena.py so the numbers are directly comparable
to every LLM-vs-LLM result already on record.
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.dnn.net import MahjongPolicyNet, load_compatible  # noqa: E402
from src.tasks.mahjong.arena import run_match            # noqa: E402


def load_dnn(path, device):
    blob = torch.load(path, map_location=device)
    # .get(): the trainer has always written "arch" (None for the default
    # CNN) since exp10 — `"arch" in blob` sent every post-exp10 default-CNN
    # checkpoint into the zoo branch and KeyError'd on None (exp12 arenas).
    if blob.get("arch"):
        from src.agents.dnn.arch_zoo import ZOO
        net = ZOO[blob["arch"]][0]().to(device)
    else:
        net = MahjongPolicyNet(channels=blob.get("channels", 64),
                               blocks=blob.get("blocks", 3)).to(device)
    # Policy keys must load fully; critic keys (value*/hazard_head*) may be
    # absent (pre-critic) or shape-mismatched (exp11 privileged variants).
    # The value path is unused at play time, so both are harmless —
    # load_compatible enforces exactly that split and raises otherwise.
    skipped = load_compatible(net, blob["state_dict"])
    if skipped:
        print(f"[load] {path}: {len(skipped)} critic key(s) not loaded; "
              f"policy loaded fully", flush=True)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dnn_a", default=None, help="checkpoint for policy A")
    ap.add_argument("--dnn_b", default=None, help="checkpoint for policy B")
    ap.add_argument("--adapter_a", default=None, help="LoRA adapter for A")
    ap.add_argument("--adapter_b", default=None, help="LoRA adapter for B")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--deals", type=int, default=64)
    ap.add_argument("--seed0", type=int, default=20260802)
    ap.add_argument("--parallel", type=int, default=12)
    ap.add_argument("--dnn_temperature", type=float, default=1.0)
    ap.add_argument("--llm_temperature", type=float, default=0.9,
                    help="LLM sampling temperature in the arena. Lower = less\n                         evaluation noise AND (measured) better teacher fidelity.")
    ap.add_argument("--out", default="arena_dnn_result.json")
    ap.add_argument("--transcript", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dnn_policies = {}
    if args.dnn_a:
        dnn_policies["A"] = load_dnn(args.dnn_a, device)
    if args.dnn_b:
        dnn_policies["B"] = load_dnn(args.dnn_b, device)

    model = tokenizer = None
    if args.adapter_a or args.adapter_b:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        base = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map={"": 0})
        first = ("A", args.adapter_a) if args.adapter_a else ("B", args.adapter_b)
        model = PeftModel.from_pretrained(base, first[1], adapter_name=first[0])
        second = ("B", args.adapter_b) if first[0] == "A" else ("A", args.adapter_a)
        if second[1]:
            model.load_adapter(second[1], adapter_name=second[0])
        else:
            # only one LLM side: point the other name at the same adapter so
            # set_adapter() never fails; that side is served by the DNN anyway
            model.load_adapter(first[1], adapter_name=second[0])
        model.eval()

    seeds = [args.seed0 + i for i in range(args.deals)]
    rows = run_match(model, tokenizer, seeds, parallel=args.parallel,
                     transcript_path=args.transcript,
                     dnn_policies=dnn_policies or None,
                     dnn_device=device, dnn_temperature=args.dnn_temperature,
                     llm_temperature=args.llm_temperature)

    diffs = [r["diff"] for r in rows]
    n = len(diffs)
    mean = sum(diffs) / max(n, 1)
    var = sum((d - mean) ** 2 for d in diffs) / max(n - 1, 1)
    ci = 1.96 * math.sqrt(var / max(n, 1))
    wins_a = sum(r["wins_a"] for r in rows)
    wins_b = sum(r["wins_b"] for r in rows)
    verdict = ("A stronger" if mean - ci > 0 else
               "B stronger" if mean + ci < 0 else "no significant difference")
    label_a = args.dnn_a or args.adapter_a
    label_b = args.dnn_b or args.adapter_b
    print(f"A = {label_a}")
    print(f"B = {label_b}")
    print(f"deals={n}  paired diff = {mean:+.1f} +/- {ci:.1f}  "
          f"wins {wins_a}:{wins_b}  => {verdict}")
    json.dump({"a": label_a, "b": label_b, "deals": n, "mean_diff": mean,
               "ci95": ci, "wins_a": wins_a, "wins_b": wins_b,
               "verdict": verdict, "rows": rows},
              open(args.out, "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
