"""Arena driver: adapter A vs adapter B, duplicate deals, paired stats.

Usage (on a GPU VM):
    python -m scripts.run_arena --adapter_a <path> --adapter_b <path> \
        --deals 32 --parallel 12 [--value_facts] --out arena_result.json
"""

import argparse
import json
import math
import statistics

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_a", required=True, help="policy A (e.g. final RL checkpoint)")
    ap.add_argument("--adapter_b", required=True, help="policy B (e.g. SFT anchor)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--deals", type=int, default=32)
    ap.add_argument("--seed0", type=int, default=20260802)
    ap.add_argument("--parallel", type=int, default=12)
    ap.add_argument("--value_facts", action="store_true")
    ap.add_argument("--llm_temperature", type=float, default=0.9,
                    help="LLM sampling temperature in the arena. Lower = less\n                         evaluation noise AND (measured) better teacher fidelity.")
    ap.add_argument("--out", default="arena_result.json")
    ap.add_argument("--transcript", default=None,
                    help="write replayable game transcripts (rollout format)")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from src.tasks.mahjong.arena import run_match

    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, args.adapter_a, adapter_name="A")
    model.load_adapter(args.adapter_b, adapter_name="B")
    model.eval()
    print(f"A = {args.adapter_a}\nB = {args.adapter_b}")

    seeds = [args.seed0 + i for i in range(args.deals)]
    if args.transcript:
        import os
        os.makedirs(os.path.dirname(args.transcript), exist_ok=True)
        open(args.transcript, "w").close()
    rows = run_match(model, tok, seeds, value_facts=args.value_facts,
                     llm_temperature=args.llm_temperature,
                     parallel=args.parallel, log_path=args.out + ".log",
                     transcript_path=args.transcript)

    diffs = [r["diff"] for r in rows]
    wa = sum(r["wins_a"] for r in rows)
    wb = sum(r["wins_b"] for r in rows)
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    se = sd / math.sqrt(len(diffs)) if diffs else 0.0
    print(f"\ndeals={len(rows)} (games={2*len(rows)})")
    print(f"paired point differential (A−B): mean {mean:+.0f} ± {1.96*se:.0f} (95% CI)")
    print(f"wins: A {wa} / B {wb}")
    verdict = "A stronger" if mean - 1.96 * se > 0 else \
              "B stronger" if mean + 1.96 * se < 0 else "no significant difference"
    print(f"verdict: {verdict}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"adapter_a": args.adapter_a, "adapter_b": args.adapter_b,
                   "deals": len(rows), "mean_diff": mean, "ci95": 1.96 * se,
                   "wins_a": wa, "wins_b": wb, "verdict": verdict,
                   "rows": rows}, f, ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
