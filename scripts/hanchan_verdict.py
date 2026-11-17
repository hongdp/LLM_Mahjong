"""成对半庄裁决：检查点 × 种子段对同一对手打 2v2 复式半庄（双方 T=0），检查点之间同种子按场配对。

关键节点（判决里程碑 / 加冕候选 / run 终点）自动跑的那一个刻度，规则见 CLAUDE.md
"关键节点自动跑成对半庄裁决"。每个 (检查点, 种子段) 写一个 json，已存在的跳过，所以可以
随里程碑落地反复调用；汇总按判决段合并（SKILLS 10-15：半庄读数一律两段合并）。

指标：每 A 座位每半庄 uma = diff / 8（diff = 两个座位朝向的 (A uma − B uma) 之和，每场 4 个
A 座位半庄），SE 按场聚类；match-win 为场胜率。

Usage:
  python scripts/hanchan_verdict.py --opponent experiments/_anchors_epoch6/bc49.pt \
      --ckpts a192=.../games_192000000.pt s256=.../games_256000000.pt \
      --out-dir experiments/probes/hanchan_verdict_<run> --baseline a192
  python scripts/hanchan_verdict.py --analyze-only --ckpts a192= s256= --out-dir ... --baseline a192
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def seg_path(out_dir, tag, seed):
    return os.path.join(out_dir, f"{tag}_{seed}.json")


def run_segment(ckpt, opponent, n, seed, out, parallel, device):
    from scripts.run_elo_league import play_pair_vector

    t0 = time.time()
    scores, diffs, rows = play_pair_vector(ckpt, opponent, n, seed, parallel, device, temp_a=0.0, temp_b=0.0, hanchan=True)
    json.dump({"ckpt": ckpt, "opponent": opponent, "seed0": seed, "n": n, "diff": list(map(float, diffs)),
               "score": list(map(float, scores)), "rows": rows}, open(out, "w"))
    return (time.time() - t0) / 60


def load(out_dir, tag, seeds):
    d, s = [], []
    for seed in seeds:
        p = seg_path(out_dir, tag, seed)
        if not os.path.exists(p):
            return None, None
        j = json.load(open(p))
        d += j["diff"]
        s += j["score"]
    return np.asarray(d, float) / 8.0, np.asarray(s, float)


def mse(x):
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def analyze(out_dir, tags, seeds, baseline):
    base = load(out_dir, baseline, seeds)[0] if baseline else None
    out = {}
    print(f"== seeds={seeds}")
    for tag in tags:
        u, s = load(out_dir, tag, seeds)
        if u is None:
            print(f"{tag}: missing")
            continue
        m, se = mse(u)
        w, wse = mse(s)
        row = {"n": len(u), "uma": m, "uma_se": se, "z": m / se, "match_win": w, "match_win_se": wse}
        line = f"{tag}: n={len(u)} uma/seat/hanchan {m:+.0f} ± {se:.0f} (z {m / se:+.2f})  match-win {w:.4f} ± {wse:.4f}"
        if base is not None and tag != baseline and len(base) == len(u):
            pm, pse = mse(u - base)
            row.update({"vs_baseline": pm, "vs_baseline_se": pse})
            line += f"  | − {baseline} {pm:+.0f} ± {pse:.0f}"
        print(line)
        out[tag] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="tag=path；--analyze-only 时路径可留空（tag=）")
    ap.add_argument("--opponent", help="对手权重（常规为 bc49；加冕时另跑一次对现役冠军）")
    ap.add_argument("--seeds", default="202000000,202100000", help="判决段 seed0，逗号分隔；间距须 ≥ 段长")
    ap.add_argument("--matches", type=int, default=1500, help="每段场数")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--baseline", default=None, help="同场配对的基准 tag")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    specs = [c.split("=", 1) for c in a.ckpts]
    seeds = [int(s) for s in a.seeds.split(",")]
    os.makedirs(a.out_dir, exist_ok=True)
    if not a.analyze_only:
        if not a.opponent:
            ap.error("--opponent is required unless --analyze-only")
        for seed in seeds:
            for tag, ckpt in specs:
                out = seg_path(a.out_dir, tag, seed)
                if os.path.exists(out):
                    continue
                mins = run_segment(ckpt, a.opponent, a.matches, seed, out, a.parallel, a.device)
                print(f"{tag} seed={seed} done ({mins:.1f} min)", flush=True)
    summary = analyze(a.out_dir, [t for t, _ in specs], seeds, a.baseline)
    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
