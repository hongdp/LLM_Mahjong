"""Is mid-game value predictable AT ALL — from engine ground truth?

The LLM-hidden-state probe (probe_value_decodability.py) found only ~0.02
held-out explained variance. Two very different readings survive that:

  (A1) the LLM REPRESENTATION is the bottleneck -> richer context (v3)
       would give a critic something to read;
  (A2) mid-game mahjong outcome is INTRINSICALLY unpredictable at this
       horizon -> no critic helps, however good its features, and the
       whole variance-reduction premise is weak for this game.

This probe settles it without a GPU: it rebuilds ENGINE ground-truth
features (shanten, ukeire, melds, own points, dora held, tiles left,
riichi flag, ...) from the same prompts and fits the same episode-split
ridge against the same returns. If engine features also land near 0.02,
the game is the limit, not the representation (A2).
"""

import argparse
import glob
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.value_head import explained_variance
from src.tasks.mahjong.rewards import MahjongPotentialReward
from src.tasks.mahjong.shanten import pad_for_melds

HAND_RE = re.compile(r'手牌: ((?:[1-9][mpsz] )*[1-9][mpsz])')
FULU_RE = re.compile(r'私有[^\n]*?副露: ([^\n]*)')
POINTS_RE = re.compile(r'私有[^\n]*?点数: (\d+)')
LEFT_RE = re.compile(r'剩余牌数: (\d+)')
DORA_IND_RE = re.compile(r'宝牌指示牌: ([^\n,]*)')
RIICHI_RE = re.compile(r'已立直|立直宣言')
KYOTAKU_RE = re.compile(r'供托: (\d+)')


def engine_features(prompt, rw):
    """[shanten, ukeire, n_melds, points/1000, tiles_left/10, riichi,
        kyotaku, hand_size, energy] — None if the hand cannot be parsed."""
    hm = HAND_RE.search(prompt)
    if not hm:
        return None
    tiles = hm.group(1).split()
    fulu = FULU_RE.search(prompt)
    n_melds = fulu.group(1).count('(') if fulu else 0
    try:
        padded = pad_for_melds(tiles, n_melds)
        if len(tiles) % 3 == 2:
            # 14-tile turn decision: ukeire is undefined until a discard is
            # chosen, so score the BEST discard (same convention as the PBRS
            # reward's _pre_energy). Without this the probe silently keeps
            # only claim decisions — a biased 21% of states.
            ranked = rw.te.evaluate_discards_ranked(padded)
            cands = [(s_, len(u_)) for t, (s_, u_) in ranked.items() if t in tiles]
            if not cands:
                return None
            sh, uk = min(cands, key=lambda c: (c[0], -c[1]))
        else:
            sh = rw.te.calculate_shanten(padded)
            uk = len(rw.te.calculate_ukeire(padded))
    except Exception:
        return None
    pts = int(POINTS_RE.search(prompt).group(1)) if POINTS_RE.search(prompt) else 25000
    left = int(LEFT_RE.search(prompt).group(1)) if LEFT_RE.search(prompt) else 0
    kyo = int(KYOTAKU_RE.search(prompt).group(1)) if KYOTAKU_RE.search(prompt) else 0
    riichi = 1.0 if RIICHI_RE.search(prompt) else 0.0
    energy = -2.0 * sh + 0.05 * uk
    return [float(sh), float(uk), float(n_melds), pts / 1000.0, left / 10.0,
            riichi, float(kyo), float(len(tiles)), energy]


def ridge_fit_score(h_tr, y_tr, h_te, y_te, alpha):
    mu, sd = h_tr.mean(0, keepdim=True), h_tr.std(0, keepdim=True) + 1e-6
    xtr, xte = (h_tr - mu) / sd, (h_te - mu) / sd
    ymu = y_tr.mean()
    A = xtr.T @ xtr + alpha * torch.eye(xtr.shape[1], dtype=xtr.dtype)
    w = torch.linalg.solve(A, xtr.T @ (y_tr - ymu))
    return (explained_variance(xte @ w + ymu, y_te),
            explained_variance(xtr @ w + ymu, y_tr), w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_logs", nargs="+", required=True)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--out", default="engine_feature_probe.json")
    args = ap.parse_args()

    rw = MahjongPotentialReward(device="cpu", gamma=args.gamma)
    feats, rets, eps = [], [], []
    ep_counter = 0
    for path in sorted(sum([glob.glob(p) for p in args.from_logs], [])):
        txt = open(path, encoding="utf-8", errors="replace").read()
        for block in re.split(r"^--- Episode \d+ .*?---$", txt, flags=re.M)[1:]:
            steps = re.findall(
                r"^\[Step \d+\] Reward: (-?[\d.]+) \| Terminal: \w+\nPROMPT:\n(.*?)\nACTION:",
                block, flags=re.M | re.S)
            if not steps:
                continue
            R, ret_list = 0.0, []
            for r, _ in reversed(steps):
                R = float(r) + args.gamma * R
                ret_list.insert(0, R)
            kept = 0
            for (_, prompt), ret in zip(steps, ret_list):
                f = engine_features(prompt.strip(), rw)
                if f is None:
                    continue
                feats.append(f); rets.append(ret); eps.append(ep_counter); kept += 1
            if kept:
                ep_counter += 1
    X = torch.tensor(feats, dtype=torch.float32)
    y = torch.tensor(rets, dtype=torch.float32)
    e = torch.tensor(eps)
    print(f"[engine-probe] {len(X)} states / {ep_counter} episodes, "
          f"{X.shape[1]} engine features, return std {y.std():.3f}")

    n_ep = ep_counter
    n_te = max(1, int(n_ep * args.test_frac))
    te_eps = set(range(n_ep - n_te, n_ep))
    te = torch.tensor([int(i) in te_eps for i in e]); tr = ~te
    print(f"[engine-probe] split {int(tr.sum())} train / {int(te.sum())} test states")

    tr_eps = sorted({int(i) for i in e[tr]})
    K = 5
    folds = [set(tr_eps[i::K]) for i in range(K)]
    cv = {}
    for a in (0.1, 1.0, 10.0, 100.0, 1000.0):
        s = []
        for f in folds:
            va = torch.tensor([int(i) in f for i in e]) & tr
            t2 = tr & ~va
            if va.sum() < 5:
                continue
            ev, _, _ = ridge_fit_score(X[t2], y[t2], X[va], y[va], a)
            s.append(ev)
        cv[a] = sum(s) / max(len(s), 1)
        print(f"[engine-probe] alpha={a:<7g} CV_EV={cv[a]:+.4f}")
    best_a = max(cv, key=cv.get)
    te_ev, tr_ev, w = ridge_fit_score(X[tr], y[tr], X[te], y[te], best_a)
    names = ["shanten", "ukeire", "n_melds", "points_k", "tiles_left",
             "riichi", "kyotaku", "hand_size", "energy"]
    print(f"[engine-probe] alpha={best_a:g} train_EV={tr_ev:+.4f} "
          f"HELD-OUT test_EV={te_ev:+.4f}")
    print("[engine-probe] standardized weights:")
    for n, wi in sorted(zip(names, w.tolist()), key=lambda kv: -abs(kv[1])):
        print(f"    {n:11s} {wi:+.4f}")
    verdict = ("A2_game_intrinsically_unpredictable" if te_ev < 0.06 else
               "A1_llm_representation_is_the_bottleneck")
    print(f"[engine-probe] VERDICT: {verdict}  (LLM hidden-state probe was 0.0215)")
    json.dump({"n_states": len(X), "n_episodes": n_ep, "test_ev": te_ev,
               "train_ev": tr_ev, "alpha": best_a, "return_std": y.std().item(),
               "weights": dict(zip(names, w.tolist())), "verdict": verdict},
              open(args.out, "w"), indent=2)
    print(f"[engine-probe] wrote {args.out}")


if __name__ == "__main__":
    main()
