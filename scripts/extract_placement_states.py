"""W training set for exp55-D: (between-deal match state) -> final uma.

Every INIT tag in a houou mjlog carries the deal-start context (ten =
scores, oya = dealer, seed = "round,honba,kyotaku,d0,d1,dora"); the last
AGARI/RYUUKYOKU carries owari (final scores). One match with K deals
yields 4K rows (each seat's perspective). Final uma uses OUR convention:
placement from final scores (ties -> seat closer to starting East),
UMA = [+15,+5,-5,-15]k on the 25k start — identical to run_hanchan_arena.

Output: experiments/placement_value/states.npz
  X: float32 [N, 12] = [scores_rel_self/1e5 x4 (self first, then off-seat
     order), round_idx/8, honba/8, kyotaku/4e3, dealer_rel one-hot x4,
     deals_left_est/8]
  y: float32 [N] = final uma (points, our convention) for the self seat
  g: int64 [N] = game hash (holdout split by game)
"""
import glob
import hashlib
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TAG = re.compile(r"<(\w+)([^>]*?)/?>")
ATTR = re.compile(r'(\w+)="([^"]*)"')
UMA = [15000, 5000, -5000, -15000]


def rank_order(points, start_dealer=0):
    return sorted(range(4), key=lambda s: (-points[s], (s - start_dealer) % 4))


def parse_game(xml):
    states, owari = [], None
    for m in TAG.finditer(xml):
        tag = m.group(1)
        attrs = dict(ATTR.findall(m.group(2)))
        if tag == "INIT":
            seed = [int(x) for x in attrs["seed"].split(",")]
            ten = [int(x) * 100 for x in attrs["ten"].split(",")]
            states.append({"round": seed[0], "honba": seed[1],
                           "kyotaku": seed[2] * 1000, "oya": int(attrs["oya"]),
                           "ten": ten})
        elif "owari" in attrs:
            ow = attrs["owari"].split(",")
            owari = [int(float(ow[i * 2])) * 100 for i in range(4)]
    if owari is None or not states:
        return []
    order = rank_order(owari)
    uma = [0] * 4
    for rank, seat in enumerate(order):
        uma[seat] = owari[seat] - 25000 + UMA[rank]
    rows = []
    n_deals = len(states)
    for i, st in enumerate(states):
        for me in range(4):
            rel = [st["ten"][(me + k) % 4] for k in range(4)]
            drel = (st["oya"] - me) % 4
            x = ([v / 1e5 for v in rel]
                 + [st["round"] / 8.0, st["honba"] / 8.0,
                    st["kyotaku"] / 4000.0]
                 + [1.0 if drel == k else 0.0 for k in range(4)]
                 + [(n_deals - i) / 8.0])
            rows.append((x, uma[me]))
    return rows


def main():
    out_dir = "experiments/placement_value"
    os.makedirs(out_dir, exist_ok=True)
    X, y, g = [], [], []
    files = sorted(glob.glob("data/tenhou/raw/*/*"))
    skipped = 0
    for i, f in enumerate(files):
        try:
            rows = parse_game(open(f, encoding="utf-8", errors="ignore").read())
        except Exception:
            skipped += 1
            continue
        if not rows:
            skipped += 1
            continue
        gh = int(hashlib.md5(os.path.basename(f).encode()).hexdigest()[:12], 16)
        for x, u in rows:
            X.append(x)
            y.append(u)
            g.append(gh)
        if (i + 1) % 5000 == 0:
            print(f"[{i+1}/{len(files)}] rows={len(y)}", flush=True)
    np.savez_compressed(os.path.join(out_dir, "states.npz"),
                        X=np.array(X, dtype=np.float32),
                        y=np.array(y, dtype=np.float32),
                        g=np.array(g, dtype=np.int64))
    print(f"DONE games={len(files)-skipped} skipped={skipped} rows={len(y)}"
          f" -> {out_dir}/states.npz")


if __name__ == "__main__":
    main()
