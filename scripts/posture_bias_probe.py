"""exp115: persistent posture-bias probe on the Rust engine (frozen policy, all T=0, paired same wall).

Each batch holds every wall twice (slots 2i and 2i+1 share a seed; K = batch size so slot i never changes game).
Even slot = the frozen policy; odd slot = the same policy, except seat 0 adds a persistent logit bias for the whole
hand. Delta per wall = seat-0 final points (biased slot) - seat-0 final points (baseline slot).

Bias knobs (only on rows of seat 0 in odd slots):
  f0/f1/f2 = +b on genbutsu discard slots (discard + discard0) when >= 1 opponent is in riichi and seat 0 is not,
             split by own shanten (0 / 1 / >= 2; Rust safe_info: genbutsu = riichi player's river + tiles others
             discarded after the riichi)
  c        = +b on chi / pon / kan slots at claim decisions
usage: posture_bias_probe.py --ckpt C.pt --walls 40000 --batch 2048 --seed0 S --conf "f1=2" [--conf "f2=2,f1=2" ...]
       --out results.json
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_conf(s):
    d = {"f0": 0.0, "f1": 0.0, "f2": 0.0, "c": 0.0}
    if s and s != "AA":
        for kv in s.split(","):
            k, v = kv.split("="); d[k.strip()] = float(v)
    return d


def run_conf(net, variant, conf, walls, batch, seed0, dev):
    import riichi_rs
    from src.agents.dnn.encoder import TYPE_TO_ID, TILE_TYPES
    q = float(riichi_rs.PLANE_Q)
    D, D0 = TYPE_TO_ID["discard"] * TILE_TYPES, TYPE_TO_ID["discard0"] * TILE_TYPES
    CALL = [TYPE_TO_ID[t] * TILE_TYPES for t in ("chi", "pon", "kan")]
    deltas, flips, rows_b = [], 0, 0
    stats = {"base": np.zeros(3), "bias": np.zeros(3)}      # win, deal-in, n
    for b0 in range(0, walls, batch // 2):
        nw = min(batch // 2, walls - b0)
        seeds = [seed0 + b0 + i for i in range(nw) for _ in range(2)]
        env = riichi_rs.VecEnv([int(s) for s in seeds], len(seeds), 0.995, False, 1.0, True, variant)
        res = {}
        with torch.no_grad():
            while not env.done():
                planes, scalars, mask, seats, gids = env.observe()
                n = planes.shape[0]
                if n > 0:
                    P = torch.from_numpy(planes).to(dev).view(n, -1, 34).float().div_(q)
                    S = torch.from_numpy(scalars).to(dev); M = torch.from_numpy(mask).to(dev)
                    logits = net(P, S, M).float()
                    g = np.asarray(gids, dtype=np.int64); st = np.asarray(seats, dtype=np.int64)
                    sel = (st == 0) & (g % 2 == 1)
                    if sel.any() and any(conf.values()):
                        genb, info = env.safe_info()
                        bias = torch.zeros_like(logits)
                        nr, sh = info[:, 0], info[:, 1]
                        own_turn = mask[:, D:D + 34].any(1) | mask[:, D0:D0 + 34].any(1)
                        for key, cond in (("f0", sh <= 0), ("f1", sh == 1), ("f2", sh >= 2)):
                            if conf[key]:
                                rws = np.nonzero(sel & (nr > 0) & cond & own_turn)[0]
                                if len(rws):
                                    gb = torch.from_numpy(genb[rws].astype(np.float32)).to(dev) * conf[key]
                                    r_t = torch.from_numpy(rws).to(dev)
                                    bias[r_t, D:D + 34] += gb; bias[r_t, D0:D0 + 34] += gb
                        if conf["c"]:
                            claim = np.zeros(n, bool)
                            for c0 in CALL:
                                claim |= mask[:, c0:c0 + 34].any(1)
                            rws = np.nonzero(sel & claim)[0]
                            if len(rws):
                                r_t = torch.from_numpy(rws).to(dev)
                                for c0 in CALL:
                                    bias[r_t, c0:c0 + 34] += conf["c"]
                        b_idx = (logits + bias).argmax(1); o_idx = logits.argmax(1)
                        touched = (bias.abs().sum(1) > 0)
                        rows_b += int(touched.sum()); flips += int(((b_idx != o_idx) & touched).sum())
                        idx = b_idx
                    else:
                        idx = logits.argmax(1)
                    lp = torch.log_softmax(logits, 1).gather(1, idx[:, None]).squeeze(1)
                    env.step(idx.cpu().numpy().astype(np.int64).tolist(), lp.cpu().numpy().tolist())
                else:
                    env.step([], [])
                for gm in env.drain_finished():
                    res[int(gm["slot"])] = gm
        for i in range(nw):
            a, bb = res[2 * i], res[2 * i + 1]
            assert int(a["seed"]) == int(bb["seed"])
            deltas.append(bb["points"][0] - a["points"][0])
            for tag, gm in (("base", a), ("bias", bb)):
                d0 = gm["points"][0] - gm["start_points"][0]
                stats[tag] += [int(d0 > 1500), int(d0 < -1500 and not any(gm["points"][o] - gm["start_points"][o] < 0 for o in (1, 2, 3) if o != 0) ), 1]
    d = np.asarray(deltas, float)
    return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))), "n": len(d),
            "frac_changed": float((d != 0).mean()), "rows_biased": rows_b, "argmax_flips": flips,
            "win_proxy": {k: float(v[0] / v[2]) for k, v in stats.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--walls", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=2048); ap.add_argument("--seed0", type=int, default=1_100_000)
    ap.add_argument("--conf", action="append", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from scripts.run_elo_league import load_dnn
    dev = "cuda"
    net = load_dnn(a.ckpt, dev).eval()
    variant = getattr(net, "encoder_variant", "v3r")
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for c in a.conf:
        t0 = time.time()
        r = run_conf(net, variant, parse_conf(c), a.walls, a.batch, a.seed0, dev)
        r["minutes"] = round((time.time() - t0) / 60, 1); out[c] = r
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"{c:14s} Δ {r['mean']:+7.1f} ± {r['se']:5.1f} pts/hand  (n={r['n']}, changed {r['frac_changed']:.3f}, "
              f"flips {r['argmax_flips']}/{r['rows_biased']}, {r['minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
