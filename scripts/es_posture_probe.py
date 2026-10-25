"""exp117: rule-free persistent-posture search on the policy head (frozen policy, Rust engine, all T=0, same wall).

Pure-line version of exp115: instead of a hand-picked rule feature (genbutsu), every wall draws its own random
low-rank perturbation of the LAST policy layer, dW = U V^T (U [A, r], V [512, r]), held fixed for the whole hand
(a persistent, state-dependent "posture": logits += (h V) U^T, h = post-ReLU head features). Only seat 0 is
perturbed; the other three seats are the frozen policy.

  estimate: slots 2i / 2i+1 play the same wall with +dW_i / -dW_i (antithetic). Delta_i = seat-0 points(+) - (-).
            G = sum_i Delta_i U_i V_i^T  (an evolution-strategies direction from common-random-number pairs).
            Also accumulates G on the first and second half of walls separately (split-half cosine = signal check).
  step:     W' = W + alpha * (G / ||G||_F) * ||W||_F   for each alpha; slots 2i / 2i+1 = base W / W' on fresh walls.
            Reports Delta (W' - W) per hand, plus the genbutsu rate when seat 0 is not tenpai under riichi threat
            (did the rule-free search rediscover exp115's direction?).  --save_step writes W' checkpoints so the
            standard pairdiff evaluator can play them against bc49.

usage: es_posture_probe.py estimate --ckpt C.pt --walls 400000 --rank 8 --sigma S --seed0 X --out G.pt
       es_posture_probe.py step --ckpt C.pt --G G.pt --alpha 0.01,0.02 --walls 200000 --seed0 Y --out r.json
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _feats(net, P, S):
    """post-ReLU input of the last policy layer, and the unperturbed (unmasked) logits."""
    h = net.head[1](net.head[0](net.trunk(P, S)))
    return h, net.head[2](h)


def _consts():
    from src.agents.dnn.encoder import TYPE_TO_ID, TILE_TYPES
    return TYPE_TO_ID["discard"] * TILE_TYPES, TYPE_TO_ID["discard0"] * TILE_TYPES


def play(net, variant, walls, batch, seed0, dev, logit_fn, genb_stats=False):
    """Generic paired loop. logit_fn(h, base_logits, slot_idx[n], seats[n], wall_off) -> logits.
    Returns per-wall seat-0 deltas (odd slot - even slot) and optional genbutsu stats per parity."""
    import riichi_rs
    q = float(riichi_rs.PLANE_Q)
    D, D0 = _consts()
    deltas = []
    gs = np.zeros((2, 2))          # [parity][genbutsu chosen, eligible rows]
    for b0 in range(0, walls, batch // 2):
        nw = min(batch // 2, walls - b0)
        seeds = [seed0 + b0 + i for i in range(nw) for _ in range(2)]
        env = riichi_rs.VecEnv([int(s) for s in seeds], len(seeds), 0.995, False, 1.0, True, variant)
        res = {}
        with torch.no_grad():
            while not env.done():
                planes, scalars, mask, seats, gids = env.observe()
                n = planes.shape[0]
                if n == 0:
                    env.step([], [])
                else:
                    Pt = torch.from_numpy(planes).to(dev).view(n, -1, 34).float().div_(q)
                    St = torch.from_numpy(scalars).to(dev); M = torch.from_numpy(mask).to(dev)
                    g = torch.as_tensor(np.asarray(gids, dtype=np.int64), device=dev)
                    st = torch.as_tensor(np.asarray(seats, dtype=np.int64), device=dev)
                    h, base = _feats(net, Pt, St)
                    logits = logit_fn(h, base.float(), g, st, b0).masked_fill(~M, float("-inf"))
                    idx = logits.argmax(1)
                    if genb_stats:
                        genb, info = env.safe_info()
                        nr, sh = info[:, 0], info[:, 1]
                        own = mask[:, D:D + 34].any(1) | mask[:, D0:D0 + 34].any(1)
                        gn, sn = np.asarray(gids) % 2, np.asarray(seats)
                        elig = (sn == 0) & (nr > 0) & (sh >= 1) & own
                        if elig.any():
                            ii = idx.cpu().numpy()
                            for r in np.nonzero(elig)[0]:
                                a = ii[r]
                                tile = a - D if D <= a < D + 34 else (a - D0 if D0 <= a < D0 + 34 else -1)
                                gs[gn[r], 1] += 1
                                gs[gn[r], 0] += float(tile >= 0 and genb[r, tile])
                    lp = torch.log_softmax(logits, 1).gather(1, idx[:, None]).squeeze(1)
                    env.step(idx.cpu().numpy().astype(np.int64).tolist(), lp.cpu().numpy().tolist())
                for gm in env.drain_finished():
                    res[int(gm["slot"])] = gm
        for i in range(nw):
            a, bb = res[2 * i], res[2 * i + 1]
            assert int(a["seed"]) == int(bb["seed"])
            deltas.append(bb["points"][0] - a["points"][0])
    return np.asarray(deltas, float), gs


def basis(a, net, variant, dev):
    """Pure, data-derived low-dim perturbation space: top-k PCs of head features h and top-m PCs of the raw logits,
    from the frozen policy's own self-play states (seat 0 rows, unperturbed)."""
    Hs, Ls = [], []

    def fn(h, base, g, st, b0):
        sel = (st == 0) & (g % 2 == 0)
        if bool(sel.any()) and sum(x.shape[0] for x in Hs) < 400000:
            Hs.append(h[sel].double().cpu()); Ls.append(base[sel].double().cpu())
        return base

    play(net, variant, a.walls, a.batch, a.seed0, dev, fn)
    H = torch.cat(Hs); L = torch.cat(Ls)
    H0 = H - H.mean(0); L0 = L - L.mean(0)
    eh, vh = torch.linalg.eigh(H0.T @ H0 / len(H0)); el, vl = torch.linalg.eigh(L0.T @ L0 / len(L0))
    k, m = a.k, a.m
    Bh = vh[:, -k:].flip(1); Bl = vl[:, -m:].flip(1)
    out = {"Bh": Bh.float(), "Bu": Bl.float(), "eig_h": eh.flip(0)[:k].float(), "eig_l": el.flip(0)[:m].float(),
           "var_frac_h": float(eh.flip(0)[:k].sum() / eh.sum()), "var_frac_l": float(el.flip(0)[:m].sum() / el.sum()),
           "rows": len(H), "ckpt": a.ckpt, "seed0": a.seed0}
    torch.save(out, a.out)
    print(f"basis -> {a.out}: rows {len(H)}, var explained h {out['var_frac_h']:.3f} / logits {out['var_frac_l']:.3f}")


def estimate_sub(a, net, variant, dev):
    """ES in the k x m subspace: dW = Bu Z Bh^T, Z ~ N(0, I_{m x k}) per wall, antithetic +/- on the same wall."""
    B = torch.load(a.basis, map_location=dev)
    Bh, Bu = B["Bh"].to(dev), B["Bu"].to(dev)
    k, m = Bh.shape[1], Bu.shape[1]
    sd_h = B["eig_h"].to(dev).sqrt()                    # whiten so every Z entry moves logits comparably
    gen = torch.Generator(device=dev)
    Gs = torch.zeros(m, k, device=dev, dtype=torch.float64); Gh = [torch.zeros_like(Gs), torch.zeros_like(Gs)]
    allD = []; t0 = time.time()
    for c0 in range(0, a.walls, a.chunk):
        nw = min(a.chunk, a.walls - c0)
        gen.manual_seed(a.seed0 * 7 + c0)
        Z = torch.randn(nw, m, k, device=dev, generator=gen)
        scale = a.sigma / np.sqrt(k)

        def fn(h, base, g, st, b0, Z=Z):
            sel = st == 0
            if not bool(sel.any()):
                return base
            w = (g[sel] // 2) + b0
            sign = 1.0 - 2.0 * (g[sel] % 2).float()
            ph = (h[sel] @ Bh) / sd_h                          # [n, k] whitened
            add = torch.einsum("nk,nmk->nm", ph, Z[w]) @ Bu.T   # [n, A]
            out = base.clone(); out[sel] += add * (scale * sign)[:, None]
            return out

        D, _ = play(net, variant, nw, a.batch, a.seed0 + c0, dev, fn)
        D = -D
        Dt = torch.as_tensor(D, device=dev, dtype=torch.float64)
        gz = torch.einsum("n,nmk->mk", Dt, Z.double())
        Gs += gz; Gh[0 if c0 < a.walls // 2 else 1] += gz
        allD.append(D)
        print(f"  chunk {c0 + nw}/{a.walls}  changed {(D != 0).mean():.3f}  {(time.time() - t0) / 60:.1f} min", flush=True)
    D = np.concatenate(allD)
    cos = float((Gh[0] * Gh[1]).sum() / (Gh[0].norm() * Gh[1].norm() + 1e-12))
    # full-space direction for `step`: dW = Bu diag(.) Gs (Bh / sd_h)^T
    G = (Bu.double() @ Gs @ (Bh.double() / sd_h.double()).T)
    torch.save({"G": G.float().cpu(), "G_sub": Gs.float().cpu(), "G_half_sub": [x.float().cpu() for x in Gh],
                "split_cos": cos, "sigma": a.sigma, "walls": a.walls, "seed0": a.seed0, "ckpt": a.ckpt,
                "basis": a.basis, "frac_changed": float((D != 0).mean())}, a.out)
    print(f"G saved -> {a.out}; split-half cosine {cos:+.4f} (null SE {1 / np.sqrt(m * k):.3f}); "
          f"frac changed {(D != 0).mean():.3f}", flush=True)


def _tile_feats(net, P, S):
    """trunk output split into per-tile channels [n, C, 34] (trunk lays out C*34 in (C, L) order, then scalars)."""
    t = net.trunk(P, S)
    C = net.stem.out_channels
    return t, t[:, :C * 34].view(-1, C, 34)


GATE = "none"          # exp119: "riichi" = the tile perturbation only acts when >= 1 opponent has declared riichi
GATE_ROWS = [0, 0]     # [seat-0 own-discard rows gated on, seat-0 own-discard rows total]


def play_tile(net, variant, walls, batch, seed0, dev, tile_fn, genb_stats=False):
    """play() variant whose hook gets per-tile trunk channels: tile_fn(ht[n,C,34], g, st, b0) -> [n,34] or None
    (added to discard and discard0 slots)."""
    D, D0 = _consts()

    import riichi_rs
    q = float(riichi_rs.PLANE_Q)
    deltas = []; gs = np.zeros((2, 2)); corr = []
    for b0 in range(0, walls, batch // 2):
        nw = min(batch // 2, walls - b0)
        seeds = [seed0 + b0 + i for i in range(nw) for _ in range(2)]
        env = riichi_rs.VecEnv([int(s) for s in seeds], len(seeds), 0.995, False, 1.0, True, variant)
        res = {}
        with torch.no_grad():
            while not env.done():
                planes, scalars, mask, seats, gids = env.observe()
                n = planes.shape[0]
                if n == 0:
                    env.step([], [])
                else:
                    Pt = torch.from_numpy(planes).to(dev).view(n, -1, 34).float().div_(q)
                    St = torch.from_numpy(scalars).to(dev); M = torch.from_numpy(mask).to(dev)
                    g = torch.as_tensor(np.asarray(gids, dtype=np.int64), device=dev)
                    st = torch.as_tensor(np.asarray(seats, dtype=np.int64), device=dev)
                    t, ht = _tile_feats(net, Pt, St)
                    base = net.head(t).float()
                    add = tile_fn(ht, g, st, b0)
                    if add is not None and GATE == "riichi":
                        _, ginfo = env.safe_info()
                        gmask = ginfo[:, 0] > 0
                        own0 = (np.asarray(seats) == 0) & (mask[:, D:D + 34].any(1) | mask[:, D0:D0 + 34].any(1))
                        GATE_ROWS[0] += int((gmask & own0).sum()); GATE_ROWS[1] += int(own0.sum())
                        add = add * torch.from_numpy(gmask.astype(np.float32)).to(dev)[:, None]
                    logits = base.clone()
                    if add is not None:
                        logits[:, D:D + 34] += add; logits[:, D0:D0 + 34] += add
                    logits = logits.masked_fill(~M, float("-inf"))
                    idx = logits.argmax(1)
                    if genb_stats:
                        genb, info = env.safe_info()
                        nr, sh = info[:, 0], info[:, 1]
                        own = mask[:, D:D + 34].any(1) | mask[:, D0:D0 + 34].any(1)
                        gn, sn = np.asarray(gids) % 2, np.asarray(seats)
                        elig = (sn == 0) & (nr > 0) & (sh >= 1) & own
                        if elig.any():
                            ii = idx.cpu().numpy()
                            for r in np.nonzero(elig)[0]:
                                a_ = ii[r]
                                tile = a_ - D if D <= a_ < D + 34 else (a_ - D0 if D0 <= a_ < D0 + 34 else -1)
                                gs[gn[r], 1] += 1; gs[gn[r], 0] += float(tile >= 0 and genb[r, tile])
                    lp = torch.log_softmax(logits, 1).gather(1, idx[:, None]).squeeze(1)
                    env.step(idx.cpu().numpy().astype(np.int64).tolist(), lp.cpu().numpy().tolist())
                for gm in env.drain_finished():
                    res[int(gm["slot"])] = gm
        for i in range(nw):
            a_, bb = res[2 * i], res[2 * i + 1]
            assert int(a_["seed"]) == int(bb["seed"])
            deltas.append(bb["points"][0] - a_["points"][0])
    return np.asarray(deltas, float), gs


def tile_scale(net, variant, dev, seed0):
    """rms of a unit-direction tile score u.h_t over seat-0 rows (for whitening sigma)."""
    acc = []

    def fn(ht, g, st, b0):
        sel = st == 0
        if bool(sel.any()) and len(acc) < 50:
            acc.append(ht[sel].double().pow(2).sum(1).mean().item() / ht.shape[1])   # E[(u.h)^2] for random unit u
        return None

    play_tile(net, variant, 512, 1024, seed0, dev, fn)
    return float(np.sqrt(np.mean(acc)))


def estimate_tile(a, net, variant, dev):
    """ES over a tile-shared direction u in R^C: every wall draws u_i ~ N(0, I), seat 0 adds
    +/- sigma * (u_i . h_t) / rms to the discard logit of tile t for the whole hand (antithetic on the same wall)."""
    C = net.stem.out_channels
    rms = tile_scale(net, variant, dev, a.seed0 + 999_999)
    gen = torch.Generator(device=dev)
    Gs = torch.zeros(C, device=dev, dtype=torch.float64); Gh = [torch.zeros_like(Gs), torch.zeros_like(Gs)]
    allD = []; t0 = time.time()
    for c0 in range(0, a.walls, a.chunk):
        nw = min(a.chunk, a.walls - c0)
        gen.manual_seed(a.seed0 * 7 + c0)
        Uu = torch.randn(nw, C, device=dev, generator=gen)

        def fn(ht, g, st, b0, Uu=Uu):
            sel = st == 0
            if not bool(sel.any()) or a.sigma == 0:
                return None
            w = (g[sel] // 2) + b0
            sign = 1.0 - 2.0 * (g[sel] % 2).float()
            sc = torch.einsum("nct,nc->nt", ht[sel].float(), Uu[w]) / (rms * np.sqrt(C)) * a.sigma
            out = torch.zeros(ht.shape[0], 34, device=dev); out[sel] = sc * sign[:, None]
            return out

        D, _ = play_tile(net, variant, nw, a.batch, a.seed0 + c0, dev, fn)
        D = -D
        Dt = torch.as_tensor(D, device=dev, dtype=torch.float64)
        gz = Dt @ Uu.double()
        Gs += gz; Gh[0 if c0 < a.walls // 2 else 1] += gz
        allD.append(D)
        print(f"  chunk {c0 + nw}/{a.walls}  changed {(D != 0).mean():.3f}  {(time.time() - t0) / 60:.1f} min", flush=True)
    D = np.concatenate(allD)
    cos = float((Gh[0] * Gh[1]).sum() / (Gh[0].norm() * Gh[1].norm() + 1e-12))
    torch.save({"u": Gs.float().cpu(), "u_half": [x.float().cpu() for x in Gh], "split_cos": cos, "rms": rms,
                "sigma": a.sigma, "walls": a.walls, "seed0": a.seed0, "ckpt": a.ckpt,
                "frac_changed": float((D != 0).mean()), "gate": GATE, "gate_rows": list(GATE_ROWS)}, a.out)
    print(f"gate={GATE}: seat-0 discard rows perturbed {GATE_ROWS[0]}/{GATE_ROWS[1]}", flush=True)
    print(f"u saved -> {a.out}; split-half cosine {cos:+.4f} (null SE {1 / np.sqrt(C):.3f}); "
          f"frac changed {(D != 0).mean():.3f}", flush=True)


def step_tile(a, net, variant, dev):
    """seat 0 (odd slots) adds alpha * (u_hat . h_t) / rms to every discard decision; base in even slots."""
    Ud = torch.load(a.G, map_location=dev)
    C = net.stem.out_channels
    u = Ud["u"].to(dev); u = u / u.norm(); rms = Ud["rms"]
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    out["_meta"] = {"u": a.G, "split_cos": Ud.get("split_cos"), "ckpt": a.ckpt, "seed0": a.seed0, "walls": a.walls}
    for al in [float(x) for x in a.alpha.split(",")]:
        t0 = time.time()

        def fn(ht, g, st, b0, al=al):
            sel = (st == 0) & (g % 2 == 1)
            if not bool(sel.any()) or al == 0:
                return None
            o = torch.zeros(ht.shape[0], 34, device=dev)
            o[sel] = torch.einsum("nct,c->nt", ht[sel].float(), u) / rms * al
            return o

        D, gs = play_tile(net, variant, a.walls, a.batch, a.seed0, dev, fn, genb_stats=True)
        r = {"mean": float(D.mean()), "se": float(D.std(ddof=1) / np.sqrt(len(D))), "n": len(D),
             "frac_changed": float((D != 0).mean()),
             "genb_rate_base": float(gs[0, 0] / max(gs[0, 1], 1)), "genb_rate_step": float(gs[1, 0] / max(gs[1, 1], 1)),
             "minutes": round((time.time() - t0) / 60, 1)}
        out[f"alpha={al}"] = r; json.dump(out, open(a.out, "w"), indent=1)
        print(f"alpha={al:<6} Δ {r['mean']:+7.1f} ± {r['se']:5.1f} pts/hand  changed {r['frac_changed']:.3f}  "
              f"genbutsu {r['genb_rate_base']:.3f} -> {r['genb_rate_step']:.3f}  ({r['minutes']} min)", flush=True)


def estimate(a, net, variant, dev):
    A, H = net.head[2].weight.shape
    r, sig = a.rank, a.sigma
    gen = torch.Generator(device=dev)
    G = torch.zeros(A, H, device=dev, dtype=torch.float64)
    Gh = [torch.zeros_like(G), torch.zeros_like(G)]
    allD = []
    t0 = time.time()
    for c0 in range(0, a.walls, a.chunk):
        nw = min(a.chunk, a.walls - c0)
        gen.manual_seed(a.seed0 * 7 + c0)
        U = torch.randn(nw, A, r, device=dev, generator=gen)
        V = torch.randn(nw, H, r, device=dev, generator=gen)
        scale = sig / np.sqrt(r * H)          # so (h V) U^T has logit std ~ sig * rms(h)

        def fn(h, base, g, st, b0, U=U, V=V):
            sel = st == 0
            if not bool(sel.any()):
                return base
            w = (g[sel] // 2) + b0             # wall index inside this chunk
            sign = 1.0 - 2.0 * (g[sel] % 2).float()          # even slot +dW, odd slot -dW
            hv = torch.einsum("nh,nhr->nr", h[sel], V[w])
            add = torch.einsum("nr,nar->na", hv, U[w]) * (scale * sign)[:, None]
            out = base.clone(); out[sel] += add
            return out

        # play() passes b0 = wall offset inside this chunk, so b0 + slot//2 indexes U/V
        D, _ = play(net, variant, nw, a.batch, a.seed0 + c0, dev, fn)
        D = -D                                 # play() returns odd - even = (-dW) - (+dW); flip to (+) - (-)
        Dt = torch.as_tensor(D, device=dev)
        G += torch.einsum("n,nar,nhr->ah", Dt, U.double(), V.double())
        half = 0 if c0 < a.walls // 2 else 1
        Gh[half] += torch.einsum("n,nar,nhr->ah", Dt, U.double(), V.double())
        allD.append(D)
        print(f"  chunk {c0 + nw}/{a.walls}  |Δ| mean {np.abs(D).mean():.0f}  changed {(D != 0).mean():.3f}  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    D = np.concatenate(allD)
    cos = float((Gh[0] * Gh[1]).sum() / (Gh[0].norm() * Gh[1].norm() + 1e-12))
    torch.save({"G": G.float().cpu(), "G_half": [x.float().cpu() for x in Gh], "split_cos": cos,
                "rank": r, "sigma": sig, "walls": a.walls, "seed0": a.seed0, "ckpt": a.ckpt,
                "delta_mean_abs": float(np.abs(D).mean()), "frac_changed": float((D != 0).mean())}, a.out)
    print(f"G saved -> {a.out}; split-half cosine {cos:+.4f}; frac changed {(D != 0).mean():.3f}", flush=True)


def step(a, net, variant, dev):
    Gd = torch.load(a.G, map_location=dev)
    G = Gd["G"].to(dev)
    W = net.head[2].weight.detach()
    dir_ = G / G.norm() * W.norm()
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    out["_meta"] = {"G": a.G, "split_cos": Gd.get("split_cos"), "ckpt": a.ckpt, "seed0": a.seed0, "walls": a.walls}
    for al in [float(x) for x in a.alpha.split(",")]:
        dW = al * dir_
        t0 = time.time()

        def fn(h, base, g, st, b0, dW=dW):
            sel = (st == 0) & (g % 2 == 1)
            if not bool(sel.any()) or al == 0:
                return base
            out_ = base.clone(); out_[sel] += h[sel] @ dW.T
            return out_

        D, gs = play(net, variant, a.walls, a.batch, a.seed0, dev, fn, genb_stats=True)
        r = {"mean": float(D.mean()), "se": float(D.std(ddof=1) / np.sqrt(len(D))), "n": len(D),
             "frac_changed": float((D != 0).mean()),
             "genb_rate_base": float(gs[0, 0] / max(gs[0, 1], 1)), "genb_rate_step": float(gs[1, 0] / max(gs[1, 1], 1)),
             "genb_rows": [int(gs[0, 1]), int(gs[1, 1])], "minutes": round((time.time() - t0) / 60, 1)}
        out[f"alpha={al}"] = r
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"alpha={al:<6} Δ {r['mean']:+7.1f} ± {r['se']:5.1f} pts/hand  changed {r['frac_changed']:.3f}  "
              f"genbutsu {r['genb_rate_base']:.3f} -> {r['genb_rate_step']:.3f}  ({r['minutes']} min)", flush=True)
        if a.save_step:
            ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
            ck["state_dict"]["head.2.weight"] = (W + dW).float().cpu()
            ck.pop("optimizer", None)
            ck["es_step"] = {"G": a.G, "alpha": al}
            p = a.save_step.replace("{alpha}", str(al))
            torch.save(ck, p)
            print(f"  saved {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["estimate", "step", "basis", "estimate_sub", "estimate_tile", "step_tile"])
    ap.add_argument("--gate", default="none", choices=["none", "riichi"])
    ap.add_argument("--basis"); ap.add_argument("--k", type=int, default=16); ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--walls", type=int, default=400000)
    ap.add_argument("--batch", type=int, default=2048); ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--seed0", type=int, default=3_000_000)
    ap.add_argument("--rank", type=int, default=8); ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--G"); ap.add_argument("--alpha", default="0,0.01,0.02,0.05")
    ap.add_argument("--save_step", default=""); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    global GATE
    GATE = a.gate
    from scripts.run_elo_league import load_dnn
    dev = "cuda"
    net = load_dnn(a.ckpt, dev).eval()
    variant = getattr(net, "encoder_variant", "v3r")
    {"estimate": estimate, "step": step, "basis": basis, "estimate_sub": estimate_sub,
     "estimate_tile": estimate_tile, "step_tile": step_tile}[a.mode](a, net, variant, dev)


if __name__ == "__main__":
    main()
