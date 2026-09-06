"""exp67 follow-up: how much single-round return variance do oracle (hidden-state)
features explain beyond the public observation? Controlled offline fit, no RL.

Collect self-play episodes with `critic_feats=oracle` (public planes/scalars,
oracle cfeats, returns-to-go), then fit three ridge regressions on frozen
features and report held-out R^2:
  P   : public obs only        (planes flattened + scalars)
  P+O : public + oracle
  O   : oracle only
If R^2(P+O) - R^2(P) is ~0, the hidden state at decision time carries no usable
information about the round's outcome given the wall's *order* is unknown —
and no critic architecture would have fixed exp67.

  python scripts/oracle_info_probe.py --ckpt experiments/_anchors_epoch6/bc65.pt --games 2048 \
      --out experiments/probes/exp67_oracle_info.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

from scripts.run_arena_dnn import load_dnn                             # noqa: E402


def collect(ckpt, games, seed, device, workers, temperature):
    from src.agents.dnn.parallel_rollout import collect_parallel
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = load_dnn(ckpt, "cpu")
    cfg = dict(channels=blob.get("channels", 64), blocks=blob.get("blocks", 3), arch=blob.get("arch"),
               temperature=temperature, gamma=1.0, games_per_worker=16, rollout_temps=None, shaping=False,
               seed=seed, hanchan=False, hanchan_w_path=None, critic_feats="oracle", gpu_infer=True,
               gpu_infer_opponents=True, infer_max_batch=128, infer_wait_ms=0.0, infer_device=device,
               bf16_infer=False, arena=False, no_episodes=False, league_frac=0.0, league=[],
               encoder_variant=getattr(net, "encoder_variant", "v1"),
               action_space=getattr(net, "action_space", "native"))
    eps, _results = collect_parallel(net, games, cfg, workers, list(range(seed, seed + games)))
    return eps


def _standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)      # constant columns stay 0, never explode
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge_r2(Xtr, ytr, Xte, yte, lam, device):
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    Xte = torch.as_tensor(Xte, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    yte = torch.as_tensor(yte, dtype=torch.float32, device=device)
    Xtr, Xte = _standardize(Xtr, Xte)
    Xtr = torch.cat([Xtr, torch.ones(len(Xtr), 1, device=device)], 1)
    Xte = torch.cat([Xte, torch.ones(len(Xte), 1, device=device)], 1)
    A = Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1], device=device)
    w = torch.linalg.solve(A, Xtr.T @ ytr)
    pred = Xte @ w
    return float(1 - ((yte - pred) ** 2).mean() / yte.var())


def mlp_r2(Xtr, ytr, Xte, yte, device, epochs=30):
    torch.manual_seed(0)
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32, device=device); Xte = torch.as_tensor(Xte, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(ytr, dtype=torch.float32, device=device); yte = torch.as_tensor(yte, dtype=torch.float32, device=device)
    Xtr, Xte = _standardize(Xtr, Xte)
    ym, ys = ytr.mean(), ytr.std() + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 512), torch.nn.GELU(), torch.nn.Dropout(0.1),
                              torch.nn.Linear(512, 256), torch.nn.GELU(), torch.nn.Linear(256, 1)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-3)
    n = len(Xtr)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            loss = ((net(Xtr[idx]).squeeze(-1) - (ytr[idx] - ym) / ys) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(Xte[i:i + 8192]).squeeze(-1) for i in range(0, len(Xte), 8192)]) * ys + ym
    return float(1 - ((yte - pred) ** 2).mean() / yte.var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/_anchors_epoch6/bc65.pt")
    ap.add_argument("--games", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=75_000_000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    eps = collect(a.ckpt, a.games, a.seed, a.device, a.workers, a.temperature)
    P, O, S, R, K, T = [], [], [], [], [], []
    for e in eps:
        planes = e["planes"] if e.get("planes") is not None else None
        if planes is None or e.get("cfeats") is None:
            continue
        n = len(e["returns"])
        P.append(np.asarray(planes, dtype=np.float32).reshape(n, -1)); S.append(e["scalars"]); O.append(e["cfeats"])
        R.append(e["returns"]); K.extend([e["key"][0]] * n); T.extend(range(n, 0, -1))      # steps-to-go
    P = np.concatenate(P); S = np.concatenate(S); O = np.concatenate(O); R = np.concatenate(R).astype(np.float32)
    K = np.asarray(K); T = np.asarray(T)
    print(f"collected {len(R)} steps from {len(eps)} episodes [{time.time() - t0:.0f}s]; return std {R.std():.3f}", flush=True)
    # split by GAME (seed) so no leakage of the same deal across train/test
    seeds = np.unique(K); rng = np.random.default_rng(0); rng.shuffle(seeds)
    te_seeds = set(seeds[:len(seeds) // 5].tolist()); te = np.array([k in te_seeds for k in K]); tr = ~te
    pub = np.concatenate([P, S], 1)
    # compact, non-overfitting summaries (~30 dims): public scalars + oracle-derived
    # opponent tenpai flags / wait counts / shanten (a linear model can use these
    # directly, unlike one-hot planes)
    waits = O[:, 111:213].reshape(-1, 3, 34).sum(2)                    # waits per opponent
    tenpai = (waits > 0).astype(np.float32)
    shanten = O[:, 213:216]
    comp_pub = S[:, :20]
    comp_orc = np.concatenate([waits / 10.0, tenpai, shanten, O[:, 250:251]], 1)
    res = {"steps": int(len(R)), "episodes": len(eps), "return_std": float(R.std()), "test_frac": float(te.mean())}
    feats = (("public", pub), ("public+oracle", np.concatenate([pub, O], 1)), ("oracle", O),
             ("compact_public", comp_pub), ("compact_public+oracle", np.concatenate([comp_pub, comp_orc], 1)),
             ("compact_oracle", comp_orc))
    # all steps, then late-hand buckets: hidden info should matter most when few decisions remain
    for bucket, sel in (("all", np.ones(len(R), bool)), ("last5", T <= 5), ("last15", T <= 15), ("first15", T > (T.max() - 15))):
        res[bucket] = {"n": int(sel.sum()), "return_std": float(R[sel].std())}
        for name, X in feats:
            trs, tes = tr & sel, te & sel
            r_ridge = ridge_r2(X[trs], R[trs], X[tes], R[tes], lam=float(trs.sum()) * (0.5 if X.shape[1] > 100 else 0.01), device=a.device)
            r_mlp = mlp_r2(X[trs], R[trs], X[tes], R[tes], device=a.device, epochs=12) if X.shape[1] <= 100 else None
            res[bucket][name] = {"ridge_r2": round(r_ridge, 4), "mlp_r2": None if r_mlp is None else round(r_mlp, 4), "dim": int(X.shape[1])}
            print(bucket, name, res[bucket][name], flush=True)
    res["recorded"] = time.strftime("%F %T"); res["ckpt"] = a.ckpt; res["games"] = a.games
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
