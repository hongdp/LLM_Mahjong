"""exp119 diagnostic: does the frozen policy's per-tile trunk channel h_t (R^128) linearly encode genbutsu under riichi?
Collect seat-0 own-discard rows with >=1 riichi opponent, tiles held in hand; fit logistic regression h_t -> genbutsu;
report held-out AUC; save the probe direction u (normalized, rms-scaled like estimate_tile) for an oracle step."""
import sys, os, numpy as np, torch
sys.path.insert(0, "/home/hongdp/Workspace/LLM_Mahjong/.claude/worktrees/continue-task-3da2e7")
import scripts.es_posture_probe as E
from scripts.run_elo_league import load_dnn
import riichi_rs
ck, out = sys.argv[1], sys.argv[2]
dev = "cuda"; net = load_dnn(ck, dev).eval(); variant = getattr(net, "encoder_variant", "v3r")
D, D0 = E._consts(); q = float(riichi_rs.PLANE_Q)
X, Y = [], []
seeds = list(range(7_000_000, 7_000_000 + 4096))
env = riichi_rs.VecEnv(seeds, len(seeds), 0.995, False, 1.0, True, variant)
with torch.no_grad():
    while not env.done():
        planes, scalars, mask, seats, gids = env.observe(); n = planes.shape[0]
        if n == 0: env.step([], []); continue
        Pt = torch.from_numpy(planes).to(dev).view(n, -1, 34).float().div_(q); St = torch.from_numpy(scalars).to(dev)
        M = torch.from_numpy(mask).to(dev)
        t, ht = E._tile_feats(net, Pt, St)
        genb, info = env.safe_info()
        own = mask[:, D:D + 34] | mask[:, D0:D0 + 34]
        rows = np.nonzero((np.asarray(seats) == 0) & (info[:, 0] > 0) & own.any(1))[0]
        if len(rows):
            H = ht[torch.from_numpy(rows).to(dev)].float().cpu().numpy()     # [r, C, 34]
            for k, r in enumerate(rows):
                tl = np.nonzero(own[r])[0]
                X.append(H[k][:, tl].T); Y.append(genb[r, tl].astype(np.float32))
        logits = net.head(t).float().masked_fill(~M, float("-inf")); idx = logits.argmax(1)
        lp = torch.log_softmax(logits, 1).gather(1, idx[:, None]).squeeze(1)
        env.step(idx.cpu().numpy().astype(np.int64).tolist(), lp.cpu().numpy().tolist())
        env.drain_finished()
X = np.concatenate(X); Y = np.concatenate(Y); print("samples", X.shape, "pos rate", Y.mean())
mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd
rng = np.random.default_rng(0); perm = rng.permutation(len(Z)); tr, te = perm[: int(0.8 * len(Z))], perm[int(0.8 * len(Z)):]
Zt = torch.tensor(Z, dtype=torch.float32, device=dev); Yt = torch.tensor(Y, device=dev)
w = torch.zeros(Z.shape[1], device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True)
opt = torch.optim.LBFGS([w, b], max_iter=200)
def closure():
    opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(Zt[tr] @ w + b, Yt[tr]) + 1e-4 * w.pow(2).sum(); l.backward(); return l
opt.step(closure)
s = (Zt[te] @ w + b).detach().cpu().numpy(); y = Y[te]
o = np.argsort(s); ranks = np.empty(len(s)); ranks[o] = np.arange(len(s))
auc = (ranks[y == 1].sum() - (y == 1).sum() * ((y == 1).sum() - 1) / 2) / ((y == 1).sum() * (y == 0).sum())
print(f"held-out AUC {auc:.3f}")
# direction in raw h space (ignore centering offset; the score is used as a relative per-tile bias)
u = (w.detach().cpu() / torch.tensor(sd, dtype=torch.float32))
torch.save({"u": u, "auc": float(auc), "rms": E.tile_scale(net, variant, dev, 7_999_999), "note": "linear genbutsu probe direction (diagnostic oracle)"}, out)
print("saved", out)
