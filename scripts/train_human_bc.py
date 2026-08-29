"""exp45: behaviour-clone any ZOO architecture from tenhou houou logs.

Streaming pipeline — no tensor files. `HumanBCDataset` workers replay
mjlogs through the verified MJAI bridge and emit (planes, scalars, mask,
label); the encoder variant is read off the constructed net, so every
arm (v1r CNNs, ConvFormer, handset, v4 HRF) trains from one script.
Mortal-46 needs a label remap and is NOT wired yet (--arch will refuse).

Holdout is per game (stable hash; SKILLS:106). Metrics per epoch:
overall top-1 plus the preregistered buckets — discard / riichi / call
(chi+pon+kan) / win (ron+tsumo) / skip, and the defensive slice
(decisions while an opponent riichi is live).

Convergence protocol (the preregistered "同早停规则"): train until the
holdout accuracy gains < min_delta for `patience` consecutive epochs
(cap max_epochs) — a fixed-epoch budget systematically biases against
slow-hot architectures (exp19's ConvFormer lesson).

Usage (10% scaling point ≈ 2000 games):
  PYTHONPATH=. python scripts/train_human_bc.py --arch cnn_m_r \
      --limit_games 2000 --out experiments/exp45_bc_<ts>
"""

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.dnn.arch_zoo import ZOO                                # noqa: E402
from src.agents.dnn.human_bc_data import HumanBCDataset, list_games    # noqa: E402

# label -> reporting bucket. native: via ACTION_TYPES id (label // 34);
# mortal46: declare/select slots (tile-select second steps count as discard)
_BUCKET_OF = {0: "discard", 8: "discard", 1: "riichi", 9: "riichi",
              2: "call", 3: "call", 4: "call", 5: "win", 6: "win",
              7: "skip", 10: "kyuushu"}
BUCKETS = ["discard", "riichi", "call", "win", "skip", "kyuushu"]


def bucket_of(label: int, space: str) -> str:
    if space == "mortal46":
        return {37: "riichi", 38: "call", 39: "call", 40: "call",
                41: "call", 42: "call", 43: "win", 44: "kyuushu",
                45: "skip"}.get(label, "discard")
    return _BUCKET_OF[label // 34]


def evaluate(net, loader, dev, space="native"):
    net.eval()
    hit = {b: 0 for b in BUCKETS}
    tot = {b: 0 for b in BUCKETS}
    dhit = dtot = 0
    with torch.no_grad():
        for planes, scalars, mask, y, phase, vsr in loader:
            with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                lg = net(planes.to(dev), scalars.to(dev), mask.to(dev))
            pred = lg.float().argmax(1).cpu()
            ok = pred == y
            for b in BUCKETS:
                sel = torch.tensor([bucket_of(int(t), space) == b for t in y])
                hit[b] += int(ok[sel].sum())
                tot[b] += int(sel.sum())
            dsel = vsr.bool()
            dhit += int(ok[dsel].sum())
            dtot += int(dsel.sum())
    n = sum(tot.values())
    return {"acc": sum(hit.values()) / max(n, 1), "n": n,
            "defense_acc": dhit / max(dtot, 1), "defense_n": dtot,
            **{f"acc_{b}": hit[b] / max(tot[b], 1) for b in BUCKETS},
            **{f"n_{b}": tot[b] for b in BUCKETS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--raw", default="data/tenhou/raw")
    ap.add_argument("--limit_games", type=int, default=0,
                    help="cap on TRAIN games (holdout always full 10%)")
    ap.add_argument("--max_epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=3,
                    help="stop after this many epochs without >min_delta gain")
    ap.add_argument("--min_delta", type=float, default=0.0005)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--holdout_games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--riichi_weight", type=float, default=1.0,
                    help="CE weight for riichi-labelled samples (exp48 arm C)")
    ap.add_argument("--lr_schedule", choices=["const", "cosine"], default="const",
                    help="cosine anneals per epoch over max_epochs to 0.1x (exp49)")
    ap.add_argument("--cache_dir", default=None,
                    help="materialized shards (materialize_bc.py); replaces "
                         "per-epoch replay with mmap reads (exp51 optimization)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    factory, _ = ZOO[a.arch]
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = factory().to(dev)
    variant = getattr(net, "encoder_variant", "v1")
    from src.agents.dnn.action_space import get_space, space_of_arch
    aspace = getattr(net, "action_space", None) or space_of_arch(a.arch)
    npar = sum(p.numel() for p in net.parameters())

    os.makedirs(a.out, exist_ok=True)
    if a.cache_dir:
        from src.agents.dnn.human_bc_data import MaterializedBCDataset
        ds = MaterializedBCDataset(a.cache_dir, "train")
        hds = MaterializedBCDataset(a.cache_dir, "holdout")
        print(f"🏗 {a.arch} ({npar/1e6:.1f}M, variant={variant}, space={aspace}) "
              f"cache {a.cache_dir}: train {len(ds)} rows / holdout {len(hds)} rows",
              flush=True)
        hloader = DataLoader(hds, batch_size=2048, num_workers=4)
    else:
        train_files = list_games(a.raw, holdout=False, limit=a.limit_games)
        hold_files = list_games(a.raw, holdout=True, limit=a.holdout_games)
        print(f"🏗 {a.arch} ({npar/1e6:.1f}M, variant={variant}, space={aspace}) "
              f"train {len(train_files)} games / holdout {len(hold_files)}", flush=True)
        ds = HumanBCDataset(train_files, variant=variant, seed=a.seed,
                            action_space=aspace)
        hds = HumanBCDataset(hold_files, variant=variant, shuffle_buffer=1,
                             seed=a.seed, action_space=aspace)
        hloader = DataLoader(hds, batch_size=2048, num_workers=max(2, a.workers // 2))

    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(a.out, f"tensorboard_{a.arch}"))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
                 opt, T_max=a.max_epochs, eta_min=a.lr * 0.1)
             if a.lr_schedule == "cosine" else None)
    hist, best, stale = [], 0.0, 0
    t0 = time.time()
    for e in range(a.max_epochs):
        if a.cache_dir:
            loader = DataLoader(ds, batch_size=a.batch, shuffle=True,
                                num_workers=a.workers, pin_memory=True,
                                persistent_workers=False)
        else:
            ds.set_epoch(e)
            loader = DataLoader(ds, batch_size=a.batch, num_workers=a.workers,
                                persistent_workers=False)
        net.train()
        n_seen, loss_sum, t_ep = 0, 0.0, time.time()
        for planes, scalars, mask, y, _, _ in loader:
            with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                lg = net(planes.to(dev), scalars.to(dev), mask.to(dev))
                yd = y.to(dev)
                if a.riichi_weight != 1.0:
                    is_r = ((yd // 34 == 1) | (yd // 34 == 9)) if aspace == "native" \
                        else (yd == 37)
                    w = torch.where(is_r, a.riichi_weight, 1.0).float()
                    ce = torch.nn.functional.cross_entropy(lg.float(), yd,
                                                           reduction="none")
                    loss = (ce * w).sum() / w.sum()
                else:
                    loss = torch.nn.functional.cross_entropy(lg.float(), yd)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            n_seen += len(y)
            loss_sum += float(loss.detach()) * len(y)
            if n_seen % (a.batch * 200) < a.batch:
                print(f"  [ep{e}] {n_seen} seen loss {loss_sum/n_seen:.4f} "
                      f"{n_seen/(time.time()-t_ep):.0f}/s", flush=True)
        m = evaluate(net, hloader, dev, aspace)
        m.update({"epoch": e, "train_loss": loss_sum / max(n_seen, 1),
                  "train_n": n_seen, "wall_min": (time.time() - t0) / 60})
        hist.append(m)
        for k in ("acc", "defense_acc", "acc_riichi", "acc_call",
                  "acc_discard", "train_loss"):
            tb.add_scalar(k, m[k], e)
        tb.flush()
        print(f"[ep{e}] loss {m['train_loss']:.4f} acc {m['acc']:.4f} "
              f"defense {m['defense_acc']:.4f} riichi {m['acc_riichi']:.3f} "
              f"call {m['acc_call']:.3f}", flush=True)
        if m["acc"] > best + a.min_delta:
            best, stale = m["acc"], 0
            torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                        "arch": a.arch, "encoder_variant": variant,
                        "bc_acc": best,
                       "train_games": (a.limit_games or "cache"),
                        "epoch": e},
                       os.path.join(a.out, f"bc_{a.arch}_best.pt"))
        else:
            stale += 1
        with open(os.path.join(a.out, f"bc_{a.arch}_metrics.json"), "w") as f:
            json.dump(hist, f, indent=1)
        if sched is not None:
            sched.step()
        if stale >= a.patience:
            print(f"[early-stop] no >{a.min_delta} gain for {a.patience} epochs", flush=True)
            break
    print(f"✅ {a.arch}: best holdout acc {best:.4f} "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
