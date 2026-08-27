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

Usage (10% scaling point ≈ 2000 games):
  PYTHONPATH=. python scripts/train_human_bc.py --arch cnn_m_r \
      --limit_games 2000 --epochs 4 --out experiments/exp45_bc_<ts>
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

# label -> reporting bucket, via ACTION_TYPES id (label // 34)
_BUCKET_OF = {0: "discard", 8: "discard", 1: "riichi", 9: "riichi",
              2: "call", 3: "call", 4: "call", 5: "win", 6: "win",
              7: "skip", 10: "kyuushu"}
BUCKETS = ["discard", "riichi", "call", "win", "skip", "kyuushu"]


def evaluate(net, loader, dev):
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
                sel = torch.tensor([_BUCKET_OF[int(t) // 34] == b for t in y])
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
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--holdout_games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if "m46" in a.arch:
        raise SystemExit("mortal-46 label remap not wired yet (see exp45 notes)")
    factory, _ = ZOO[a.arch]
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = factory().to(dev)
    variant = getattr(net, "encoder_variant", "v1")
    npar = sum(p.numel() for p in net.parameters())

    train_files = list_games(a.raw, holdout=False, limit=a.limit_games)
    hold_files = list_games(a.raw, holdout=True, limit=a.holdout_games)
    os.makedirs(a.out, exist_ok=True)
    print(f"🏗 {a.arch} ({npar/1e6:.1f}M, variant={variant}) "
          f"train {len(train_files)} games / holdout {len(hold_files)}", flush=True)

    ds = HumanBCDataset(train_files, variant=variant, seed=a.seed)
    hds = HumanBCDataset(hold_files, variant=variant, shuffle_buffer=1,
                         seed=a.seed)
    hloader = DataLoader(hds, batch_size=2048, num_workers=max(2, a.workers // 2))

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    hist, best = [], 0.0
    t0 = time.time()
    for e in range(a.epochs):
        ds.set_epoch(e)
        loader = DataLoader(ds, batch_size=a.batch, num_workers=a.workers,
                            persistent_workers=False)
        net.train()
        n_seen, loss_sum = 0, 0.0
        for planes, scalars, mask, y, _, _ in loader:
            with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                lg = net(planes.to(dev), scalars.to(dev), mask.to(dev))
                loss = torch.nn.functional.cross_entropy(lg.float(), y.to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            n_seen += len(y)
            loss_sum += float(loss.detach()) * len(y)
            if n_seen % (a.batch * 200) < a.batch:
                print(f"  [ep{e}] {n_seen} seen loss {loss_sum/n_seen:.4f} "
                      f"{n_seen/(time.time()-t0):.0f}/s", flush=True)
        m = evaluate(net, hloader, dev)
        m.update({"epoch": e, "train_loss": loss_sum / max(n_seen, 1),
                  "train_n": n_seen, "wall_min": (time.time() - t0) / 60})
        hist.append(m)
        print(f"[ep{e}] loss {m['train_loss']:.4f} acc {m['acc']:.4f} "
              f"defense {m['defense_acc']:.4f} riichi {m['acc_riichi']:.3f} "
              f"call {m['acc_call']:.3f}", flush=True)
        if m["acc"] > best:
            best = m["acc"]
            torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                        "arch": a.arch, "encoder_variant": variant,
                        "bc_acc": best, "train_games": len(train_files)},
                       os.path.join(a.out, f"bc_{a.arch}_best.pt"))
        with open(os.path.join(a.out, f"bc_{a.arch}_metrics.json"), "w") as f:
            json.dump(hist, f, indent=1)
    print(f"✅ {a.arch}: best holdout acc {best:.4f} "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
