"""exp10: architecture sweep via behaviour cloning.

BC is the right harness for architecture search here: minutes per model,
a deterministic metric (held-out teacher agreement on identical states),
and no RL noise. The winner graduates to RL.

The dataset is collected ONCE (parallel workers, cached to disk) with BOTH
encoder variants per state, so binary-river and order-aware models train
on exactly the same decisions.

Reported per model: held-out agreement, params, and batch-1 CPU inference
latency — the quantity that actually gates self-play throughput.
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.dnn.arch_zoo import ZOO                       # noqa: E402
from src.agents.dnn.encoder import (action_to_index, encode_state,  # noqa: E402
                                    legal_mask)
from src.tasks.mahjong.orchestrator import _resolve_claims    # noqa: E402
from src.tasks.mahjong.table import PyMahjongTable            # noqa: E402
import scripts.generate_sft_data as teacher                   # noqa: E402
import re as _re                                              # noqa: E402


def _collect_worker(args):
    lo, n_games, seed0 = args
    torch.set_num_threads(1)
    P, P2, S, M, A, EP = [], [], [], [], [], []
    ep = 0
    for g in range(lo, lo + n_games):
        random.seed(seed0 + g)
        table = PyMahjongTable(randomize_round=True)
        guard, took = 0, False
        while not table.finished and guard < 600:
            guard += 1
            pid = table.turn
            legal = table.get_legal_actions(pid)
            if not legal:
                break
            a_xml, _ = teacher.pick_turn_action(table, pid, table.hands[pid], legal)
            if a_xml in legal and len(legal) > 1:
                idx = action_to_index(a_xml)
                mask, lookup = legal_mask(legal)
                if idx is not None and idx in lookup:
                    p1, s1 = encode_state(table, pid, with_order=False)
                    p2, _ = encode_state(table, pid, with_order=True)
                    P.append(p1.numpy()); P2.append(p2.numpy())
                    S.append(s1.numpy()); M.append(mask.numpy())
                    A.append(idx); EP.append(ep); took = True
            if a_xml not in legal:
                a_xml = legal[0]
            _, _, done, info = table.step(pid, a_xml)
            if done:
                break
            if not (info.get("discarded") or info.get("chankan")):
                continue
            cands = []
            for off in range(1, 4):
                other = (pid + off) % 4
                options = table.get_interrupt_actions(other)
                if len(options) == 1:
                    continue
                picked = teacher.pick_interrupt_action(table, other, options)
                a2 = picked[0] if picked else '<action type="skip" />'
                if a2 not in options:
                    a2 = '<action type="skip" />'
                idx = action_to_index(a2)
                mask, lookup = legal_mask(options)
                if idx is not None and idx in lookup:
                    p1, s1 = encode_state(table, other, with_order=False)
                    p2, _ = encode_state(table, other, with_order=True)
                    P.append(p1.numpy()); P2.append(p2.numpy())
                    S.append(s1.numpy()); M.append(mask.numpy())
                    A.append(idx); EP.append(ep); took = True
                m = _re.search(r'type="(\w+)"', a2)
                cands.append({"player_id": other, "parsed": a2,
                              "type": m.group(1) if m else None, "reward": 0.0})
            executed, done = _resolve_claims(table, cands)
            if done:
                break
            if not executed:
                if table.pending_kan:
                    table.resolve_pending_kan()
                else:
                    _, r_done = table.advance_turn()
                    if r_done:
                        break
        if took:
            ep += 1
    return (np.array(P, np.float32), np.array(P2, np.float32),
            np.array(S, np.float32), np.array(M), np.array(A), np.array(EP))


def build_dataset(path, games, workers, seed0=770000):
    per = games // workers
    jobs = [(w * per, per, seed0) for w in range(workers)]
    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_collect_worker, jobs)
    off, EPs = 0, []
    for part in parts:
        EPs.append(part[5] + off)
        off += (part[5].max() + 1) if len(part[5]) else 0
    data = {
        "planes": np.concatenate([p[0] for p in parts]),
        "planes_order": np.concatenate([p[1] for p in parts]),
        "scalars": np.concatenate([p[2] for p in parts]),
        "mask": np.concatenate([p[3] for p in parts]),
        "actions": np.concatenate([p[4] for p in parts]),
        "episode": np.concatenate(EPs),
    }
    torch.save(data, path)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--collect_workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dataset", default="experiments/arch_sweep/teacher_ds.pt")
    ap.add_argument("--out", default="experiments/arch_sweep/results.json")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--save_dir", default=None,
                    help="also save each trained model's weights here")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.dataset), exist_ok=True)
    if os.path.exists(args.dataset):
        data = torch.load(args.dataset, weights_only=False)  # our own cache, contains numpy arrays
        print(f"[ds] cached: {len(data['actions'])} decisions", flush=True)
    else:
        t0 = time.time()
        print(f"[ds] collecting {args.games} teacher games "
              f"({args.collect_workers} workers)...", flush=True)
        data = build_dataset(args.dataset, args.games, args.collect_workers)
        print(f"[ds] {len(data['actions'])} decisions in "
              f"{(time.time()-t0)/60:.1f} min", flush=True)

    ep = torch.from_numpy(data["episode"])
    n_ep = int(ep.max()) + 1
    te = ep >= (n_ep - int(n_ep * 0.15))
    tr = ~te
    Y = torch.from_numpy(data["actions"]).long()
    Msk = torch.from_numpy(data["mask"])
    Sc = torch.from_numpy(data["scalars"])
    Pl = {False: torch.from_numpy(data["planes"]),
          True: torch.from_numpy(data["planes_order"])}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ds] train {int(tr.sum())} / test {int(te.sum())} states "
          f"({n_ep} episodes)", flush=True)

    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out))
    names = args.only or list(ZOO)
    for name in names:
        factory, order = ZOO[name]
        torch.manual_seed(0)
        net = factory().to(dev)
        P = Pl[order]
        npar = sum(p.numel() for p in net.parameters())
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        idx = torch.nonzero(tr, as_tuple=True)[0]
        best = 0.0
        t0 = time.time()
        for e in range(args.epochs):
            net.train()
            perm = idx[torch.randperm(len(idx))]
            for lo in range(0, len(perm), args.batch):
                sel = perm[lo:lo + args.batch]
                logits = net(P[sel].to(dev), Sc[sel].to(dev), Msk[sel].to(dev))
                loss = torch.nn.functional.cross_entropy(logits, Y[sel].to(dev))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            net.eval()
            hit = tot = 0
            with torch.no_grad():
                tidx = torch.nonzero(te, as_tuple=True)[0]
                for lo in range(0, len(tidx), 8192):
                    sel = tidx[lo:lo + 8192]
                    lg = net(P[sel].to(dev), Sc[sel].to(dev), Msk[sel].to(dev))
                    hit += int((lg.argmax(1).cpu() == Y[sel]).sum())
                    tot += len(sel)
            best = max(best, hit / tot)
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                        "arch": name},
                       os.path.join(args.save_dir, f"{name}.pt"))
        # batch-1 CPU latency: the self-play regime
        net_cpu = factory()
        net_cpu.load_state_dict(net.state_dict())
        net_cpu.eval()
        torch.set_num_threads(1)
        with torch.no_grad():
            xs = (P[:1], Sc[:1], Msk[:1])
            for _ in range(10):
                net_cpu(*xs)
            t1 = time.time()
            for _ in range(100):
                net_cpu(*xs)
            lat_ms = (time.time() - t1) / 100 * 1000
        torch.set_num_threads(max(1, os.cpu_count() // 3))
        results[name] = {"val_agreement": best, "params": npar,
                         "cpu_batch1_ms": lat_ms,
                         "order_planes": order,
                         "train_s": round(time.time() - t0, 1)}
        print(f"[{name:16s}] val={best:.1%}  params={npar/1e6:.2f}M  "
              f"cpu_b1={lat_ms:.2f}ms  ({results[name]['train_s']}s)", flush=True)
        json.dump(results, open(args.out, "w"), indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
