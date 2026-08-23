"""Behaviour-clone the DNN baseline from the SAME teacher as the LLM's SFT.

This is the apples-to-apples arm: identical teacher policy, identical
engine, identical decisions — only the function approximator differs
(1.4M-param CNN reading tensors vs 2B-param LLM reading text). It answers
"how much of the LLM's ability is the LLM, and how much is the teacher?"

The teacher functions are imported from scripts/generate_sft_data.py, the
exact code that produced data/sft_mahjong.jsonl, so no second
implementation can drift from it.
"""

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.dnn.encoder import action_to_index, encode_state, legal_mask  # noqa: E402
from src.agents.dnn.net import MahjongPolicyNet                    # noqa: E402
from src.tasks.mahjong.claims import _resolve_claims         # noqa: E402
from src.tasks.mahjong.table import PyMahjongTable                 # noqa: E402
import scripts.generate_sft_data as teacher                        # noqa: E402


def collect_teacher_games(n_games, seed0=0, max_steps=600):
    """Drive the engine with the teacher policy, recording (state, action)."""
    planes_l, scal_l, mask_l, act_l = [], [], [], []
    finished = 0
    for g in range(n_games):
        random.seed(seed0 + g)
        table = PyMahjongTable(randomize_round=True)
        guard = 0
        while not table.finished and guard < max_steps:
            guard += 1
            pid = table.turn
            legal = table.get_legal_actions(pid)
            if not legal:
                break
            a_xml, _ = teacher.pick_turn_action(table, pid, table.hands[pid], legal)
            if a_xml in legal:
                idx = action_to_index(a_xml)
                mask, lookup = legal_mask(legal)
                if idx is not None and idx in lookup:
                    p, s = encode_state(table, pid)
                    planes_l.append(p); scal_l.append(s)
                    mask_l.append(mask); act_l.append(idx)
            else:
                a_xml = legal[0]
            _, _, done, info = table.step(pid, a_xml)
            if done:
                break
            if not (info.get("discarded") or info.get("chankan")):
                continue

            candidates = []
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
                    p, s = encode_state(table, other)
                    planes_l.append(p); scal_l.append(s)
                    mask_l.append(mask); act_l.append(idx)
                import re as _re
                m = _re.search(r'type="(\w+)"', a2)
                candidates.append({"player_id": other, "parsed": a2,
                                   "type": m.group(1) if m else None,
                                   "reward": 0.0})
            executed, done = _resolve_claims(table, candidates)
            if done:
                break
            if not executed:
                if table.pending_kan:
                    table.resolve_pending_kan()
                else:
                    _, r_done = table.advance_turn()
                    if r_done:
                        break
        if table.finished:
            finished += 1
    return (torch.stack(planes_l), torch.stack(scal_l),
            torch.stack(mask_l), torch.tensor(act_l), finished)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--exp_dir", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    exp_dir = args.exp_dir or f"experiments/dnn_bc_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(exp_dir, exist_ok=True)
    json.dump(vars(args), open(f"{exp_dir}/config.json", "w"), indent=2)

    t0 = time.time()
    print(f"🎓 collecting teacher decisions from {args.games} games...", flush=True)
    P, S, M, A, finished = collect_teacher_games(args.games)
    print(f"   {len(A)} decisions from {finished}/{args.games} completed games "
          f"({time.time()-t0:.0f}s)", flush=True)

    n_val = int(len(A) * args.val_frac)
    perm = torch.randperm(len(A))
    val, tr = perm[:n_val], perm[n_val:]

    net = MahjongPolicyNet(channels=args.channels, blocks=args.blocks).to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    log = []
    for ep in range(1, args.epochs + 1):
        net.train()
        order = tr[torch.randperm(len(tr))]
        tot, correct, nll = 0, 0, 0.0
        for lo in range(0, len(order), args.batch):
            sel = order[lo:lo + args.batch]
            logits = net(P[sel].to(args.device), S[sel].to(args.device),
                         M[sel].to(args.device))
            y = A[sel].to(args.device)
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            nll += loss.item() * len(sel); tot += len(sel)
            correct += (logits.argmax(1) == y).sum().item()
        net.eval()
        with torch.no_grad():
            vlogits = net(P[val].to(args.device), S[val].to(args.device),
                          M[val].to(args.device))
            vy = A[val].to(args.device)
            vacc = (vlogits.argmax(1) == vy).float().mean().item()
            vnll = torch.nn.functional.cross_entropy(vlogits, vy).item()
        row = {"epoch": ep, "train_nll": nll / tot, "train_acc": correct / tot,
               "val_nll": vnll, "val_acc": vacc}
        log.append(row)
        print(f"[BC {ep}/{args.epochs}] train_nll={row['train_nll']:.4f} "
              f"train_acc={row['train_acc']:.1%} val_nll={vnll:.4f} "
              f"val_acc={vacc:.1%}", flush=True)
        torch.save({"state_dict": net.state_dict(), "channels": args.channels,
                    "blocks": args.blocks, "epoch": ep},
                   f"{exp_dir}/bc_ep{ep}.pt")
    json.dump(log, open(f"{exp_dir}/bc_log.json", "w"), indent=1)
    print(f"✅ BC done in {(time.time()-t0)/60:.1f} min -> {exp_dir}")


if __name__ == "__main__":
    main()
