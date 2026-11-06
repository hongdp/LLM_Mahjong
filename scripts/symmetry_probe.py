"""Suit-symmetry breaking diagnostic for a policy checkpoint (exp61).

Plays greedy self-play with the checkpoint (identity frame drives the game)
and, at every multi-choice decision, evaluates the same state under all 6 suit
permutations mapped back to the identity frame. Reports how often the greedy
action changes under a non-identity permutation ("breaking rate"), how many of
the 5 views disagree, whether the 6-view average changes the pick, and how the
breaking rate depends on the identity top-2 logit margin.

  python scripts/symmetry_probe.py --ckpt experiments/_anchors_epoch6/bc49.pt \
      --games 200 --out experiments/probes/exp61_symmetry_bc49.json
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

from scripts.run_arena_dnn import load_dnn                             # noqa: E402
from src.agents.dnn.action_space import get_space                      # noqa: E402
from src.agents.dnn.encoder import encode_state                        # noqa: E402
from src.agents.dnn.selfplay import _resolve_claims                    # noqa: E402
from src.agents.dnn.symmetry import SuitSymmetrized, green_count       # noqa: E402
from src.tasks.mahjong.table import PyMahjongTable, ACTION_RE          # noqa: E402

PHASES = {"turn": 0, "claim": 1, "followup": 2}


class Probe:
    def __init__(self, ckpt, device, green_max):
        self.net = load_dnn(ckpt, device)
        self.sym = SuitSymmetrized(self.net)
        self.space = get_space(self.net)
        self.variant = getattr(self.net, "encoder_variant", "v1")
        self.dev = device
        self.green_max = green_max
        self.rows = []            # per decision: [phase, n_legal, margin, n_diff, sym_changed, skipped_green]
        self.n_skipped = 0

    @torch.no_grad()
    def decide(self, table, pid, actions, mode=None, phase="turn"):
        planes, sc = encode_state(table, pid, variant=self.variant)
        mask, lookup = self.space.mask(actions, mode)
        if int(mask.sum()) <= 1:
            slot = int(mask.nonzero()[0]) if int(mask.sum()) == 1 else 0
        else:
            views = self.sym.per_view_logits(planes[None].to(self.dev), sc[None].to(self.dev),
                                             mask[None].to(self.dev))[:, 0].cpu()   # [K, A]
            views = views.masked_fill(~mask[None], float("-inf"))
            picks = views.argmax(1)
            slot = int(picks[0])
            top2 = torch.topk(views[0], 2).values
            margin = float(top2[0] - top2[1]) if torch.isfinite(top2[1]) else float("inf")
            skipped = green_count(table.hands[pid], table.melds[pid]) > self.green_max
            n_diff = int((picks[1:] != picks[0]).sum())
            sym_pick = int(views.mean(0).argmax())
            if not skipped:
                self.rows.append([PHASES[phase], int(mask.sum()), round(margin, 4), n_diff,
                                  int(sym_pick != slot)])
            else:
                self.n_skipped += 1
        fu = self.space.follow_up(slot, actions, mode)
        if fu is not None:
            return self.decide(table, pid, actions, mode=fu, phase="followup")
        return self.space.resolve(slot, lookup)

    def play(self, seed):
        random.seed(seed)
        table = PyMahjongTable(randomize_round=True)
        table.text_obs = False
        guard = 0
        while not table.finished and guard < 600:
            guard += 1
            pid = table.turn
            actions = table.get_legal_actions(pid)
            if not actions:
                break
            chosen = self.decide(table, pid, actions, phase="turn")
            _, _, done, info = table.step(pid, chosen)
            if done:
                break
            if not (info.get("discarded") or info.get("chankan")):
                continue
            candidates = []
            for offset in range(1, 4):
                other = (pid + offset) % 4
                options = table.get_interrupt_actions(other)
                if len(options) == 1:
                    continue
                chosen = self.decide(table, other, options, phase="claim")
                mm = ACTION_RE.search(chosen)
                candidates.append({"player_id": other, "parsed": chosen,
                                   "type": mm.group(1) if mm else None, "reward": 0.0})
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


def summarize(rows, n_skipped):
    R = np.asarray(rows, dtype=np.float64)
    n = len(R)
    out = {"decisions": n, "skipped_green": n_skipped}
    if n == 0:
        return out
    phase, n_legal, margin, n_diff, sym_changed = R.T
    out["breaking_rate_any"] = float((n_diff > 0).mean())
    out["view_disagreement_rate"] = float(n_diff.mean() / 5)
    out["sym_avg_changes_pick_rate"] = float(sym_changed.mean())
    out["by_phase"] = {}
    for name, code in PHASES.items():
        sel = phase == code
        if sel.any():
            out["by_phase"][name] = {"n": int(sel.sum()), "breaking_rate": float((n_diff[sel] > 0).mean()),
                                     "sym_changes": float(sym_changed[sel].mean())}
    bins = [0, 0.1, 0.25, 0.5, 1.0, 2.0, np.inf]
    out["by_margin"] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (margin >= lo) & (margin < hi)
        if sel.any():
            out["by_margin"].append({"margin": f"[{lo},{hi})", "n": int(sel.sum()),
                                     "breaking_rate": float((n_diff[sel] > 0).mean())})
    out["margin_median_changed"] = float(np.median(margin[n_diff > 0])) if (n_diff > 0).any() else None
    out["margin_median_stable"] = float(np.median(margin[n_diff == 0])) if (n_diff == 0).any() else None
    out["by_n_legal"] = []
    for lo, hi in ((2, 3), (3, 6), (6, 10), (10, 99)):
        sel = (n_legal >= lo) & (n_legal < hi)
        if sel.any():
            out["by_n_legal"].append({"n_legal": f"[{lo},{hi})", "n": int(sel.sum()),
                                      "breaking_rate": float((n_diff[sel] > 0).mean())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=71_000_000)
    ap.add_argument("--green_max", type=int, default=7, help="skip symmetry stats when own green tiles exceed this")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pr = Probe(a.ckpt, a.device, a.green_max)
    t0 = time.time()
    for g in range(a.games):
        pr.play(a.seed0 + g)
        if (g + 1) % 20 == 0:
            s = summarize(pr.rows, pr.n_skipped)
            print(f"[{g + 1}/{a.games}] decisions={s['decisions']} breaking={s.get('breaking_rate_any', 0):.4f} "
                  f"view_disagree={s.get('view_disagreement_rate', 0):.4f} sym_changes={s.get('sym_avg_changes_pick_rate', 0):.4f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    s = summarize(pr.rows, pr.n_skipped)
    s.update({"ckpt": a.ckpt, "games": a.games, "seed0": a.seed0, "green_max": a.green_max,
              "rows_schema": ["phase(0 turn/1 claim/2 followup)", "n_legal", "margin", "n_diff_of_5", "sym_changed"],
              "rows": pr.rows, "recorded": time.strftime("%F %T")})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(s, open(a.out, "w"))
    print(json.dumps({k: v for k, v in s.items() if k != "rows"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
