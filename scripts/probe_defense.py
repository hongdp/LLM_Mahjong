"""Defense probe for DNN checkpoints — measures whether a policy has
learned CONDITIONAL folding, straight from engine state (no transcript
parsing). exp21 instrument; design: the five-probe suite, behavioral part.

Per model, plays G self-play games (4 copies) and records, at every
turn-phase discard decision where >=1 opponent has declared riichi and
we have not ("exposed"):

  - own 14-tile shanten bucket: tenpai (0) / mid (1) / weak (>=2)
  - whether the chosen discard is genbutsu vs EVERY riichi opponent
    (tile present in that opponent's river; honest lower bound of
    deliberate folding — no suji model in the engine)

and per (game, seat): exposure and whether that seat dealt into a ron
(parsed from result_summary's 放铳:玩家X tag, which is exact).

Headline metric: defense_iq = fold_weak - fold_tenpai. A policy that
never folds (or always folds) scores ~0; genuine push/fold judgment
scores positive.

Usage:
    python scripts/probe_defense.py --ckpt NAME=path [NAME=path ...] \
        --games 800 --workers 12 --out experiments/exp21_defense_probe.json
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_arena_dnn import load_dnn                      # noqa: E402
from src.agents.dnn.encoder import encode_state, legal_mask     # noqa: E402
from src.agents.dnn.selfplay import _choose                     # noqa: E402
from src.tasks.mahjong.table import PyMahjongTable, ACTION_RE   # noqa: E402
from src.tasks.mahjong.shanten import TileEfficiency, pad_for_melds  # noqa: E402

_te = TileEfficiency()
HOUJUU_RE = re.compile(r"放铳:玩家(\d)")


def _shanten(tiles, n_melds):
    n_melds = max(0, min(n_melds, (14 - len(tiles)) // 3))
    try:
        return _te.calculate_shanten(pad_for_melds(list(tiles), n_melds))
    except ValueError:
        return 8


def play_chunk(args):
    ckpt, seeds, temperature = args
    import random
    torch.set_num_threads(1)
    net = load_dnn(ckpt, "cpu")
    c = Counter()
    for seed in seeds:
        random.seed(seed)
        table = PyMahjongTable(randomize_round=True)
        table.text_obs = False
        exposed_seats = set()
        guard = 0
        while not table.finished and guard < 600:
            guard += 1
            pid = table.turn
            actions = table.get_legal_actions(pid)
            if not actions:
                break
            riichi_opps = [o for o in range(4)
                           if o != pid and table.riichi[o]]
            exposed = bool(riichi_opps) and not table.riichi[pid]
            hand14 = list(table.hands[pid])
            # V threat probe (suite #3): critic's danger awareness, bucketed
            # by own shanten so exposed/unexposed states are comparable.
            sh_now = _shanten(hand14, len(table.melds[pid]))
            vb = ("tenpai" if sh_now <= 0 else "mid" if sh_now == 1 else "weak")
            planes, scalars = encode_state(
                table, pid, variant=getattr(net, "encoder_variant", "v1"))
            vmask, _ = legal_mask(actions)
            with torch.no_grad():
                _, vv = net.forward_with_value(planes[None], scalars[None],
                                               vmask[None])
            vkey = f"v_{'exp' if exposed else 'un'}_{vb}"
            c[vkey + "_sum"] += float(vv)
            c[vkey + "_n"] += 1
            _, action_str = _choose(net, table, pid, actions, temperature,
                                    "cpu", cmode="none")
            m = ACTION_RE.search(action_str)
            if exposed and m and m.group(1) in ("discard", "riichi"):
                exposed_seats.add(pid)
                tile = m.group(2)
                bucket = vb
                genbutsu = all(
                    tile in (t.replace("*", "") for t in table.discards[o])
                    for o in riichi_opps)
                c[f"exp_{bucket}"] += 1
                if genbutsu:
                    c[f"fold_{bucket}"] += 1
            _, _, done, info = table.step(pid, action_str)
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
                _, a_str = _choose(net, table, other, options, temperature,
                                   "cpu", cmode="none")
                mm = ACTION_RE.search(a_str)
                candidates.append({"player_id": other, "parsed": a_str,
                                   "type": mm.group(1) if mm else None,
                                   "reward": 0.0})
            from src.agents.dnn.selfplay import _resolve_claims
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
        c["games"] += 1
        c["exposed_seat_games"] += len(exposed_seats)
        for hm in HOUJUU_RE.finditer(table.result_summary or ""):
            payer = int(hm.group(1))
            c["houjuu_any"] += 1
            if payer in exposed_seats:
                c["houjuu_exposed"] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="NAME=path pairs")
    ap.add_argument("--games", type=int, default=800)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=50000000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {}
    ctx = mp.get_context("fork")
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        seeds = [args.seed0 + i for i in range(args.games)]
        chunks = [seeds[i::args.workers] for i in range(args.workers)]
        with ctx.Pool(args.workers) as pool:
            parts = pool.map(play_chunk,
                             [(path, ch, args.temperature) for ch in chunks])
        c = Counter()
        for p in parts:
            c.update(p)
        rate = lambda f, e: round(c[f] / c[e], 3) if c[e] else None
        rec = {
            "games": c["games"],
            "exposed_steps": {b: c[f"exp_{b}"] for b in ("weak", "mid", "tenpai")},
            "fold_weak": rate("fold_weak", "exp_weak"),
            "fold_mid": rate("fold_mid", "exp_mid"),
            "fold_tenpai": rate("fold_tenpai", "exp_tenpai"),
            "defense_iq": None,
            "houjuu_given_exposed_seat": rate("houjuu_exposed", "exposed_seat_games"),
            "exposed_seat_games": c["exposed_seat_games"],
        }
        if rec["fold_weak"] is not None and rec["fold_tenpai"] is not None:
            rec["defense_iq"] = round(rec["fold_weak"] - rec["fold_tenpai"], 3)
        vgap = {}
        for b in ("weak", "mid", "tenpai"):
            e_n, u_n = c[f"v_exp_{b}_n"], c[f"v_un_{b}_n"]
            if e_n and u_n:
                vgap[b] = round(c[f"v_exp_{b}_sum"] / e_n
                                - c[f"v_un_{b}_sum"] / u_n, 3)
        rec["v_danger_gap"] = vgap
        out[name] = rec
        print(name, json.dumps(rec), flush=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
