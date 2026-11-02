"""Riichi-decision probe for DNN checkpoints (exp124 follow-up, 2026-11-01).

At every turn-phase decision of self-play (4 copies) where a riichi action is
legal and no win is available, record what the policy did:
  riichi  — declared riichi
  dama    — discarded a tile that keeps tenpai (a riichi-eligible tile) without declaring
  break   — discarded another tile (tenpai given up)
bucketed by the BEST riichi option (the eligible tile whose post-discard waits
have the most unseen copies):
  yaku : all / some / none of the waits win without riichi (HandCalculator, closed hand)
  wait : good (>=5 unseen waiting tiles) / bad
  junme: early (<=5 own discards) / mid (6-11) / late (>=12)
  opp  : an opponent has declared riichi or not
and, for 46-slot nets, the probability the policy puts on the riichi-declare
slot (mean per bucket). Headline: dama rate in the "should riichi" cell
(no yaku, good wait, not late, no opponent riichi) — a head bias shows up
there; flexibility shows up as dama concentrated in the yaku / opp cells.

Usage:
    python scripts/probe_riichi.py --ckpt NAME=path [...] --games 800 --workers 12 \
        --temperature 0 --out experiments/probes/riichi_X.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mahjong.hand_calculating.hand import HandCalculator                 # noqa: E402
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules  # noqa: E402
from mahjong.tile import TilesConverter                                  # noqa: E402
from mahjong.constants import EAST, SOUTH, WEST, NORTH                   # noqa: E402

from scripts.run_arena_dnn import load_dnn                               # noqa: E402
from src.agents.dnn import mortal_action as ma                           # noqa: E402
from src.agents.dnn.action_space import get_space                        # noqa: E402
from src.agents.dnn.encoder import encode_state                          # noqa: E402
from src.eval.probe_loop import play_probe_game                          # noqa: E402
from src.tasks.mahjong.table import ACTION_RE, norm_tile                 # noqa: E402
from src.tasks.mahjong.shanten import TileEfficiency                     # noqa: E402

_te = TileEfficiency()
_calc = HandCalculator()
WINDS = [EAST, SOUTH, WEST, NORTH]


def _to_136(tiles):
    """closed-hand tile strings ('5m', '7z') -> sorted 136 ids (no reds; reds are not yaku)."""
    by = {"m": [], "p": [], "s": [], "z": []}
    for t in tiles:
        by[t[-1]].append(t[:-1].replace("0", "5"))
    return TilesConverter.string_to_136_array(man="".join(by["m"]), pin="".join(by["p"]),
                                              sou="".join(by["s"]), honors="".join(by["z"]))


def _has_yaku_without_riichi(hand13, win_tile, player_wind, round_wind):
    ids = _to_136(hand13 + [win_tile])
    # the win tile is one of the ids of its kind; any copy works for the calculator
    want = _to_136([win_tile])[0] // 4
    win_id = next(i for i in ids if i // 4 == want)
    cfg = HandConfig(is_tsumo=False, is_riichi=False, player_wind=player_wind, round_wind=round_wind,
                     options=OptionalRules(has_open_tanyao=True, has_aka_dora=True))
    res = _calc.estimate_hand_value(ids, win_id, config=cfg)
    return res.error is None


def _unseen(tile, table, hand13):
    seen = hand13.count(tile)
    for o in range(4):
        seen += sum(1 for t in table.discards[o] if t.replace("*", "") == tile)
        for meld in table.melds[o]:
            seen += sum(1 for t in meld["tiles"] if t == tile)
    seen += sum(1 for t in table.dora_indicators if t == tile)
    return max(0, 4 - seen)


def play_chunk(args):
    ckpt, seeds, temperature = args
    torch.set_num_threads(1)
    net = load_dnn(ckpt, "cpu")
    space = get_space(net)
    is46 = getattr(net, "action_space", None) == "mortal46"
    variant = getattr(net, "encoder_variant", "v1")
    c = Counter()

    def before_turn(table, pid, actions):
        """Bucket the riichi option (None when riichi is not on the table)."""
        parsed = [ACTION_RE.search(a) for a in actions]
        types = {m.group(1) for m in parsed if m}
        riichi_tiles = sorted({norm_tile(m.group(2)) for m in parsed if m and m.group(1) == "riichi"})  # red "0m" -> "5m"
        if not (riichi_tiles and "tsumo" not in types and not table.riichi[pid] and not table.melds[pid]):
            return None
        hand14 = list(table.hands[pid])
        best = None
        for t in riichi_tiles:
            h13 = list(hand14); h13.remove(t)
            waits = _te.calculate_ukeire(h13)
            n_unseen = sum(_unseen(w, table, h13) for w in waits)
            if best is None or n_unseen > best[1]:
                best = (t, n_unseen, h13, waits)
        t, n_unseen, h13, waits = best
        pw = WINDS[(pid - table.dealer) % 4]; rw = WINDS[table.round_wind_idx]
        try:
            yk = [_has_yaku_without_riichi(h13, w, pw, rw) for w in waits]
            yaku = "all" if all(yk) else ("none" if not any(yk) else "some")
        except Exception:
            yaku = "err"
        wait = "good" if n_unseen >= 5 else "bad"
        j = table.discard_count[pid]
        junme = "early" if j <= 5 else ("mid" if j <= 11 else "late")
        opp = "opp" if any(table.riichi[o] for o in range(4) if o != pid) else "noopp"
        p_r = None
        if is46:
            planes, scalars = encode_state(table, pid, variant=variant)
            vmask, _ = space.mask(actions)
            with torch.no_grad():
                logits = net(planes[None], scalars[None], vmask[None])
            p_r = float(torch.softmax(logits[0].float(), 0)[ma.IDX_RIICHI])
        return (yaku, wait, junme, opp, set(riichi_tiles), p_r)

    def after_turn(rec, table, pid, actions, action_str):
        if rec is None:
            return
        m = ACTION_RE.search(action_str)
        kind = m.group(1) if m else "other"
        if kind == "riichi":
            choice = "riichi"
        elif kind == "discard":
            choice = "dama" if norm_tile(m.group(2)) in rec[4] else "break"
        else:
            choice = "other"
        key = f"{rec[0]}|{rec[1]}|{rec[2]}|{rec[3]}"
        c[f"n|{key}"] += 1
        c[f"{choice}|{key}"] += 1
        if rec[5] is not None:
            c[f"pr_sum|{key}"] += rec[5]

    for seed in seeds:
        play_probe_game(net, seed, temperature, "cpu",
                        before_turn=before_turn, after_turn=after_turn)
        c["games"] += 1
    return c


def summarize(c):
    keys = sorted({k.split("|", 1)[1] for k in c if k.startswith("n|")})
    cells = {}
    for k in keys:
        n = c[f"n|{k}"]
        cells[k] = {"n": n, **{ch: round(c[f"{ch}|{k}"] / n, 3) for ch in ("riichi", "dama", "break", "other")},
                    "p_riichi": round(c[f"pr_sum|{k}"] / n, 3) if f"pr_sum|{k}" in c else None}

    def agg(pred):
        n = sum(c[f"n|{k}"] for k in keys if pred(k))
        if n == 0:
            return None
        out = {"n": n, **{ch: round(sum(c[f"{ch}|{k}"] for k in keys if pred(k)) / n, 3)
                          for ch in ("riichi", "dama", "break")}}
        ps = sum(c[f"pr_sum|{k}"] for k in keys if pred(k) and f"pr_sum|{k}" in c)
        out["p_riichi"] = round(ps / n, 3) if ps else None
        return out
    head = {
        "all": agg(lambda k: True),
        "should_riichi(no_yaku,good,not_late,noopp)": agg(lambda k: k.startswith("none|good|") and "late" not in k and k.endswith("noopp")),
        "no_yaku": agg(lambda k: k.startswith("none|")), "all_yaku": agg(lambda k: k.startswith("all|")),
        "good_wait": agg(lambda k: "|good|" in k), "bad_wait": agg(lambda k: "|bad|" in k),
        "early": agg(lambda k: "|early|" in k), "late": agg(lambda k: "|late|" in k),
        "opp_riichi": agg(lambda k: k.endswith("|opp")), "no_opp": agg(lambda k: k.endswith("noopp")),
    }
    return {"games": c["games"], "headline": head, "cells": cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="NAME=path")
    ap.add_argument("--games", type=int, default=800)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=50000000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    results = {}
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        seeds = list(range(args.seed0, args.seed0 + args.games))
        chunks = [seeds[i::args.workers] for i in range(args.workers)]
        with mp.get_context("spawn").Pool(args.workers) as pool:
            parts = pool.map(play_chunk, [(path, ch, args.temperature) for ch in chunks])
        c = Counter()
        for p in parts:
            c.update(p)
        results[name] = summarize(c)
        print(name, json.dumps(results[name]["headline"]), flush=True)
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
