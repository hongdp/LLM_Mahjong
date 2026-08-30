"""Exact human style reference from our downloaded houou logs (exp50).

The HUMAN_EXPERT bands in eval_style_profile.py are cited published
aggregates — ranges because sources, periods and player populations
differ. This script replaces them with measured values (mean ± 95% CI)
over data/tenhou/raw (20k+ 鳳凰卓 hanchan, the exact distribution our BC
models are trained on), using eval_style_profile's definitions:

  per seat-hand: agari / houjuu / riichi / call (>=1 meld incl ankan)
  per hand: draw rate; win_turn = total discards / 4 at the win
  tsumo_share = tsumo wins / wins;  win_points = winner delta incl sticks
  draw_tenpai_rate = tenpai seats at exhaustive draws (revealed hands)

Parses the raw XML directly (regex, independent of the verified mjai
converter). Also splits East vs South rounds so the "single-hand vs
hanchan ecology" caveat can be quantified instead of hand-waved.

Usage: python scripts/human_style_reference.py [--limit N] [--out path]
"""

import argparse
import glob
import json
import math
import os
import re
import sys

TAG = re.compile(r"<(\w+)((?:\s+\w+=\"[^\"]*\")*)\s*/?>")
ATTR = re.compile(r"(\w+)=\"([^\"]*)\"")
DRAW_T = {"T": 0, "U": 1, "V": 2, "W": 3}
DISC = {"D": 0, "E": 1, "F": 2, "G": 3}


def tally_game(xml: str, agg: dict):
    kyoku = None          # per-kyoku state
    def flush():
        nonlocal kyoku
        if kyoku is None:
            return
        k = kyoku
        seat_hands = 4
        band = "east" if k["round_idx"] < 4 else "south"
        for b in ("all", band):
            a = agg[b]
            a["hands"] += 1
            a["seat_hands"] += seat_hands
            a["wins"] += len(k["winners"])
            a["tsumo"] += k["tsumo"]
            a["deal_ins"] += len(k["deal_ins"])
            a["riichi"] += len(k["riichi"])
            a["called"] += len(k["called"])
            a["win_points"] += k["win_points"]
            if k["winners"]:
                a["win_turns"] += k["discards"] / 4.0
                a["win_n"] += 1
                if k["deal_ins"]:
                    a["dealin_turns"] += k["discards"] / 4.0
                    a["dealin_n"] += 1
            if k["is_draw"]:
                a["draws"] += 1
                if k["draw_tenpai"] is not None:
                    a["draw_seats"] += 4
                    a["draw_tenpai"] += k["draw_tenpai"]
        kyoku = None

    for m in TAG.finditer(xml):
        tag, attrs_s = m.group(1), m.group(2)
        if tag == "INIT":
            flush()
            at = dict(ATTR.findall(attrs_s))
            kyoku = {"round_idx": int(at["seed"].split(",")[0]),
                     "discards": 0, "winners": set(), "deal_ins": set(),
                     "riichi": set(), "called": set(), "tsumo": 0,
                     "win_points": 0.0, "is_draw": False, "draw_tenpai": None}
        elif kyoku is None:
            continue
        elif tag[0] in DISC and tag[1:].isdigit():
            kyoku["discards"] += 1
        elif tag == "N":
            at = dict(ATTR.findall(attrs_s))
            mm = int(at["m"])
            # every meld type counts for the >=1-meld proxy (ankan included)
            kyoku["called"].add(int(at["who"]))
        elif tag == "REACH":
            at = dict(ATTR.findall(attrs_s))
            if at.get("step") != "2":
                kyoku["riichi"].add(int(at["who"]))
        elif tag == "AGARI":
            at = dict(ATTR.findall(attrs_s))
            who, frm = int(at["who"]), int(at["fromWho"])
            kyoku["winners"].add(who)
            if who == frm:
                kyoku["tsumo"] += 1
            else:
                kyoku["deal_ins"].add(frm)
            sc = [int(x) for x in at["sc"].split(",")]
            kyoku["win_points"] += sc[who * 2 + 1] * 100   # delta incl sticks
        elif tag == "RYUUKYOKU":
            at = dict(ATTR.findall(attrs_s))
            if at.get("type") in (None, "", "nm"):        # exhaustive / nagashi
                kyoku["is_draw"] = True
                kyoku["draw_tenpai"] = sum(1 for i in range(4)
                                           if f"hai{i}" in at)
    flush()


def ci95(p, n):
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))


def summarize(a):
    n, s = a["hands"], a["seat_hands"]
    w = max(a["wins"], 1)
    out = {
        "hands": n,
        "agari_rate": a["wins"] / s, "agari_ci": ci95(a["wins"] / s, s),
        "houjuu_rate": a["deal_ins"] / s, "houjuu_ci": ci95(a["deal_ins"] / s, s),
        "riichi_rate": a["riichi"] / s, "riichi_ci": ci95(a["riichi"] / s, s),
        "call_rate": a["called"] / s, "call_ci": ci95(a["called"] / s, s),
        "draw_rate": a["draws"] / n,
        "tsumo_share": a["tsumo"] / w,
        "win_turn": a["win_turns"] / max(a["win_n"], 1),
        "dealin_turn": a["dealin_turns"] / max(a["dealin_n"], 1),
        "draw_tenpai_rate": a["draw_tenpai"] / max(a["draw_seats"], 1),
        "win_points": a["win_points"] / w,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/tenhou/raw")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/tenhou/human_style_reference.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.raw, "*", "*.mjlog")))
    if args.limit:
        files = files[: args.limit]
    fresh = lambda: {k: 0 for k in
                     ("hands", "seat_hands", "wins", "tsumo", "deal_ins",
                      "riichi", "called", "draws", "win_n", "dealin_n",
                      "draw_seats", "draw_tenpai")} | \
                    {k: 0.0 for k in ("win_points", "win_turns", "dealin_turns")}
    agg = {"all": fresh(), "east": fresh(), "south": fresh()}
    bad = 0
    for i, f in enumerate(files, 1):
        try:
            tally_game(open(f).read(), agg)
        except Exception:                                  # noqa: BLE001
            bad += 1
        if i % 5000 == 0:
            print(f"[{i}/{len(files)}]", flush=True)

    out = {band: summarize(a) for band, a in agg.items()}
    out["source"] = {"games": len(files) - bad, "parse_failures": bad,
                     "room": "四鳳南喰赤", "window": "2026-06..08 tenhou houou"}
    json.dump(out, open(args.out, "w"), indent=1)
    for band in ("all", "east", "south"):
        r = out[band]
        print(f"[{band}] hands={r['hands']} "
              f"agari {r['agari_rate']:.4f}±{r['agari_ci']:.4f}  "
              f"houjuu {r['houjuu_rate']:.4f}±{r['houjuu_ci']:.4f}  "
              f"riichi {r['riichi_rate']:.4f}±{r['riichi_ci']:.4f}  "
              f"call {r['call_rate']:.4f}±{r['call_ci']:.4f}", flush=True)
        print(f"      draw {r['draw_rate']:.4f}  tsumo_share {r['tsumo_share']:.4f}  "
              f"win_turn {r['win_turn']:.2f}  dealin_turn {r['dealin_turn']:.2f}  "
              f"draw_tenpai {r['draw_tenpai_rate']:.4f}  win_points {r['win_points']:.0f}",
              flush=True)


if __name__ == "__main__":
    main()
