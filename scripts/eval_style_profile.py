"""Style profile of a self-play policy vs human reference statistics.

Self-play win rate cannot measure absolute strength (four copies of one net
split decisive games by symmetry at ANY skill level). What CAN be compared
against published human统计 is the STYLE vector: per-seat agari rate,
deal-in (houjuu) rate, riichi rate, call rate, and the table's draw rate.
The deal-in rate is the sharpest mirror — a reckless policy can fake a
human-like draw rate but not a human-like houjuu rate.

Caveats printed with the output: our games are single hands (no hanchan
placement strategy), EMA RCR-flavored rules, and the call rate counts ankan
as "called" (proxy). Human reference numbers must be cited in the report,
not hardcoded here as truth.

Usage:
  python scripts/eval_style_profile.py --ckpt A=path1.pt B=path2.pt \
      --games 4000 --workers 15
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.dnn.selfplay import play_game                 # noqa: E402
from scripts.run_arena_dnn import load_dnn                    # noqa: E402

WIN_RE = re.compile(r"玩家(\d)\s*(?:自摸|荣和|抢杠)")
HOUJUU_RE = re.compile(r"放铳:玩家(\d)")
TSUMO_RE = re.compile(r"玩家(\d)\s*自摸")
DRAW_TENPAI_RE = re.compile(r"流局 \| 听牌: \[([^\]]*)\]")

# Human-expert reference bands (Tenhou 鳳凰卓 published aggregates, per-hand).
# Caveats: humans play full hanchan (orasu/placement distortions, continuing
# scores); our env is one hand with randomized context. Direction, not decimals.
# 2026-08-28: if data/tenhou/human_style_reference.json exists (exact values
# measured from our 20k+ downloaded houou logs by human_style_reference.py),
# it OVERRIDES these cited bands with mean±95%CI from the very distribution
# the BC models are trained on. East/South splits live in the same file.
HUMAN_EXPERT = {
    "agari_rate":      (0.21, 0.25),
    "houjuu_rate":     (0.11, 0.13),
    "riichi_rate":     (0.18, 0.22),
    "call_rate":       (0.33, 0.40),
    "win_turn":        (11.1, 11.7),
    "tenpai_turn":     (8.0, 9.5),    # first-tenpai 巡目 (approx; least standardized)
    "tenpai_rate":     (0.55, 0.65),  # hands reaching tenpai at least once
    "draw_tenpai_rate": (0.42, 0.50), # tenpai share at exhaustive draw
    "win_points":      (5800, 6800),  # 平均打点 (here: winner net delta incl. sticks)
}

_MEASURED = "data/tenhou/human_style_reference.json"
if os.path.exists(_MEASURED):
    _m = json.load(open(_MEASURED))["all"]
    def _band(key, ci):
        v = _m[key]
        return (round(v - ci, 4), round(v + ci, 4))
    HUMAN_EXPERT = {
        "agari_rate": _band("agari_rate", _m["agari_ci"]),
        "houjuu_rate": _band("houjuu_rate", _m["houjuu_ci"]),
        "riichi_rate": _band("riichi_rate", _m["riichi_ci"]),
        "call_rate": _band("call_rate", _m["call_ci"]),
        "win_turn": _band("win_turn", 0.05),              # ±1% relative-ish
        "tenpai_turn": HUMAN_EXPERT["tenpai_turn"],       # not derivable from logs yet
        "tenpai_rate": HUMAN_EXPERT["tenpai_rate"],
        "draw_tenpai_rate": _band("draw_tenpai_rate", 0.005),
        "win_points": _band("win_points", 70.0),
    }
    print(f"[ref] measured houou reference loaded ({_m['hands']} hands)")


def _worker(args):
    path, n_games, seed0, temperature = args
    torch.set_num_threads(1)
    net = load_dnn(path, "cpu")
    agg = {"games": 0, "draws": 0, "wins": 0, "tsumo": 0, "deal_ins": 0,
           "riichi": 0, "called": 0, "seats": 0,
           "win_turns": 0.0, "win_n": 0, "dealin_turns": 0.0, "dealin_n": 0,
           "tenpai_turns": 0.0, "tenpai_n": 0, "draw_tenpai": 0, "draw_seats": 0,
           "win_points": 0.0}
    for g in range(n_games):
        game = play_game(net, temperature=temperature, device="cpu",
                         deal_seed=seed0 + g, track_tenpai=True)
        r = game.result or ""
        agg["games"] += 1
        agg["seats"] += 4
        winners = {int(m) for m in WIN_RE.findall(r)}
        if not winners:
            agg["draws"] += 1
        agg["wins"] += len(winners)
        agg["tsumo"] += len(TSUMO_RE.findall(r))
        agg["deal_ins"] += len(set(HOUJUU_RE.findall(r)))
        agg["riichi"] += sum(1 for x in (game.riichi or []) if x)
        agg["called"] += sum(1 for n in (game.n_melds or []) if n > 0)
        # speed (user 2026-08-23): 巡目 of the win / deal-in ≈ table discards / 4
        turn = (game.n_discards or 0) / 4.0
        if winners:
            sp = game.start_points or [25000] * 4
            for w in winners:
                agg["win_points"] += (game.points[w] - sp[w])
            agg["win_turns"] += turn; agg["win_n"] += 1
            if HOUJUU_RE.search(r):
                agg["dealin_turns"] += turn; agg["dealin_n"] += 1
        for tt in (game.tenpai_turns or []):
            if tt is not None:
                agg["tenpai_turns"] += tt; agg["tenpai_n"] += 1
        m = DRAW_TENPAI_RE.search(r)
        if m:
            agg["draw_seats"] += 4
            agg["draw_tenpai"] += len([x for x in m.group(1).split(",") if x.strip()])
    return agg


def profile(path, games, workers, temperature, seed0):
    per = [games // workers] * workers
    for i in range(games - sum(per)):
        per[i] += 1
    jobs = [(path, per[w], seed0 + sum(per[:w]), temperature)
            for w in range(workers) if per[w]]
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        parts = pool.map(_worker, jobs)
    tot = {k: sum(p[k] for p in parts) for k in parts[0]}
    n, s = tot["games"], tot["seats"]
    return {
        "games": n,
        "draw_rate": tot["draws"] / n,
        "agari_rate": tot["wins"] / s,          # per-seat per-hand
        "tsumo_share": tot["tsumo"] / max(tot["wins"], 1),
        "houjuu_rate": tot["deal_ins"] / s,     # per-seat per-hand
        "riichi_rate": tot["riichi"] / s,
        "call_rate": tot["called"] / s,         # ankan counted (proxy)
        "win_turn": tot["win_turns"] / max(tot["win_n"], 1),        # mean 巡目 at a win
        "dealin_turn": tot["dealin_turns"] / max(tot["dealin_n"], 1),
        "tenpai_turn": tot["tenpai_turns"] / max(tot["tenpai_n"], 1),  # mean first-tenpai 巡目
        "tenpai_rate": tot["tenpai_n"] / s,       # seats ever tenpai / seat-hands
        "draw_tenpai_rate": tot["draw_tenpai"] / max(tot["draw_seats"], 1),
        "win_points": tot["win_points"] / max(tot["wins"], 1),  # winner net delta per win
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="label=path pairs")
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed0", type=int, default=90260816)
    ap.add_argument("--out", default=None)
    ap.add_argument("--vs_anchors", default=None,
                    help="comma-separated anchor names (elo_league/anchors.json): seat the "
                         "checkpoint against them (greedy, candidate seat only) instead of "
                         "mirror self-play — the ecology-free reading of houjuu/agari")
    args = ap.parse_args()

    results = {}
    for spec in args.ckpt:
        label, path = spec.split("=", 1)
        t0 = time.time()
        if args.vs_anchors:
            from src.agents.dnn.style_stats import style_vs_anchors
            from scripts.run_elo_league import LEAGUE_DIR
            anchors = json.load(open(f"{LEAGUE_DIR}/anchors.json"))["anchors"]
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            if dev == "cpu":
                torch.set_num_threads(max(1, args.workers))
            cand = load_dnn(path, dev)
            opp = [load_dnn(anchors[n]["path"], dev) for n in args.vs_anchors.split(",")]
            results[label] = style_vs_anchors(cand, opp, args.games, args.seed0,
                                              temperature=0.0, device=dev)
            results[label]["vs_anchors"] = args.vs_anchors
        else:
            results[label] = profile(path, args.games, args.workers,
                                     args.temperature, args.seed0)
        r = results[label]
        print(f"[{label}] {r['games']} games ({time.time()-t0:.0f}s)  "
              f"流局 {r['draw_rate']:.1%}  和牌 {r['agari_rate']:.1%}  "
              f"放铳 {r['houjuu_rate']:.1%}  立直 {r['riichi_rate']:.1%}  和牌巡目 {r['win_turn']:.1f}  放铳巡目 {r['dealin_turn']:.1f}  "
              f"副露 {r['call_rate']:.1%}  自摸占比 {r['tsumo_share']:.1%}"
              + (f"  听牌巡目 {r['tenpai_turn']:.1f}  听牌率 {r['tenpai_rate']:.1%}  "
                 f"流局听牌 {r['draw_tenpai_rate']:.1%}  打点 {r['win_points']:.0f}" if "tenpai_turn" in r else ""),
              flush=True)
        rows = []
        for k, (lo, hi) in HUMAN_EXPERT.items():
            v = r.get(k)
            if v is None:
                continue
            mark = "✓人类带内" if lo <= v <= hi else ("↓低于" if v < lo else "↑高于")
            rows.append(f"    {k:>16}: {v:7.3f}  vs 鳳凰卓 [{lo}, {hi}]  {mark}")
        if rows:
            print("  牌效率 vs 人类高手（方向参考，单局环境 caveat 见 HUMAN_EXPERT 注释）:\n"
                  + "\n".join(rows), flush=True)
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1, ensure_ascii=False)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
