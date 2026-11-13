"""Style / capability-ordering metrics (user hypothesis 2026-08-23: attack
metrics — win rate and win speed — track ability early; defense shows up late
and is read through the deal-in rate).

Two producers share one aggregator:
  * training-time: every rollout iteration's self-play games (mirror,
    ecological) -> TensorBoard `style/*`;
  * evaluation: a checkpoint seated against FIXED anchor opponents
    (`style_vs_anchors`) -> ladder TensorBoard + close-chain JSON. This is
    the clean reading: the deal-in rate is an ecological quantity in mirror
    self-play (it depends on how hard the opponents attack).

Per-seat definitions (rates are per seat per hand):
  agari_rate, tsumo_share, houjuu_rate, riichi_rate, call_rate (any open
  meld / ankan proxy), draw_rate (table), win_turn / dealin_turn (mean 巡目
  = table discards / 4 at the moment the hand ended).

Yaku census (user 2026-11-07): every win's yaku list is parsed from the
settlement string (both engines print "玩家p 自摸|荣和(...) | H番F符 | yaku, ...")
into flat "yk:<name>" counters (dora / aka / ura merged across counts), plus
"yakuman" (hands with a yakuman yaku; double yakuman counted once, the
multiple in "yakuman_units") and "kazoe" (>= 13 han without one). Flat keys
so merge() and the trainer's cumulative counter just sum them.
"""
import re
from typing import Dict, Iterable, List, Optional

WIN_RE = re.compile(r"玩家(\d) (?:荣和|自摸|抢杠)")
TSUMO_RE = re.compile(r"玩家(\d) 自摸")
HOUJUU_RE = re.compile(r"放铳:玩家(\d)")
WIN_SEG_RE = re.compile(r"玩家(\d) (?:荣和|自摸|抢杠)[^|]*\| (\d+)番\d+符 \| ([^|;]*)")
YAKU_COUNT_RE = re.compile(r" \d+$")

# names as printed by both engines (python `mahjong` lib str(), Rust score.rs name())
YAKUMAN = frozenset({
    "Tenhou", "Chiihou", "Renhou (yakuman)", "Daisangen", "Shousuushii", "Dai Suushii",
    "Tsuu Iisou", "Ryuuiisou", "Chinroutou", "Suu Ankou", "Suu Ankou Tanki", "Suu Kantsu",
    "Chuuren Poutou", "Daburu Chuuren Poutou", "Kokushi Musou", "Kokushi Musou Juusanmen Matchi"})
DOUBLE_YAKUMAN = frozenset({"Suu Ankou Tanki", "Daburu Chuuren Poutou", "Kokushi Musou Juusanmen Matchi",
                            "Dai Suushii"})

STYLE_KEYS = ("agari_rate", "tsumo_share", "houjuu_rate", "riichi_rate", "call_rate",
              "draw_rate", "win_turn", "dealin_turn", "win_points")


def new_agg() -> Dict[str, float]:
    return {"games": 0, "seats": 0, "draws": 0, "wins": 0, "tsumo": 0, "deal_ins": 0,
            "riichi": 0, "called": 0, "win_turns": 0.0, "win_n": 0,
            "dealin_turns": 0.0, "dealin_n": 0, "win_points_sum": 0.0}


def add_game(agg: Dict[str, float], result: str, riichi: Optional[List[bool]],
             n_melds: Optional[List[int]], n_discards: Optional[int],
             seats: Iterable[int] = (0, 1, 2, 3),
             points: Optional[List[int]] = None,
             start_points: Optional[List[int]] = None) -> None:
    """Count one finished hand for the given seats (all four in mirror
    self-play; only the candidate's seats when seated against anchors)."""
    seats = list(seats)
    r = result or ""
    winners = {int(m) for m in WIN_RE.findall(r)}
    tsumo = {int(m) for m in TSUMO_RE.findall(r)}
    houjuu = {int(m) for m in HOUJUU_RE.findall(r)}
    agg["games"] += 1
    agg["seats"] += len(seats)
    if not winners:
        agg["draws"] += 1
    turn = (n_discards or 0) / 4.0
    for p in seats:
        if p in winners:
            agg["wins"] += 1
            agg["win_turns"] += turn
            agg["win_n"] += 1
            if points:
                sp = start_points or [25000] * 4
                agg["win_points_sum"] += points[p] - sp[p]
            if p in tsumo:
                agg["tsumo"] += 1
        if p in houjuu:
            agg["deal_ins"] += 1
            agg["dealin_turns"] += turn
            agg["dealin_n"] += 1
        if riichi and riichi[p]:
            agg["riichi"] += 1
        if n_melds and n_melds[p] > 0:
            agg["called"] += 1
    for m in WIN_SEG_RE.finditer(r):
        if int(m.group(1)) not in seats:
            continue
        names = [YAKU_COUNT_RE.sub("", y.strip()) for y in m.group(3).split(",") if y.strip()]
        for y in names:
            agg[f"yk:{y}"] = agg.get(f"yk:{y}", 0) + 1
        ym = [y for y in names if y in YAKUMAN]
        if ym:
            agg["yakuman"] = agg.get("yakuman", 0) + 1
            agg["yakuman_units"] = agg.get("yakuman_units", 0) + sum(2 if y in DOUBLE_YAKUMAN else 1 for y in ym)
        elif int(m.group(2)) >= 13:
            agg["kazoe"] = agg.get("kazoe", 0) + 1


def merge(into: Dict[str, float], other: Dict[str, float]) -> None:
    for k, v in other.items():
        into[k] = into.get(k, 0) + v


def summarize(agg: Dict[str, float]) -> Dict[str, float]:
    s = max(agg["seats"], 1)
    return {
        "agari_rate": agg["wins"] / s,
        "tsumo_share": agg["tsumo"] / max(agg["wins"], 1),
        "houjuu_rate": agg["deal_ins"] / s,
        "riichi_rate": agg["riichi"] / s,
        "call_rate": agg["called"] / s,
        "draw_rate": agg["draws"] / max(agg["games"], 1),
        "win_turn": agg["win_turns"] / max(agg["win_n"], 1),
        "dealin_turn": agg["dealin_turns"] / max(agg["dealin_n"], 1),
        "win_points": agg.get("win_points_sum", 0.0) / max(agg["wins"], 1),
        "yakuman_per_deal": agg.get("yakuman", 0) / max(agg["games"], 1),
        "kazoe_per_deal": agg.get("kazoe", 0) / max(agg["games"], 1),
        "games": agg["games"],
    }


def yaku_counts(agg: Dict[str, float]) -> Dict[str, int]:
    """{yaku name: count} plus the yakuman / yakuman_units / kazoe totals."""
    out = {k[3:]: int(v) for k, v in agg.items() if k.startswith("yk:")}
    for k in ("yakuman", "yakuman_units", "kazoe"):
        out[f"_{k}"] = int(agg.get(k, 0))
    return out


def style_vs_anchors(net, anchor_nets: List, games: int, seed0: int,
                     temperature: float = 0.0, device: str = None) -> Dict[str, float]:
    """Seat `net` at one seat per game (rotating) against three anchor nets
    (rotating through the pool) and aggregate ONLY the candidate's seat.
    Deterministic in seed0; temperature 0 = the live/greedy reading.

    Sequential, one game at a time (no batching) — GPU still wins by a wide
    margin on today's models (handset/HRF: ~250ms/decision CPU vs ~8ms GPU,
    2026-08-24 bench), so this defaults to cuda when available. `net` and
    every entry in `anchor_nets` must already live on `device` (the caller's
    job, same as everywhere else in this module — see load_dnn(path, device)).
    torch.set_num_threads(1) is kept regardless of device: the mahjong engine
    itself (shanten/legal-move calc) is always CPU, and this guards against
    thread oversubscription when several of these run concurrently (2026-08-24
    incident: 6 concurrent CPU callers each grabbed the full core count,
    5x-oversubscribing a 24-core box)."""
    from src.agents.dnn.selfplay import play_game
    import torch
    if device is None:
        # infer from where the caller already put the net: passing CPU nets and
        # getting cuda inputs is a silent crash (regression 2026-08-24), so the
        # net is the source of truth, not cuda availability.
        try:
            device = str(next(net.parameters()).device)
        except StopIteration:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    agg = new_agg()
    for g in range(games):
        me = g % 4
        seat_nets = {}
        k = 0
        for p in range(4):
            if p == me:
                continue
            seat_nets[p] = anchor_nets[(g + k) % len(anchor_nets)]
            k += 1
        game = play_game(net, temperature=temperature, device=device,
                         deal_seed=seed0 + g, seat_nets=seat_nets)
        add_game(agg, game.result, game.riichi, game.n_melds, game.n_discards, seats=[me],
                 points=game.points, start_points=game.start_points)
    return summarize(agg)
