"""Hanchan (半庄) match layer over the single-deal engine (exp53).

Evaluation-only: the engine stays byte-identical (fingerprint untouched).
A hanchan is a driver-side state machine — scores, dealer rotation,
renchan, honba, kyotaku carry, bust, uma — that plays each deal on a
`HanchanTable` (context-injected `PyMahjongTable`) through the verified
`play_game_mjai` loop.

v1 rules (documented divergences are explicit):
  * E1..S4; renchan when the dealer wins or is tenpai at an exhaustive
    draw; honba +1 on dealer win and on every draw, reset on non-dealer
    win; no 西入, no dealer-top-stop, no double-ron split subtleties
    beyond the engine's own multi-ron settle.
  * Honba payments are applied by the DRIVER after settle (ron: winner
    +300*honba from the discarder; tsumo: +100*honba from each loser) —
    the engine has no honba concept (documented known gap). Multi-ron:
    each winner collects from the discarder.
  * Kyotaku: in-deal sticks settle inside the engine; carried sticks are
    injected via start context and go to the deal's (first) winner, or
    carry onward across draws. Leftover sticks at match end go to 1st.
  * Bust (<0 after a deal) ends the match immediately.
  * Placement: final points, ties broken toward the seat closest to the
    starting East; uma (+15,+5,-5,-15)x1000 added for reporting.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

from src.agents.dnn.mjai_export import play_game_mjai
from src.tasks.mahjong.table import PyMahjongTable

Policy = Callable[[PyMahjongTable, int, List[str]], str]

UMA = [15000, 5000, -5000, -15000]


class HanchanTable(PyMahjongTable):
    """PyMahjongTable that starts from an injected match context."""

    def __init__(self, dealer: int, round_wind_idx: int,
                 points: List[int], kyotaku: int):
        self._ctx = (dealer, round_wind_idx, list(points), kyotaku)
        super().__init__()

    def reset(self):
        out = super().reset()
        # re-impose the match context and re-deal so the 14th tile goes to
        # the right seat (super().reset dealt with dealer=0 defaults)
        dealer, rw, pts, kyo = self._ctx
        self.dealer = dealer
        self.round_wind_idx = rw
        from src.tasks.mahjong.table import WIND_CONST
        self.round_wind = WIND_CONST[min(rw, len(WIND_CONST) - 1)]
        self.round_number = dealer + 1
        self.points = list(pts)
        self.kyotaku = kyo
        self.start_points = list(pts)
        self.start_kyotaku = kyo
        if dealer != 0:
            # default reset dealt seat 0 the 14-tile dealer hand; rotate
            # EVERY seat-indexed piece of dealt state together so the
            # intended dealer holds it (hands + red counts + drawn flags —
            # rotating hands alone would corrupt red bookkeeping).
            shift = dealer
            self.hands = {(i + shift) % 4: self.hands[i] for i in range(4)}
            self.red = {(i + shift) % 4: self.red[i] for i in range(4)}
            ld, ldr = list(self.last_drawn), list(self.last_drawn_red)
            self.last_drawn = [None] * 4
            self.last_drawn_red = [False] * 4
            self.last_drawn[dealer] = ld[0]
            self.last_drawn_red[dealer] = ldr[0]
        self.turn = dealer
        return out


def rank_order(points: List[int], start_dealer: int) -> List[int]:
    """Seats in placement order; ties -> closer to starting East wins."""
    def key(seat):
        return (-points[seat], (seat - start_dealer) % 4)
    return sorted(range(4), key=key)


class HanchanResult:
    def __init__(self):
        self.deals: List[dict] = []
        self.final_points: Optional[List[int]] = None
        self.placements: Optional[List[int]] = None   # placement per seat, 1..4
        self.uma_points: Optional[List[int]] = None   # points-25000 + uma
        self.busted = False


def play_hanchan(policies: Dict[int, Policy], seed: int,
                 max_deals: int = 24) -> HanchanResult:
    random.seed(seed)
    res = HanchanResult()
    points = [25000, 25000, 25000, 25000]
    dealer, rw, honba, kyotaku = 0, 0, 0, 0
    start_dealer = 0
    n = 0
    while n < max_deals:
        n += 1
        # deterministic per-deal wall, distinct across deals
        random.seed(seed * 1000003 + n)
        table = HanchanTable(dealer, rw, points, kyotaku)
        table.text_obs = False
        play_game_mjai(table, policies, observer=None, sink=lambda ev: None)

        r = table.result_summary or ""
        import re
        winners = [int(m) for m in re.findall(r"玩家(\d)\s*(?:自摸|荣和|抢杠)", r)]
        dealt_in = re.search(r"放铳:玩家(\d)", r)
        points = list(table.points)
        # driver-side honba payments (engine has no honba)
        if winners and honba:
            for w in set(winners):
                if dealt_in:
                    loser = int(dealt_in.group(1))
                    points[w] += 300 * honba
                    points[loser] -= 300 * honba
                else:                                   # tsumo
                    for p in range(4):
                        if p != w:
                            points[w] += 100 * honba
                            points[p] -= 100 * honba
        kyotaku = table.kyotaku if not winners else 0   # engine pays winner
        dealer_won = any(w == dealer for w in winners)
        is_draw = not winners
        dealer_tenpai_at_draw = False
        if is_draw:
            m = re.search(r"流局 \| 听牌: \[([^\]]*)\]", r)
            if m:
                dealer_tenpai_at_draw = str(dealer) in [
                    x.strip().replace("玩家", "") for x in m.group(1).split(",")]
        res.deals.append({"deal": n, "wind": rw, "dealer": dealer,
                          "honba": honba, "result": r,
                          "points_after": list(points)})
        if min(points) < 0:                             # bust ends the match
            res.busted = True
            break
        if dealer_won or (is_draw and dealer_tenpai_at_draw):
            honba += 1                                  # renchan
        else:
            honba = honba + 1 if is_draw else 0
            dealer = (dealer + 1) % 4
            if dealer == start_dealer:
                if rw >= 1:                             # completed South 4
                    break
                rw += 1
        if rw >= 2:                                     # safety (no 西入 in v1)
            break
    if kyotaku:                                          # leftovers to 1st
        top = rank_order(points, start_dealer)[0]
        points[top] += kyotaku
    res.final_points = points
    order = rank_order(points, start_dealer)
    res.placements = [0] * 4
    for rank, seat in enumerate(order):
        res.placements[seat] = rank + 1
    res.uma_points = [points[s] - 25000 + UMA[res.placements[s] - 1]
                      for s in range(4)]
    return res
