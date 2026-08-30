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

YAOCHUU = ({f"{n}{s}" for s in "mps" for n in "19"}
           | {f"{n}z" for n in "1234567"})


def nagashi_players(table) -> List[int]:
    """流し満貫 candidates at an exhaustive draw: every discard is a
    terminal/honor ("0x" red fives are fives, so they break it) and none
    was claimed (river_events[i] = [tile, tsumogiri, riichi, claimed, n])."""
    return [p for p in range(4)
            if table.river_events[p]
            and all(e[0] in YAOCHUU and not e[3]
                    for e in table.river_events[p])]


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


class MatchState:
    """Between-deal match bookkeeping — ONE implementation shared by the
    eval driver (play_hanchan) and the training rollout generator, so
    renchan/honba/kyotaku/nagashi/bust rules can never diverge."""

    def __init__(self, max_deals: int = 24):
        self.points = [25000, 25000, 25000, 25000]
        self.dealer, self.rw, self.honba, self.kyotaku = 0, 0, 0, 0
        self.start_dealer = 0
        self.n = 0
        self.max_deals = max_deals
        self.done = False
        self.busted = False
        self.deals: List[dict] = []

    def begin_deal(self):
        """Returns the HanchanTable context for the next deal."""
        self.n += 1
        return self.dealer, self.rw, self.points, self.kyotaku

    def settle(self, table) -> None:
        """Consume a finished deal's table and advance the match."""
        import re
        r = table.result_summary or ""
        dealer, honba = self.dealer, self.honba
        winners = [int(m) for m in
                   re.findall(r"玩家(\d)\s*(?:自摸|荣和|抢杠)", r)]
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
        self.kyotaku = table.kyotaku if not winners else 0
        dealer_won = any(w == dealer for w in winners)
        is_draw = not winners
        # 途中流局 (four winds / four riichi / four kans): dealer always
        # repeats regardless of tenpai — the engine's abort summary has no
        # 听牌 list, so it must not fall through to the rotate branch
        is_abort = "途中流局" in r
        dealer_tenpai_at_draw = is_abort
        tenpai_seats: List[int] = []
        if is_draw and not is_abort:
            m = re.search(r"流局 \| 听牌: \[([^\]]*)\]", r)
            if m:
                tenpai_seats = [int(x.strip().replace("玩家", ""))
                                for x in m.group(1).split(",") if x.strip()]
                dealer_tenpai_at_draw = dealer in tenpai_seats
            # 流し満貫 (driver-side, Tenhou rules): replaces the engine's
            # tenpai payments with a mangan tsumo; renchan/honba/kyotaku
            # keep the normal draw semantics.
            nagashi = nagashi_players(table)
            if nagashi:
                nt = len(tenpai_seats)
                if 0 < nt < 4:                      # undo engine tenpai split
                    for p in range(4):
                        if p in tenpai_seats:
                            points[p] -= 3000 // nt
                        else:
                            points[p] += 3000 // (4 - nt)
                for w in nagashi:
                    for p in range(4):
                        if p == w:
                            continue
                        pay = 4000 if (w == dealer or p == dealer) else 2000
                        points[p] -= pay
                        points[w] += pay
        self.points = points
        self.deals.append({"deal": self.n, "wind": self.rw, "dealer": dealer,
                           "honba": honba, "result": r,
                           "points_after": list(points)})
        if min(points) < 0:                             # bust ends the match
            self.busted = True
            self.done = True
            return
        if dealer_won or (is_draw and dealer_tenpai_at_draw):
            self.honba += 1                             # renchan
        else:
            self.honba = self.honba + 1 if is_draw else 0
            self.dealer = (self.dealer + 1) % 4
            if self.dealer == self.start_dealer:
                if self.rw >= 1:                        # completed South 4
                    self.done = True
                    return
                self.rw += 1
        if self.rw >= 2 or self.n >= self.max_deals:    # safety (no 西入)
            self.done = True

    def result(self) -> HanchanResult:
        res = HanchanResult()
        res.deals = self.deals
        res.busted = self.busted
        points = list(self.points)
        if self.kyotaku:                                # leftovers to 1st
            top = rank_order(points, self.start_dealer)[0]
            points[top] += self.kyotaku
        res.final_points = points
        order = rank_order(points, self.start_dealer)
        res.placements = [0] * 4
        for rank, seat in enumerate(order):
            res.placements[seat] = rank + 1
        res.uma_points = [points[s] - 25000 + UMA[res.placements[s] - 1]
                          for s in range(4)]
        return res


def play_hanchan(policies: Dict[int, Policy], seed: int,
                 max_deals: int = 24) -> HanchanResult:
    random.seed(seed)
    ms = MatchState(max_deals)
    while not ms.done:
        dealer, rw, points, kyotaku = ms.begin_deal()
        # deterministic per-deal wall, distinct across deals
        random.seed(seed * 1000003 + ms.n)
        table = HanchanTable(dealer, rw, points, kyotaku)
        table.text_obs = False
        play_game_mjai(table, policies, observer=None, sink=lambda ev: None)
        ms.settle(table)
    return ms.result()


class TrainHanchanTable(HanchanTable):
    """HanchanTable for TRAINING rollouts: the per-deal RANK_BONUS is
    zeroed — placement pressure arrives once, as the real uma at match
    end — so the reward is pure point delta inside deals. Subclass attr
    only; the engine file stays byte-identical."""
    RANK_BONUS = [0.0, 0.0, 0.0, 0.0]


def play_hanchan_gen(match_seed: int, shaping: bool = False,
                     max_deals: int = 24, credit=None):
    """Vectorized-rollout hanchan (exp46-D): chains per-deal
    play_game_gen through the shared MatchState via `yield from`, so the
    worker-side protocol is IDENTICAL to a single deal — just longer.

    Trajectories accumulate across deals into one DnnGame per match;
    intermediate deal ends are NOT terminal (credit flows across the
    match, gamma applies over the whole trajectory) and each seat's last
    step gets the real uma (+-15k/+-5k * REWARD_SCALE) as the terminal
    placement signal. `game.deals` carries per-deal facts for style
    aggregation at per-deal semantics."""
    from src.agents.dnn.selfplay import play_game_gen, DnnGame
    from src.tasks.mahjong.table import PyMahjongTable

    ms = MatchState(max_deals)
    match = DnnGame()
    deal_facts = []
    while not ms.done:
        dealer, rw, points, kyotaku = ms.begin_deal()
        random.seed(match_seed * 1000003 + ms.n)
        table = TrainHanchanTable(dealer, rw, points, kyotaku)
        table.text_obs = False
        table.honba = ms.honba          # v3rh scalar; engine has no honba
        if credit is not None:
            w_before = [credit.w(p, points, dealer, rw, ms.honba, kyotaku,
                                 max(1, 9 - ms.n)) for p in range(4)]
        g = yield from play_game_gen(shaping=shaping, table=table)
        ms.settle(table)
        last_deal = ms.done
        for p in range(4):
            steps = g.trajectories[p]
            if not last_deal:
                for st in steps:
                    st.is_terminal = False
            if credit is not None and steps:
                # exp55-D per-deal credit: replace the engine's raw point
                # reward on the deal's last step with the placement-weighted
                # increment W(after)-W(before); the FINAL deal instead pays
                # true_uma - W(before) so credits telescope exactly to uma
                from src.tasks.mahjong.table import PyMahjongTable as _T
                steps[-1].reward -= (table.final_rewards[p]
                                     if table.final_rewards else 0.0)
                if not last_deal:
                    w_after = credit.w(p, ms.points, ms.dealer, ms.rw,
                                       ms.honba, ms.kyotaku,
                                       max(1, 9 - ms.n - 1))
                    steps[-1].reward += ((w_after - w_before[p])
                                         * _T.REWARD_SCALE)
                else:
                    match._pending_final_credit = getattr(
                        match, "_pending_final_credit", {})
                    match._pending_final_credit[p] = w_before[p]
            match.trajectories[p].extend(steps)
        deal_facts.append({
            "result": g.result or "", "riichi": list(g.riichi or []),
            "n_melds": list(g.n_melds or []), "n_discards": g.n_discards,
            "points": list(g.points or []),
            "start_points": list(g.start_points or [])})
    res = ms.result()
    scale = PyMahjongTable.REWARD_SCALE
    pend = getattr(match, "_pending_final_credit", None)
    for p in range(4):
        if match.trajectories[p]:
            last = match.trajectories[p][-1]
            if pend is not None and p in pend:
                last.reward += (res.uma_points[p] - pend[p]) * scale
            else:
                last.reward += res.uma_points[p] * scale
            last.is_terminal = True
    final = deal_facts[-1] if deal_facts else {}
    match.result = final.get("result", "")
    match.points = res.final_points
    match.riichi = final.get("riichi", [False] * 4)
    match.n_melds = final.get("n_melds", [0] * 4)
    match.n_discards = final.get("n_discards", 0)
    match.start_points = [25000] * 4
    match.deals = deal_facts
    match.hanchan = {"placements": res.placements,
                     "uma_points": res.uma_points, "busted": res.busted,
                     "n_deals": len(res.deals)}
    return match


class PlacementCredit:
    """W(state) = rank-uma analytic + MLP residual (exp55-D). Feature
    layout MUST match extract_placement_states.py. Lazy per-process."""

    def __init__(self, w_path: str):
        self.w_path = w_path
        self._net = None

    def _load(self):
        import torch
        from scripts.train_placement_value import PlacementValue
        blob = torch.load(self.w_path, map_location="cpu")
        net = PlacementValue(d_in=blob["d_in"])
        net.load_state_dict(blob["state_dict"])
        net.eval()
        self._net = net

    def w(self, me: int, points, dealer: int, rw: int, honba: int,
          kyotaku: int, deals_left: float) -> float:
        import torch
        if self._net is None:
            self._load()
        rel = [points[(me + k) % 4] for k in range(4)]
        drel = (dealer - me) % 4
        round_idx = rw * 4 + (dealer % 4)   # approximation; extractor used
        # tenhou seed round which equals rw*4 + dealer-rotation count
        x = ([v / 1e5 for v in rel]
             + [round_idx / 8.0, honba / 8.0, kyotaku / 4000.0]
             + [1.0 if drel == k else 0.0 for k in range(4)]
             + [deals_left / 8.0])
        import numpy as np
        xa = np.array([x], dtype=np.float32)
        base = float(_rank_uma_single(rel))
        with torch.no_grad():
            _, u = self._net(torch.from_numpy(xa))
        return base + float(u[0]) * 1000.0


def _rank_uma_single(rel_points) -> float:
    """rank_uma_baseline for one row: UMA by current order (ties favour
    self) + own delta from 25k."""
    self_rank = sum(1 for v in rel_points[1:] if v > rel_points[0])
    return UMA[self_rank] + (rel_points[0] - 25000.0)
