"""Hanchan driver state machine tests (exp53 layer).

The deal loop is exercised with a scripted engine: play_game_mjai is
monkeypatched to stamp a canned result_summary per deal, so renchan /
honba / rotation rules are checked without playing real games.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.tasks.mahjong.hanchan as H


def scripted(results):
    """play_game_mjai stand-in: pops one canned summary per deal."""
    seq = list(results)

    def fake(table, policies, observer=None, sink=None):
        table.result_summary = seq.pop(0)
        # points unchanged: table.points stays as injected by the context
        return None
    return fake


def run(results, max_deals=None):
    orig = H.play_game_mjai
    H.play_game_mjai = scripted(results)
    try:
        return H.play_hanchan({}, seed=1, max_deals=max_deals or len(results))
    finally:
        H.play_game_mjai = orig


PTS = " | 点数: [25000, 25000, 25000, 25000]"


def dealers(res):
    return [d["dealer"] for d in res.deals]


def test_abortive_draw_is_renchan():
    # deal 1 aborts (四风连打): dealer 0 must repeat with honba +1
    res = run(["途中流局(四风连打)" + PTS,
               "流局 | 听牌: []" + PTS,
               "流局 | 听牌: []" + PTS])
    assert dealers(res)[:2] == [0, 0]
    assert res.deals[1]["honba"] == 1


def test_noten_draw_rotates_with_honba():
    res = run(["流局 | 听牌: []" + PTS, "流局 | 听牌: []" + PTS])
    assert dealers(res) == [0, 1]
    assert res.deals[1]["honba"] == 1


def test_tenpai_dealer_draw_is_renchan():
    res = run(["流局 | 听牌: [玩家0, 玩家2]" + PTS, "流局 | 听牌: []" + PTS])
    assert dealers(res) == [0, 0]
    assert res.deals[1]["honba"] == 1


def test_nondealer_win_resets_honba_and_rotates():
    res = run(["途中流局(四杠散了)" + PTS,          # honba -> 1, dealer stays
               "玩家1 荣和 (放铳:玩家2)" + PTS,      # non-dealer win
               "流局 | 听牌: []" + PTS])
    assert dealers(res) == [0, 0, 1]
    assert res.deals[2]["honba"] == 0


def run_with_rivers(results, rivers, max_deals=None):
    orig = H.play_game_mjai
    seq = list(results)
    riv = list(rivers)

    def fake(table, policies, observer=None, sink=None):
        table.result_summary = seq.pop(0)
        rv = riv.pop(0)
        if rv:
            for p, tiles in rv.items():
                table.river_events[p] = [[t, False, False, False, i]
                                         for i, t in enumerate(tiles)]
        return None
    H.play_game_mjai = fake
    try:
        return H.play_hanchan({}, seed=1, max_deals=max_deals or len(results))
    finally:
        H.play_game_mjai = orig


def test_nagashi_mangan_replaces_tenpai_payment():
    # seat 2 (non-dealer) nagashi; engine already paid the [玩家2] tenpai
    # split -- driver must undo it and settle a non-dealer mangan tsumo
    res = run_with_rivers(
        ["流局 | 听牌: [玩家2]" + PTS, "流局 | 听牌: []" + PTS],
        [{2: ["1z", "9m", "1p", "7z"]}, None])
    pts = res.deals[0]["points_after"]
    assert pts == [22000, 24000, 30000, 24000]
    assert sum(pts) == 100000


def test_claimed_discard_breaks_nagashi():
    res = run_with_rivers(
        ["流局 | 听牌: []" + PTS, "流局 | 听牌: []" + PTS],
        [{2: ["1z", "9m"]}, None])
    # mark the discard claimed via a second scripted run
    orig = H.play_game_mjai

    def fake(table, policies, observer=None, sink=None):
        table.result_summary = "流局 | 听牌: []" + PTS
        table.river_events[2] = [["1z", False, False, True, 0]]
        return None
    H.play_game_mjai = fake
    try:
        res2 = H.play_hanchan({}, seed=1, max_deals=1)
    finally:
        H.play_game_mjai = orig
    assert res.deals[0]["points_after"] != [25000] * 4      # nagashi applied
    assert res2.deals[0]["points_after"] == [25000] * 4     # claimed -> no nagashi
