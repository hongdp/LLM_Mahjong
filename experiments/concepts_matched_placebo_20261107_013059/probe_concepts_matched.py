"""R3 (2026-11-07) copy of exp129 probe_concepts.py @843a2a3 with a MATCHED placebo added (original outputs untouched):
extra "m_" records stratified by (own shanten bucket, junme bucket j0<=5 / j1 6-11 / j2>=12); the placebo uses each of
the three opponents' rivers in turn as pseudo-genbutsu (weight 1/3 each) instead of only the opposite seat; the summary
post-stratifies the placebo to the threat states' (shanten, junme) distribution, SE by game-cluster bootstrap.

Human-theory concept probe for DNN checkpoints (exp129, 2026-11-05).

One pass of T=0 four-copy self-play per checkpoint records, per decision,
whether the policy follows a list of human riichi-mahjong heuristics, and —
using the engine's full information — what each heuristic is WORTH in the
model's own ecology (a heuristic that buys nothing there is "not applicable",
not "not learned"). Theory ids follow experiments/exp129_concept_emergence_prereg.

  T1 efficiency   : unthreatened discards — chosen in the min-shanten set /
                    in the max-ukeire subset (unseen copies, discarder's view)
  T2 red five     : discarding a five of a suit where both red and plain are held
  T3 dora         : min-shanten set holds dora and non-dora types — chosen dora
                    share vs uniform share
  T4 honor order  : first 6 own discards, isolated otakaze and isolated yakuhai
                    both held — share of otakaze among those cut
  T5 calls        : pon rate on yakuhai / non-yakuhai (closed / open); chi rate
  T6 riichi       : riichi / dama / break by wait / yaku / junme / opp riichi
                    (same cells as probe_riichi.py)
  T7 push/fold    : genbutsu rate when exposed to riichi, by own shanten;
                    defense_iq = weak - tenpai (exp21 definition)
  T8 suji/kabe    : one riichi opponent — every candidate tile classed as
                    genbutsu / suji / kabe / half / musuji(19|28|37) /
                    honor(visible>=3|2|<=1); a) true deal-in rate per class
                    (tile in the riichi hand's waits and riichi hand not
                    furiten), b) class preference in weak hands with no
                    genbutsu vs the uniform share, c) chosen-tile true danger
                    vs the candidates' mean
  T9 placement    : exposed weak-hand genbutsu rate when 1st vs 4th
  T10 style       : per seat-hand agari / houjuu / riichi / call (incl ankan) /
                    tenpai-ever, win turn, win points, draw-tenpai, first
                    tenpai junme

Genbutsu for T8 = the riichi player's own discards (incl. called ones) plus
every tile discarded after the declaration that it let pass; T7 keeps the
exp21 definition (tile in the visible river) for comparability. Suji/kabe
model ryanmen waits only, as the human theory does. Every rate carries a
game-clustered SE (ratio linearization).

Usage:
    python scripts/probe_concepts.py --ckpt NAME=path [...] --games 2000 \
        --workers 12 --out experiments/probes/exp129_concepts.json
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter

import torch

# PROBE_CODE_ROOT: run against another checkout's src/ + scripts/ (the branch
# the checkpoints were trained on, e.g. for archs this branch lacks).
sys.path.insert(0, os.environ.get("PROBE_CODE_ROOT")
                or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_arena_dnn import load_dnn                                # noqa: E402
from scripts.probe_riichi import WINDS, _has_yaku_without_riichi, _unseen  # noqa: E402
from src.agents.dnn.selfplay import _choose, _resolve_claims              # noqa: E402
from src.tasks.mahjong.table import PyMahjongTable, ACTION_RE, norm_tile  # noqa: E402
from src.tasks.mahjong.shanten import (TileEfficiency, pad_for_melds,     # noqa: E402
                                       dora_from_indicator)

_te = TileEfficiency()
WIN_RE = re.compile(r"玩家(\d)\s*(?:自摸|荣和|抢杠)")
HOUJUU_RE = re.compile(r"放铳:玩家(\d)")
DISCARD_KINDS = ("discard", "riichi")
PAIRS = [("rsafe", "m37"), ("suji", "m37"), ("kabe", "m37"), ("h3", "m37"),
         ("naka", "m456"), ("half", "m37")]


def _shanten(tiles, n_melds):
    n_melds = max(0, min(n_melds, (14 - len(tiles)) // 3))
    try:
        return _te.calculate_shanten(pad_for_melds(list(tiles), n_melds))
    except ValueError:
        return 8


def _visible(table, pid):
    """Copies of each tile type the deciding player can see (own hand included)."""
    c = Counter(table.hands[pid])
    for o in range(4):
        for t in table.discards[o]:
            c[norm_tile(t.replace("*", ""))] += 1
        for m in table.melds[o]:
            for t in m["tiles"]:
                c[norm_tile(t)] += 1
    for t in table.dora_indicators:
        c[norm_tile(t)] += 1
    return c


def _yakuhai(table, pid):
    return {"5z", "6z", "7z", f"{(pid - table.dealer) % 4 + 1}z", f"{table.round_wind_idx + 1}z"}


def tile_class(t, genb, vis):
    if t in genb:
        return "genbutsu"
    n, s = int(t[0]), t[1]
    if s == "z":
        v = vis[t]
        return "honor3" if v >= 3 else ("honor2" if v == 2 else "honor1")

    def full(k):
        return 1 <= k <= 9 and vis[f"{k}{s}"] >= 4
    sides = []
    if n >= 4:      # ryanmen (n-2, n-1) waits n-3 / n
        sides.append((f"{n - 3}{s}" in genb, full(n - 1) or full(n - 2)))
    if n <= 6:      # ryanmen (n+1, n+2) waits n / n+3
        sides.append((f"{n + 3}{s}" in genb, full(n + 1) or full(n + 2)))
    excl = [a or b for a, b in sides]
    if all(excl):
        return "suji" if all(a for a, _ in sides) else "kabe"
    if any(excl):
        return "half"
    return "musuji19" if n in (1, 9) else ("musuji28" if n in (2, 8) else "musuji37")


def _group(name, t, cls):
    n = int(t[0])
    return {"rsafe": cls in ("suji", "kabe"), "suji": cls == "suji", "kabe": cls == "kabe",
            "h3": cls == "honor3", "half": cls == "half", "m37": cls == "musuji37",
            "naka": cls == "suji" and t[1] != "z" and 4 <= n <= 6,
            "m456": cls == "musuji37" and 4 <= n <= 6}[name]


def _bucket(sh):
    return "tenpai" if sh <= 0 else ("mid" if sh == 1 else "weak")


def _record_turn(c, table, pid, parsed, kind, tile, passed):
    """All turn-phase concept records for one decision (before the action is applied)."""
    hand = list(table.hands[pid])
    nm = len(table.melds[pid])
    legal = sorted({norm_tile(m.group(2)) for m in parsed
                    if m and m.group(1) in DISCARD_KINDS and m.group(2)})
    chosen = norm_tile(tile) if (kind in DISCARD_KINDS and tile) else None
    riichi_opps = [o for o in range(4) if o != pid and table.riichi[o]]
    exposed = bool(riichi_opps)
    sh_now = _shanten(hand, nm)
    b = _bucket(sh_now)
    vis = _visible(table, pid)
    jn = table.discard_count[pid]
    jb = "j0" if jn <= 5 else ("j1" if jn <= 11 else "j2")

    # T6 riichi decision (probe_riichi cells)
    types = {m.group(1) for m in parsed if m}
    r_tiles = sorted({norm_tile(m.group(2)) for m in parsed if m and m.group(1) == "riichi"})
    if r_tiles and "tsumo" not in types and not table.melds[pid]:
        best = None
        for t in r_tiles:
            h13 = list(hand)
            h13.remove(t)
            waits = _te.calculate_ukeire(h13)
            nu = sum(_unseen(w, table, h13) for w in waits)
            if best is None or nu > best[1]:
                best = (t, nu, h13, waits)
        t, nu, h13, waits = best
        pw, rw = WINDS[(pid - table.dealer) % 4], WINDS[table.round_wind_idx]
        try:
            yk = [_has_yaku_without_riichi(h13, w, pw, rw) for w in waits]
            yaku = "all" if all(yk) else ("none" if not any(yk) else "some")
        except Exception:
            yaku = "err"
        j = table.discard_count[pid]
        key = "|".join([yaku, "good" if nu >= 5 else "bad",
                        "early" if j <= 5 else ("mid" if j <= 11 else "late"),
                        "opp" if exposed else "noopp"])
        choice = ("riichi" if kind == "riichi" else
                  "dama" if (kind == "discard" and chosen in r_tiles) else
                  "break" if kind == "discard" else "other")
        c[f"ri_n|{key}"] += 1
        c[f"ri_{choice}|{key}"] += 1

    if chosen is None or len(legal) < 2:
        return

    # T2 red five
    if chosen[0] == "5" and chosen[1] in "mps" and table.red[pid][chosen[1]] > 0 \
            and table._plain_copies(pid, chosen) > 0:
        c["red_n"] += 1
        c["red_cut"] += tile[0] == "0"

    if not exposed:
        # T1 efficiency + T3 dora
        ranked = _te.evaluate_discards_ranked(pad_for_melds(hand, nm))
        cand = {t: v for t, v in ranked.items() if t in legal}
        if chosen in cand and len(cand) >= 2:
            min_sh = min(v[0] for v in cand.values())
            opt_sh = [t for t, v in cand.items() if v[0] == min_sh]
            uk = {t: sum(max(0, 4 - vis[w]) for w in cand[t][1]) for t in opt_sh}
            mx = max(uk.values())
            opt_uk = [t for t in opt_sh if uk[t] == mx]
            eb = "s0" if sh_now <= 0 else ("s1" if sh_now == 1 else ("s2" if sh_now == 2 else "s3"))
            for bb in (eb, "all"):
                c[f"eff_n|{bb}"] += 1
                c[f"eff_sh|{bb}"] += chosen in opt_sh
                c[f"eff_uk|{bb}"] += chosen in opt_uk
            dora = {dora_from_indicator(i) for i in table.dora_indicators}
            d_in = [t for t in opt_sh if t in dora]
            if d_in and len(d_in) < len(opt_sh):
                c["dora_n"] += 1
                c["dora_ch"] += chosen in dora
                c["dora_u"] += len(d_in) / len(opt_sh)
        # T4 honor order
        if table.discard_count[pid] < 6:
            yh = _yakuhai(table, pid)
            dora = {dora_from_indicator(i) for i in table.dora_indicators}
            iso = [t for t in legal if t[1] == "z" and hand.count(t) == 1 and t not in dora]
            ota = [t for t in iso if t not in yh]
            yak = [t for t in iso if t in yh]
            if ota and yak and chosen in iso:
                c["hon_n"] += 1
                c["hon_ota"] += chosen in ota
                c["hon_u"] += len(ota) / len(iso)
        # T7/T8 placebo: nobody is in riichi; treat the opposite seat's river as
        # "genbutsu" and run the same fold / class-preference records. A real
        # defensive preference is threat minus placebo (efficiency alone shows
        # up in both, e.g. weak hands cut terminals and dead honors anyway).
        po = (pid + 2) % 4
        c[f"plc_exp_{b}"] += 1
        c[f"plc_fold_{b}"] += chosen in (norm_tile(t.replace("*", "")) for t in table.discards[po])
        pgenb = {norm_tile(t.replace("*", "")) for t in table.furiten_river[po]}
        pcls = {t: tile_class(t, pgenb, vis) for t in legal}
        if not any(v == "genbutsu" for v in pcls.values()):
            _pairs(c, "plc_pr", legal, pcls, chosen, b)
        for po3 in [(pid + k3) % 4 for k3 in (1, 2, 3)]:          # R3: each opponent in turn, weight 1/3
            c[f"m_plc_exp|{b}|{jb}"] += 1 / 3
            c[f"m_plc_fold|{b}|{jb}"] += (chosen in (norm_tile(t.replace("*", "")) for t in table.discards[po3])) / 3
            pg3 = {norm_tile(t.replace("*", "")) for t in table.furiten_river[po3]}
            pc3 = {t: tile_class(t, pg3, vis) for t in legal}
            if not any(v == "genbutsu" for v in pc3.values()):
                _pairs_m(c, "plc", legal, pc3, chosen, b, jb, 1 / 3)
        return

    # T7 push/fold (exp21 genbutsu definition) + T9 placement
    gb7 = all(chosen in (norm_tile(t.replace("*", "")) for t in table.discards[o]) for o in riichi_opps)
    c[f"exp_{b}"] += 1
    c[f"fold_{b}"] += gb7
    c[f"m_thr_exp|{b}|{jb}"] += 1
    c[f"m_thr_fold|{b}|{jb}"] += gb7
    rank = 1 + sum(table.points[x] > table.points[pid] for x in range(4) if x != pid)
    if rank in (1, 4):
        c[f"pexp_{b}|r{rank}"] += 1
        c[f"pfold_{b}|r{rank}"] += gb7

    # T8 suji / kabe — one riichi opponent only
    if len(riichi_opps) != 1:
        return
    o = riichi_opps[0]
    genb = {norm_tile(t.replace("*", "")) for t in table.furiten_river[o]} | passed[o]
    waits = set(table._waits(o))
    furi = table._is_furiten(o)
    cls = {t: tile_class(t, genb, vis) for t in legal}
    dang = {t: (t in waits and not furi) for t in legal}
    for t in legal:
        c[f"eco_n|{cls[t]}"] += 1
        c[f"eco_d|{cls[t]}"] += dang[t]
    c[f"ch_n|{b}"] += 1
    c[f"ch|{b}|{cls[chosen]}"] += 1
    if any(cls[t] == "genbutsu" for t in legal):
        return
    _pairs_m(c, "thr", legal, cls, chosen, b, jb)
    for ctx in ((b, "any") if b != "tenpai" else ("any",)):
        c[f"ng_n|{ctx}"] += 1
        c[f"ng_chd|{ctx}"] += dang[chosen]
        c[f"ng_md|{ctx}"] += sum(dang.values()) / len(legal)
    _pairs(c, "pr", legal, cls, chosen, b)


def _pairs(c, prefix, legal, cls, chosen, b):
    """Class-pair preference records: chosen share of A among A∪B vs uniform share."""
    for ctx in ((b, "any") if b != "tenpai" else ("any",)):
        for A, B in PAIRS:
            a = [t for t in legal if _group(A, t, cls[t])]
            bl = [t for t in legal if _group(B, t, cls[t])]
            if a and bl and (chosen in a or chosen in bl):
                k = f"{A}>{B}|{ctx}"
                c[f"{prefix}_n|{k}"] += 1
                c[f"{prefix}_a|{k}"] += chosen in a
                c[f"{prefix}_u|{k}"] += len(a) / (len(a) + len(bl))


def _pairs_m(c, prefix, legal, cls, chosen, b, jb, w=1.0):
    """R3 stratified pair records: key m_{prefix}_{n,a,u}|A>B|b|jb (shanten bucket b, junme bucket jb)."""
    for A, B in PAIRS:
        a = [t for t in legal if _group(A, t, cls[t])]
        bl = [t for t in legal if _group(B, t, cls[t])]
        if a and bl and (chosen in a or chosen in bl):
            k = f"{A}>{B}|{b}|{jb}"
            c[f"m_{prefix}_n|{k}"] += w
            c[f"m_{prefix}_a|{k}"] += w * (chosen in a)
            c[f"m_{prefix}_u|{k}"] += w * len(a) / (len(a) + len(bl))


def play_one(net, seed, temperature):
    import random
    random.seed(seed)
    table = PyMahjongTable(randomize_round=True)
    table.text_obs = False
    c = Counter()
    passed = {o: set() for o in range(4)}
    first_tenpai = [None] * 4
    riichi_seat = [False] * 4
    guard = 0
    while not table.finished and guard < 600:
        guard += 1
        pid = table.turn
        actions = table.get_legal_actions(pid)
        if not actions:
            break
        parsed = [ACTION_RE.search(a) for a in actions]
        _, action_str = _choose(net, table, pid, actions, temperature, "cpu", cmode="none")
        m = ACTION_RE.search(action_str)
        kind, tile = (m.group(1), m.group(2)) if m else ("other", None)
        if not table.riichi[pid]:
            _record_turn(c, table, pid, parsed, kind, tile, passed)
        if kind == "riichi":
            riichi_seat[pid] = True
        _, _, done, info = table.step(pid, action_str)
        if kind in DISCARD_KINDS and tile:
            t = norm_tile(tile)
            for o in range(4):
                if o != pid and table.riichi[o]:
                    passed[o].add(t)
            if first_tenpai[pid] is None and _shanten(table.hands[pid], len(table.melds[pid])) == 0:
                first_tenpai[pid] = table.discard_count[pid]
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
            _, a_str = _choose(net, table, other, options, temperature, "cpu", cmode="none")
            mm = ACTION_RE.search(a_str)
            a_type = mm.group(1) if mm else None
            # T5 calls
            if not table.riichi[other]:
                opts = [ACTION_RE.search(x) for x in options]
                pons = [x.group(2) for x in opts if x and x.group(1) == "pon"]
                closed = table._is_closed(other)
                if pons:
                    k = ("yh" if norm_tile(pons[0]) in _yakuhai(table, other)
                         else ("ny_closed" if closed else "ny_open"))
                    c[f"pon_n|{k}"] += 1
                    c[f"pon_y|{k}"] += a_type == "pon"
                if any(x and x.group(1) == "chi" for x in opts):
                    k = "closed" if closed else "open"
                    c[f"chi_n|{k}"] += 1
                    c[f"chi_y|{k}"] += a_type == "chi"
            candidates.append({"player_id": other, "parsed": a_str, "type": a_type, "reward": 0.0})
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

    # T10 style (per seat-hand)
    s = table.result_summary or ""
    c["games"] += 1
    c["st_seats"] += 4
    winners = [int(x) for x in WIN_RE.findall(s)]
    c["st_agari"] += len(winners)
    c["st_houjuu"] += len(HOUJUU_RE.findall(s))
    c["st_riichi"] += sum(riichi_seat)
    c["st_call"] += sum(1 for i in range(4) if table.melds[i])
    c["st_tenpai_ever"] += sum(1 for x in first_tenpai if x is not None)
    c["st_tenpai_turn"] += sum(x for x in first_tenpai if x is not None)
    for w in winners:
        c["st_win_n"] += 1
        c["st_win_turn"] += sum(table.discard_count) / 4
        c["st_win_pts"] += table.points[w] - table.start_points[w]
    if s.startswith("流局 |"):
        c["st_draw"] += 1
        c["st_draw_seats"] += 4
        c["st_draw_tenpai"] += sum(1 for i in range(4) if _shanten(table.hands[i], len(table.melds[i])) <= 0)
    elif s.startswith("途中流局"):
        c["st_abort"] += 1
    return c


def play_chunk(args):
    ckpt, seeds, temperature = args
    torch.set_num_threads(1)
    net = load_dnn(ckpt, "cpu")
    return [dict(play_one(net, sd, temperature)) for sd in seeds]


# ---------------------------------------------------------------- summary
def _ratio(games, num, den):
    """Game-clustered ratio sum(num)/sum(den) with linearized SE; num/den are key fns."""
    a = [num(g) for g in games]
    b = [den(g) for g in games]
    B = sum(b)
    if B == 0:
        return None
    R = sum(a) / B
    G = len(games)
    var = G / (G - 1) * sum((x - R * y) ** 2 for x, y in zip(a, b)) / B ** 2
    return {"v": round(R, 4), "se": round(math.sqrt(var), 4), "n": round(B, 1)}


def _diff_ratio(games, n1, d1, n2, d2):
    """(sum n1/sum d1) - (sum n2/sum d2), delta-method game-clustered SE."""
    D1 = sum(d1(g) for g in games)
    D2 = sum(d2(g) for g in games)
    if not D1 or not D2:
        return None
    R1 = sum(n1(g) for g in games) / D1
    R2 = sum(n2(g) for g in games) / D2
    G = len(games)
    res = [(n1(g) - R1 * d1(g)) / D1 - (n2(g) - R2 * d2(g)) / D2 for g in games]
    se = math.sqrt(G / (G - 1) * sum(r * r for r in res))
    return {"v": round(R1 - R2, 4), "se": round(se, 4), "z": round((R1 - R2) / se, 2) if se else None}


def _combo(games, terms):
    """sum_i coef_i * (sum num_i / sum den_i) for terms [(coef, num, den)], game-clustered SE."""
    Ds = [sum(d(g) for g in games) for _, _, d in terms]
    if not all(Ds):
        return None
    Rs = [sum(n(g) for g in games) / D for (_, n, _), D in zip(terms, Ds)]
    G = len(games)
    res = [sum(cf * (n(g) - R * d(g)) / D for (cf, n, d), R, D in zip(terms, Rs, Ds)) for g in games]
    v = sum(cf * R for (cf, _, _), R in zip(terms, Rs))
    se = math.sqrt(G / (G - 1) * sum(r * r for r in res))
    return {"v": round(v, 4), "se": round(se, 4), "z": round(v / se, 2) if se else None}


def summarize(games):
    k = lambda key: (lambda g: g.get(key, 0))
    out = {"games": len(games)}
    # T1
    out["T1_eff"] = {bb: {"shanten_opt": _ratio(games, k(f"eff_sh|{bb}"), k(f"eff_n|{bb}")),
                          "ukeire_opt": _ratio(games, k(f"eff_uk|{bb}"), k(f"eff_n|{bb}"))}
                     for bb in ("all", "s0", "s1", "s2", "s3")}
    # T2
    out["T2_red_cut"] = _ratio(games, k("red_cut"), k("red_n"))
    # T3 / T4: chosen share minus uniform share
    out["T3_dora"] = {"chosen": _ratio(games, k("dora_ch"), k("dora_n")),
                      "uniform": _ratio(games, k("dora_u"), k("dora_n")),
                      "diff": _ratio(games, lambda g: g.get("dora_ch", 0) - g.get("dora_u", 0), k("dora_n"))}
    out["T4_honor_otakaze_first"] = {"chosen": _ratio(games, k("hon_ota"), k("hon_n")),
                                     "uniform": _ratio(games, k("hon_u"), k("hon_n")),
                                     "diff": _ratio(games, lambda g: g.get("hon_ota", 0) - g.get("hon_u", 0), k("hon_n"))}
    # T5
    out["T5_calls"] = {**{f"pon_{x}": _ratio(games, k(f"pon_y|{x}"), k(f"pon_n|{x}")) for x in ("yh", "ny_closed", "ny_open")},
                       **{f"chi_{x}": _ratio(games, k(f"chi_y|{x}"), k(f"chi_n|{x}")) for x in ("closed", "open")},
                       "pon_yh_minus_ny_closed": _diff_ratio(games, k("pon_y|yh"), k("pon_n|yh"),
                                                             k("pon_y|ny_closed"), k("pon_n|ny_closed"))}
    # T6
    def rsum(pred, ch):
        return lambda g: sum(v for kk, v in g.items() if kk.startswith(f"ri_{ch}|") and pred(kk.split("|", 1)[1]))
    cells = {"all": lambda x: True, "good": lambda x: x.split("|")[1] == "good",
             "bad": lambda x: x.split("|")[1] == "bad", "all_yaku": lambda x: x.startswith("all|"),
             "no_yaku": lambda x: x.startswith("none|"), "early": lambda x: x.split("|")[2] == "early",
             "late": lambda x: x.split("|")[2] == "late", "opp": lambda x: x.endswith("|opp"),
             "noopp": lambda x: x.endswith("|noopp")}
    out["T6_riichi_rate"] = {nm: _ratio(games, rsum(p, "riichi"), rsum(p, "n")) for nm, p in cells.items()}
    out["T6_contrasts"] = {
        "good_minus_bad": _diff_ratio(games, rsum(cells["good"], "riichi"), rsum(cells["good"], "n"),
                                      rsum(cells["bad"], "riichi"), rsum(cells["bad"], "n")),
        "noyaku_minus_allyaku": _diff_ratio(games, rsum(cells["no_yaku"], "riichi"), rsum(cells["no_yaku"], "n"),
                                            rsum(cells["all_yaku"], "riichi"), rsum(cells["all_yaku"], "n")),
        "early_minus_late": _diff_ratio(games, rsum(cells["early"], "riichi"), rsum(cells["early"], "n"),
                                        rsum(cells["late"], "riichi"), rsum(cells["late"], "n")),
        "noopp_minus_opp": _diff_ratio(games, rsum(cells["noopp"], "riichi"), rsum(cells["noopp"], "n"),
                                       rsum(cells["opp"], "riichi"), rsum(cells["opp"], "n"))}
    # T7 / T9
    out["T7_fold"] = {b: _ratio(games, k(f"fold_{b}"), k(f"exp_{b}")) for b in ("weak", "mid", "tenpai")}
    out["T7_defense_iq"] = _diff_ratio(games, k("fold_weak"), k("exp_weak"), k("fold_tenpai"), k("exp_tenpai"))
    out["T7_placebo_fold"] = {b: _ratio(games, k(f"plc_fold_{b}"), k(f"plc_exp_{b}")) for b in ("weak", "mid", "tenpai")}
    out["T7_placebo_iq"] = _diff_ratio(games, k("plc_fold_weak"), k("plc_exp_weak"), k("plc_fold_tenpai"), k("plc_exp_tenpai"))
    out["T7_iq_minus_placebo"] = _combo(games, [(1, k("fold_weak"), k("exp_weak")), (-1, k("fold_tenpai"), k("exp_tenpai")),
                                                (-1, k("plc_fold_weak"), k("plc_exp_weak")), (1, k("plc_fold_tenpai"), k("plc_exp_tenpai"))])
    out["T9_placement_fold_weak"] = {
        "r1": _ratio(games, k("pfold_weak|r1"), k("pexp_weak|r1")),
        "r4": _ratio(games, k("pfold_weak|r4"), k("pexp_weak|r4")),
        "r1_minus_r4": _diff_ratio(games, k("pfold_weak|r1"), k("pexp_weak|r1"), k("pfold_weak|r4"), k("pexp_weak|r4"))}
    # T8
    classes = ["genbutsu", "suji", "kabe", "half", "musuji19", "musuji28", "musuji37", "honor3", "honor2", "honor1"]
    out["T8a_dealin_by_class"] = {cl: _ratio(games, k(f"eco_d|{cl}"), k(f"eco_n|{cl}")) for cl in classes}
    out["T8a_vs_musuji37"] = {cl: _diff_ratio(games, k(f"eco_d|{cl}"), k(f"eco_n|{cl}"), k("eco_d|musuji37"), k("eco_n|musuji37"))
                              for cl in classes if cl != "musuji37"}
    out["T8_chosen_class"] = {b: {cl: _ratio(games, k(f"ch|{b}|{cl}"), k(f"ch_n|{b}")) for cl in classes}
                              for b in ("weak", "mid", "tenpai")}
    out["T8b_pref"] = {}
    for ctx in ("weak", "mid", "any"):
        for A, B in PAIRS:
            kk = f"{A}>{B}|{ctx}"
            thr = lambda g, kk=kk: g.get(f"pr_a|{kk}", 0) - g.get(f"pr_u|{kk}", 0)
            plc = lambda g, kk=kk: g.get(f"plc_pr_a|{kk}", 0) - g.get(f"plc_pr_u|{kk}", 0)
            out["T8b_pref"][kk] = {"chosen": _ratio(games, k(f"pr_a|{kk}"), k(f"pr_n|{kk}")),
                                   "uniform": _ratio(games, k(f"pr_u|{kk}"), k(f"pr_n|{kk}")),
                                   "diff": _ratio(games, thr, k(f"pr_n|{kk}")),
                                   "placebo_diff": _ratio(games, plc, k(f"plc_pr_n|{kk}")),
                                   "threat_minus_placebo": _diff_ratio(games, thr, k(f"pr_n|{kk}"),
                                                                       plc, k(f"plc_pr_n|{kk}"))}
    out["T8c_avoid"] = {ctx: {"chosen_danger": _ratio(games, k(f"ng_chd|{ctx}"), k(f"ng_n|{ctx}")),
                              "mean_cand_danger": _ratio(games, k(f"ng_md|{ctx}"), k(f"ng_n|{ctx}"))}
                        for ctx in ("weak", "mid", "any")}
    # T10
    out["T10_style"] = {
        "agari_rate": _ratio(games, k("st_agari"), k("st_seats")),
        "houjuu_rate": _ratio(games, k("st_houjuu"), k("st_seats")),
        "riichi_rate": _ratio(games, k("st_riichi"), k("st_seats")),
        "call_rate": _ratio(games, k("st_call"), k("st_seats")),
        "tenpai_rate": _ratio(games, k("st_tenpai_ever"), k("st_seats")),
        "first_tenpai_junme": _ratio(games, k("st_tenpai_turn"), k("st_tenpai_ever")),
        "win_turn": _ratio(games, k("st_win_turn"), k("st_win_n")),
        "win_points": _ratio(games, k("st_win_pts"), k("st_win_n")),
        "draw_rate": _ratio(games, k("st_draw"), k("games")),
        "draw_tenpai_rate": _ratio(games, k("st_draw_tenpai"), k("st_draw_seats")),
        "abort_rate": _ratio(games, k("st_abort"), k("games"))}
    return out


def summarize_matched(games, B=300, seed=0):
    """Threat minus (shanten, junme)-matched placebo, placebo post-stratified to the threat weights; bootstrap SE."""
    import numpy as np
    G = len(games)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(G, [1.0 / G] * G, size=B).astype(float)          # B x G cluster-bootstrap weights
    vec = lambda key: np.array([g.get(key, 0.0) for g in games], float)
    J = ("j0", "j1", "j2")

    def est(w, items):
        """items: list of (thr_num, thr_den, plc_num, plc_den) vectors per stratum -> threat rate, matched placebo, diff."""
        tn = [w @ a for a, _, _, _ in items]; td = [w @ b for _, b, _, _ in items]
        pn = [w @ c for _, _, c, _ in items]; pd = [w @ d for _, _, _, d in items]
        T = sum(td)
        if T <= 0:
            return None
        thr = sum(tn) / T
        num = den = 0.0
        for t_, pn_, pd_ in zip(td, pn, pd):
            if t_ > 0 and pd_ > 0:
                num += t_ * pn_ / pd_; den += t_
        if den <= 0:
            return None
        plc = num / den
        return thr, plc, thr - plc, den / T

    def pack(items):
        ones = np.ones(G)
        p = est(ones, items)
        if p is None:
            return None
        bs = [est(w, items) for w in W]
        bs = np.array([x for x in bs if x is not None])
        se = bs.std(axis=0, ddof=1)
        return {"threat": round(p[0], 4), "matched_placebo": round(p[1], 4), "threat_minus_matched": round(p[2], 4),
                "se": round(float(se[2]), 4), "z": round(p[2] / se[2], 2) if se[2] > 0 else None,
                "threat_weight_covered": round(p[3], 3), "n_threat": round(float(sum(b_.sum() for _, b_, _, _ in items)), 1),
                "n_placebo": round(float(sum(d_.sum() for _, _, _, d_ in items)), 1)}

    out = {"strata": "shanten bucket x junme (j0<=5, j1 6-11, j2>=12); placebo = mean over the 3 opponents' rivers"}
    t7 = {}
    for b in ("weak", "mid", "tenpai"):
        items = [(vec(f"m_thr_fold|{b}|{j}"), vec(f"m_thr_exp|{b}|{j}"), vec(f"m_plc_fold|{b}|{j}"), vec(f"m_plc_exp|{b}|{j}"))
                 for j in J]
        t7[b] = pack(items)
    out["T7_fold_matched"] = t7
    # iq = fold_weak - fold_tenpai, each matched; bootstrap the combination jointly
    def iq(w):
        r = []
        for b in ("weak", "tenpai"):
            items = [(vec(f"m_thr_fold|{b}|{j}"), vec(f"m_thr_exp|{b}|{j}"), vec(f"m_plc_fold|{b}|{j}"), vec(f"m_plc_exp|{b}|{j}"))
                     for j in J]
            e = est(w, items)
            if e is None:
                return None
            r.append(e)
        return r[0][0] - r[1][0], r[0][1] - r[1][1], (r[0][0] - r[1][0]) - (r[0][1] - r[1][1])
    p = iq(np.ones(G))
    if p:
        bs = np.array([x for x in (iq(w) for w in W) if x is not None])
        se = bs.std(axis=0, ddof=1)
        out["T7_iq_matched"] = {"threat_iq": round(p[0], 4), "matched_placebo_iq": round(p[1], 4),
                                "threat_minus_matched": round(p[2], 4), "se": round(float(se[2]), 4),
                                "z": round(p[2] / se[2], 2) if se[2] > 0 else None}
    t8 = {}
    for ctx, bs_ in (("weak", ("weak",)), ("mid", ("mid",)), ("any", ("weak", "mid", "tenpai"))):
        for A, Bn in PAIRS:
            items = []
            for b in bs_:
                for j in J:
                    k = f"{A}>{Bn}|{b}|{j}"
                    items.append((vec(f"m_thr_a|{k}") - vec(f"m_thr_u|{k}"), vec(f"m_thr_n|{k}"),
                                  vec(f"m_plc_a|{k}") - vec(f"m_plc_u|{k}"), vec(f"m_plc_n|{k}")))
            t8[f"{A}>{Bn}|{ctx}"] = pack(items)
    out["T8b_pref_matched"] = t8
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="NAME=path")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=50000000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    ctx = mp.get_context("fork")
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        if name in results:
            print(f"skip {name} (already in {args.out})", flush=True)
            continue
        t0 = time.time()
        seeds = list(range(args.seed0, args.seed0 + args.games))
        chunks = [seeds[i::args.workers] for i in range(args.workers)]
        with ctx.Pool(args.workers) as pool:
            parts = pool.map(play_chunk, [(path, ch, args.temperature) for ch in chunks])
        games = [g for p in parts for g in p]
        results[name] = {"ckpt": path, "temperature": args.temperature, "seed0": args.seed0, **summarize(games),
                         "R3_matched": summarize_matched(games)}
        json.dump(results, open(args.out, "w"), indent=1, ensure_ascii=False)
        r = results[name]
        print(f"{name} {time.time() - t0:.0f}s iq={r['T7_defense_iq']} eff={r['T1_eff']['all']} "
              f"rsafe={r['T8b_pref']['rsafe>m37|weak']['diff']}", flush=True)


if __name__ == "__main__":
    main()
