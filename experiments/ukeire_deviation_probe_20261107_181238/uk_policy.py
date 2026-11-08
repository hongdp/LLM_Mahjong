"""Forced-ukeire wrapper around a frozen DNN (2026-11-07). Own discard decisions with no opponent in riichi (= exp129
T1's unthreatened set) and own shanten >= MIN_SH: the min-shanten set and its ukeire (unseen copies, discarder's
view) are computed exactly as exp129 T1 (TileEfficiency.evaluate_discards_ranked); if the net's choice is a plain
discard outside the max-ukeire subset, the decision is re-asked with only the max-ukeire discards legal (the net
picks among them). Riichi declarations, wins, calls, kyuushu and threatened turns are untouched.
Every replaced decision is classified (what the net kept that the greedy rule would have cut, what it cut instead)."""
import re
from collections import Counter

from src.tasks.mahjong.shanten import TileEfficiency, dora_from_indicator, pad_for_melds
from src.tasks.mahjong.table import norm_tile

ACT = re.compile(r'type="([^"]+)"(?:\s+tile="([^"]+)")?')
_te = TileEfficiency()


def _visible(table, pid):
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


def _shanten(tiles, nm):
    nm = max(0, min(nm, (14 - len(tiles)) // 3))
    try:
        return _te.calculate_shanten(pad_for_melds(list(tiles), nm))
    except ValueError:
        return 8


def _yakuhai(table, pid):
    return {"5z", "6z", "7z", f"{(pid - table.dealer) % 4 + 1}z", f"{table.round_wind_idx + 1}z"}


def _cls(t, hand, dora, yh, dom):
    out = []
    if t in dora:
        out.append("dora")
    if t[1] == "z":
        out.append("yakuhai" if t in yh else "otakaze")
    elif t[0] in "19":
        out.append("terminal")
    else:
        out.append("middle")
    if hand.count(t) >= 2:
        out.append("pair")
    if t[1] == dom:
        out.append("dom_suit")
    return out


def uk_policy(base, min_sh, stats):
    def pol(table, pid, actions):
        orig = base(table, pid, actions)
        m = ACT.search(orig)
        if not m or m.group(1) not in ("discard", "discard0") or not m.group(2):
            return orig
        if any(table.riichi[o] for o in range(4) if o != pid):
            return orig
        hand, nm = list(table.hands[pid]), len(table.melds[pid])
        sh_now = _shanten(hand, nm)
        if sh_now < min_sh:
            return orig
        parsed = [(a, ACT.search(a)) for a in actions]
        disc = [(a, norm_tile(mm.group(2))) for a, mm in parsed
                if mm and mm.group(1) in ("discard", "discard0") and mm.group(2)]
        legal = {t for _, t in disc}
        ranked = _te.evaluate_discards_ranked(pad_for_melds(hand, nm))
        cand = {t: v for t, v in ranked.items() if t in legal}
        chosen = norm_tile(m.group(2))
        if chosen not in cand or len(cand) < 2:
            return orig
        vis = _visible(table, pid)
        min_post = min(v[0] for v in cand.values())
        opt_sh = [t for t, v in cand.items() if v[0] == min_post]
        uk = {t: sum(max(0, 4 - vis[w]) for w in cand[t][1]) for t in cand}
        mx = max(uk[t] for t in opt_sh)
        opt_uk = [t for t in opt_sh if uk[t] == mx]
        b = "s0" if sh_now <= 0 else ("s1" if sh_now == 1 else ("s2" if sh_now == 2 else "s3"))
        stats[f"n|{b}"] += 1
        if chosen in opt_uk:
            return orig
        keep = [a for a, t in disc if t in opt_uk]
        choice = base(table, pid, keep)
        stats[f"changed|{b}"] += 1
        stats[f"shanten_kept|{b}"] += chosen in opt_sh
        stats[f"uk_loss_sum|{b}"] += mx - uk.get(chosen, 0)
        dora = {dora_from_indicator(i) for i in table.dora_indicators}
        yh = _yakuhai(table, pid)
        suits = Counter(t[1] for t in hand if t[1] in "mps")
        dom = suits.most_common(1)[0][0] if suits else "-"
        forced = norm_tile(ACT.search(choice).group(2))
        for c in _cls(forced, hand, dora, yh, dom):           # what the net KEPT (greedy would have cut it)
            stats[f"kept:{c}|{b}"] += 1
        for c in _cls(chosen, hand, dora, yh, dom):           # what the net CUT instead
            stats[f"cut:{c}|{b}"] += 1
        return choice
    return pol
