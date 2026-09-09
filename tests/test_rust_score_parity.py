"""P2b: riichi_rs.estimate_hand == mahjong.HandCalculator as the engine drives it (136 allocation,
situational config, reds, dora/ura). Random complete hands incl. melds, chiitoitsu, kokushi."""
import random

import pytest

riichi_rs = pytest.importorskip("riichi_rs")
from mahjong.constants import EAST, SOUTH, WEST, NORTH
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld
from src.tasks.mahjong.table import PyMahjongTable, RED_136, str_to_34, WIND_CONST

ALL = [f"{i}{s}" for s in "mps" for i in range(1, 10)] + [f"{i}z" for i in range(1, 8)]
MELD_TYPE_MAP = PyMahjongTable._MELD_TYPE_MAP
T = PyMahjongTable(randomize_round=False); T.text_obs = False


def _random_complete(rng):
    """(concealed tiles incl. win, win tile, melds[(type, tiles, red)], concealed red counts) or None."""
    kind = rng.random()
    if kind < 0.06:                                  # chiitoitsu
        kinds = rng.sample(ALL, 7)
        tiles = [t for k in kinds for t in (k, k)]
        win = rng.choice(tiles); rest = list(tiles); rest.remove(win)
        creds = _reds(rng, rest)
        return tiles, win, [], creds, False
    if kind < 0.10:                                  # kokushi
        orph = [f"{n}{s}" for s in "mps" for n in "19"] + [f"{n}z" for n in "1234567"]
        tiles = orph + [rng.choice(orph)]
        return tiles, rng.choice(tiles), [], {"m": 0, "p": 0, "s": 0}, False
    sets = []
    for _ in range(4):
        r = rng.random()
        if r < 0.45:
            s = rng.choice("mps"); a = rng.randrange(1, 8); sets.append(("chi", [f"{a}{s}", f"{a+1}{s}", f"{a+2}{s}"]))
        elif r < 0.85:
            sets.append(("pon", [rng.choice(ALL)] * 3))
        else:
            sets.append(("kan4", [rng.choice(ALL)] * 4))
    pair = rng.choice(ALL)
    flat = [t for _, ts in sets for t in ts] + [pair, pair]
    if any(flat.count(x) > 4 for x in set(flat)):
        return None
    n_melds = rng.choice((0, 0, 0, 1, 1, 2, 3, 4))
    idx = list(range(4)); rng.shuffle(idx)
    melds, concealed = [], [pair, pair]
    for j, i in enumerate(idx):
        kind_, ts = sets[i]
        if j < n_melds:
            if kind_ == "kan4":
                mt = rng.choice(("kan", "ankan", "shouminkan"))
                melds.append((mt, list(ts)))
            elif kind_ == "chi":
                melds.append(("chi", sorted(ts, key=lambda t: int(t[0]))))
            else:
                melds.append(("pon", list(ts)))
        else:
            if kind_ == "kan4":
                # a 4-of-a-kind in the concealed hand can't be a set; make it a pon + leftover -> invalid, redo as pon
                concealed += ts[:3]
            else:
                concealed += ts
    if len(concealed) not in (14, 11, 8, 5, 2) or len(concealed) < 2:
        return None
    win = rng.choice(concealed)
    # reds: per suit, at most one red five and only if a 5 of that suit is present (anywhere)
    reds_meld = []
    for mt, ts in melds:
        r = 0
        if ts[0][0] == "5" and ts[0][-1] in "mps" and rng.random() < 0.5:
            r = 1
        reds_meld.append(r)
    used = {"m": 0, "p": 0, "s": 0}
    for (mt, ts), r in zip(melds, reds_meld):
        if r: used[ts[0][-1]] += 1
    # engine-consistent reds: a red five is either the win tile itself (win_red) or sits in the
    # rest of the concealed hand; one red per suit across the whole hand
    rest = list(concealed); rest.remove(win)
    win_red = bool(win[0] == "5" and win[-1] in "mps" and used[win[-1]] == 0 and rng.random() < 0.5)
    creds = {"m": 0, "p": 0, "s": 0}
    for s in "mps":
        if win_red and win[-1] == s:
            creds[s] = 1
        elif used[s] == 0 and f"5{s}" in rest and rng.random() < 0.5:
            creds[s] = 1
    melds_out = [(mt, ts, r) for (mt, ts), r in zip(melds, reds_meld)]
    return concealed, win, melds_out, creds, win_red


def _reds(rng, tiles):
    out = {"m": 0, "p": 0, "s": 0}
    for s in "mps":
        if f"5{s}" in tiles and rng.random() < 0.4:
            out[s] = 1
    return out


def _py_estimate(concealed, win, melds, creds, dora, ura, cfg_kw, win_red):
    """Mirror PyMahjongTable._win_result's 136 construction and config."""
    counter = {}
    rest = list(concealed); rest.remove(win)
    reds = dict(creds)
    if win_red:
        reds[win[-1]] -= 1
        win_id = RED_136[str_to_34(win)]
        tile_ids = T._alloc_136(rest, counter, reds)
    else:
        tile_ids = T._alloc_136(rest, counter, reds)
        win_id = T._alloc_136([win], counter)[0]
    tile_ids.append(win_id)
    meld_objs = []
    for mt, ts, r in melds:
        mtype, opened = MELD_TYPE_MAP[mt]
        m_reds = {"m": 0, "p": 0, "s": 0}
        if r and ts[0][-1] in "mps":
            m_reds[ts[0][-1]] = r
        ids = T._alloc_136(ts, counter, m_reds)
        tile_ids.extend(ids)
        meld_objs.append(Meld(meld_type=mtype, tiles=ids, opened=opened))
    indicators = list(dora)
    if cfg_kw["is_riichi"]:
        indicators += ura[:len(indicators)]
    dora_ids = T._alloc_136(indicators, {})
    config = HandConfig(is_tsumo=cfg_kw["is_tsumo"], is_riichi=cfg_kw["is_riichi"], is_daburu_riichi=cfg_kw["is_daburu_riichi"],
                        is_ippatsu=cfg_kw["is_ippatsu"], is_rinshan=cfg_kw["is_rinshan"], is_chankan=cfg_kw["is_chankan"],
                        is_haitei=cfg_kw["is_haitei"], is_houtei=cfg_kw["is_houtei"], is_tenhou=cfg_kw["is_tenhou"],
                        is_chiihou=cfg_kw["is_chiihou"], is_renhou=False,
                        player_wind=WIND_CONST[cfg_kw["player_wind"]], round_wind=WIND_CONST[cfg_kw["round_wind"]],
                        options=OptionalRules(has_open_tanyao=True, has_aka_dora=True, has_double_yakuman=True))
    res = T._calculator.estimate_hand_value(sorted(tile_ids), win_id, melds=meld_objs or None,
                                            dora_indicators=dora_ids, config=config)
    if res.error:
        return None
    return {"han": res.han, "fu": res.fu, "main": res.cost["main"], "additional": res.cost["additional"],
            "yaku": [str(y) for y in (res.yaku or [])]}


def test_score_parity_random():
    rng = random.Random(90210)
    n_ok = n_win = 0
    for _ in range(40000):
        h = _random_complete(rng)
        if h is None:
            continue
        concealed, win, melds, creds, win_red = h
        opened = any(MELD_TYPE_MAP[m[0]][1] for m in melds)
        is_tsumo = rng.random() < 0.5
        riichi = (not opened) and rng.random() < 0.4
        cfg = dict(is_tsumo=is_tsumo, is_riichi=riichi, is_daburu_riichi=riichi and rng.random() < 0.15,
                   is_ippatsu=riichi and rng.random() < 0.3, is_rinshan=is_tsumo and rng.random() < 0.08,
                   is_chankan=(not is_tsumo) and rng.random() < 0.08, is_haitei=False, is_houtei=False,
                   is_tenhou=False, is_chiihou=False, player_wind=rng.randrange(4), round_wind=rng.randrange(3))
        if is_tsumo and not cfg["is_rinshan"] and rng.random() < 0.05: cfg["is_haitei"] = True
        if (not is_tsumo) and not cfg["is_chankan"] and rng.random() < 0.05: cfg["is_houtei"] = True
        if not melds and is_tsumo and rng.random() < 0.03:
            if cfg["player_wind"] == 0: cfg["is_tenhou"] = True
            else: cfg["is_chiihou"] = True
        dora = rng.sample(ALL, rng.choice((1, 1, 1, 2, 3)))
        ura = rng.sample(ALL, len(dora))
        py = _py_estimate(concealed, win, melds, creds, dora, ura, cfg, win_red)
        all_tiles = list(concealed) + [t for _, ts, _ in melds for t in ts]
        aka = sum(creds.values()) + sum(r for _, _, r in melds)
        rs_cfg = {k: (int(v) if isinstance(v, bool) else v) for k, v in cfg.items()}
        rs_cfg["player_wind"] = 27 + cfg["player_wind"]; rs_cfg["round_wind"] = 27 + cfg["round_wind"]
        # engine quirk (table._win_result): ura indicators are appended to the DORA list, so they
        # score as "Dora N" — the Rust table does the same; the pure estimator gets them as dora here
        indicators = list(dora) + (ura[:len(dora)] if riichi else [])
        rs = riichi_rs.estimate_hand(all_tiles, win, [(m, ts, MELD_TYPE_MAP[m][1]) for m, ts, _ in melds],
                                     indicators, [], aka, rs_cfg)
        assert (py is None) == (rs is None), (concealed, win, melds, creds, cfg, py, rs)
        if py is not None:
            assert py == rs, (concealed, win, melds, creds, dora, ura, cfg, win_red, py, rs)
            n_win += 1
        n_ok += 1
    assert n_win > 5000, (n_ok, n_win)
