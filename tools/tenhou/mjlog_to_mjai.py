"""Tenhou mjlog XML -> MJAI event stream (exp45 human-BC pipeline).

Produces the exact event dialect `mjai_bridge.MjaiDnnBot`/`ShadowTable`
consume (verified consumer: tests/test_mjai_bridge.py). One converter
call yields the FULL-INFORMATION stream (all tehais and draws visible);
`mask_for_seat(events, s)` derives the per-observer stream with the
MahjongCopilot convention (others' tehais/draws are '?') so the replay
exercises only code paths the bridge was verified on.

mjlog reference facts used here (checked against live 凤凰卓 logs):
  * tiles are 0..135 (type*4+copy); copy 0 of 5m/5p/5s (ids 16/52/88) is
    the red five;
  * T/U/V/W<n> draw for seat 0..3, D/E/F/G<n> discard;
  * INIT seed="kyoku,honba,kyotaku,d0,d1,doraIndicator", ten in 100s;
  * N who m: meld bitfield (chi bit2, pon bit3, kakan bit4, else kan:
    ankan when (m&3)==0 else daiminkan);
  * REACH step 1 precedes the riichi discard, step 2 = accepted;
  * DORA hai=n: new indicator at its actual flip position;
  * AGARI may repeat (double ron); RYUUKYOKU type="yao9|..." for aborts.
"""

from __future__ import annotations

import re
from typing import Dict, List

_TAG_RE = re.compile(r"<(\w+)((?:\s+\w+=\"[^\"]*\")*)\s*/?>")
_ATTR_RE = re.compile(r"(\w+)=\"([^\"]*)\"")

_DRAW = {"T": 0, "U": 1, "V": 2, "W": 3}
_DISCARD = {"D": 0, "E": 1, "F": 2, "G": 3}
_BAKAZE = ["E", "S", "W", "N"]


def tile136_to_mjai(t: int) -> str:
    ty, cp = t // 4, t % 4
    if ty >= 27:
        return "ESWNPFC"[ty - 27]
    n, s = ty % 9 + 1, "mps"[ty // 9]
    if n == 5 and cp == 0:
        return f"5{s}r"
    return f"{n}{s}"


def decode_meld(who: int, m: int) -> Dict:
    """Tenhou N-element bitfield -> MJAI meld event (full info)."""
    target = (who + (m & 3)) % 4
    if m & (1 << 2):                                        # chi
        t = (m >> 10) & 0x3F
        r, t = t % 3, t // 3
        base = (t // 7) * 9 + t % 7                          # lowest tile type
        tiles = [4 * (base + i) + ((m >> (3 + 2 * i)) & 3) for i in range(3)]
        called = tiles[r]
        consumed = [x for i, x in enumerate(tiles) if i != r]
        return {"type": "chi", "actor": who, "target": target,
                "pai": tile136_to_mjai(called),
                "consumed": [tile136_to_mjai(c) for c in consumed]}
    if m & (1 << 3):                                        # pon
        t = (m >> 9) & 0x7F
        r, t = t % 3, t // 3
        unused = (m >> 5) & 3
        tiles = [4 * t + c for c in range(4) if c != unused]
        called = tiles[r]
        consumed = [x for i, x in enumerate(tiles) if i != r]
        return {"type": "pon", "actor": who, "target": target,
                "pai": tile136_to_mjai(called),
                "consumed": [tile136_to_mjai(c) for c in consumed]}
    if m & (1 << 4):                                        # kakan
        t = (m >> 9) & 0x7F
        added, t = (m >> 5) & 3, t // 3
        return {"type": "kakan", "actor": who,
                "pai": tile136_to_mjai(4 * t + added),
                "consumed": [tile136_to_mjai(4 * t + c)
                             for c in range(4) if c != added]}
    t136 = (m >> 8) & 0xFF                                  # kan
    tiles = [4 * (t136 // 4) + c for c in range(4)]
    if (m & 3) == 0:
        return {"type": "ankan", "actor": who,
                "consumed": [tile136_to_mjai(x) for x in tiles]}
    return {"type": "daiminkan", "actor": who, "target": target,
            "pai": tile136_to_mjai(t136),
            "consumed": [tile136_to_mjai(x) for x in tiles if x != t136]}


def parse_mjlog(xml: str) -> Dict:
    """-> {"meta": {...}, "events": [...]}; raises ValueError on 3p logs."""
    meta: Dict = {}
    events: List[Dict] = []
    last_drawn = [None] * 4          # tile136 of pending draw, per seat
    kyoku_open = False

    def close_kyoku():
        nonlocal kyoku_open
        if kyoku_open:
            events.append({"type": "end_kyoku"})
            kyoku_open = False

    for m in _TAG_RE.finditer(xml):
        tag, attrs_s = m.group(1), m.group(2)
        at = dict(_ATTR_RE.findall(attrs_s))

        if tag == "GO":
            meta["go_type"] = int(at.get("type", "0"))
            if meta["go_type"] & 0x10:
                raise ValueError("3-player log")
        elif tag == "UN" and "dan" in at:
            meta["dan"] = at["dan"]
            meta["rate"] = at.get("rate", "")
        elif tag == "INIT":
            close_kyoku()
            seed = [int(x) for x in at["seed"].split(",")]
            k = seed[0]
            events.append({
                "type": "start_kyoku",
                "bakaze": _BAKAZE[k // 4], "kyoku": k % 4 + 1,
                "honba": seed[1], "kyotaku": seed[2],
                "oya": int(at["oya"]),
                "dora_marker": tile136_to_mjai(seed[5]),
                "scores": [int(s) * 100 for s in at["ten"].split(",")],
                "tehais": [[tile136_to_mjai(int(t))
                            for t in at[f"hai{i}"].split(",")]
                           for i in range(4)],
            })
            kyoku_open = True
            last_drawn = [None] * 4
        elif tag[0] in _DRAW and tag[1:].isdigit():
            who, tile = _DRAW[tag[0]], int(tag[1:])
            last_drawn[who] = tile
            events.append({"type": "tsumo", "actor": who,
                           "pai": tile136_to_mjai(tile)})
        elif tag[0] in _DISCARD and tag[1:].isdigit():
            who, tile = _DISCARD[tag[0]], int(tag[1:])
            events.append({"type": "dahai", "actor": who,
                           "pai": tile136_to_mjai(tile),
                           "tsumogiri": tile == last_drawn[who]})
            last_drawn[who] = None
        elif tag == "N":
            ev = decode_meld(int(at["who"]), int(at["m"]))
            events.append(ev)
            if ev["type"] in ("chi", "pon", "daiminkan"):
                last_drawn[ev["actor"]] = None
        elif tag == "REACH":
            who = int(at["who"])
            if at.get("step") == "2":
                events.append({"type": "reach_accepted", "actor": who})
            else:
                events.append({"type": "reach", "actor": who})
        elif tag == "DORA":
            events.append({"type": "dora",
                           "dora_marker": tile136_to_mjai(int(at["hai"]))})
        elif tag == "AGARI":
            who, frm = int(at["who"]), int(at["fromWho"])
            events.append({"type": "hora", "actor": who, "target": frm,
                           "pai": tile136_to_mjai(int(at["machi"]))})
            if "owari" in at:
                meta["owari"] = at["owari"]
            # end_kyoku deferred: a double ron adds one more AGARI
        elif tag == "RYUUKYOKU":
            events.append({"type": "ryukyoku",
                           "reason": at.get("type", "exhaustive")})
            if "owari" in at:
                meta["owari"] = at["owari"]
        elif tag == "BYE":
            pass                       # disconnect; auto-tsumogiri follows
    close_kyoku()
    events.append({"type": "end_game"})
    return {"meta": meta, "events": events}


def mask_for_seat(events: List[Dict], me: int) -> List[Dict]:
    """Full-info stream -> what MahjongCopilot would show observer `me`."""
    out = []
    for ev in events:
        t = ev["type"]
        if t == "start_kyoku":
            ev = dict(ev, tehais=[list(h) if i == me else ["?"] * len(h)
                                  for i, h in enumerate(ev["tehais"])])
        elif t == "tsumo" and ev["actor"] != me:
            ev = dict(ev, pai="?")
        out.append(ev)
    return out
