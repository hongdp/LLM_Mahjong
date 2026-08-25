"""MJAI-protocol adapter for the DNN agent (Majsoul integration, 2026-08-23).

MahjongCopilot (https://github.com/latorc/MahjongCopilot) intercepts the
Majsoul websocket (liqi protobuf via mitmproxy), translates it into MJAI
events (https://mjai.app/docs/mjai-protocol) and asks a `Bot.react(event)`
for an MJAI reaction which it then clicks into the browser. This module is
that bot, built on two pieces:

* `ShadowTable` — the real engine class (`PyMahjongTable`) whose state is
  DRIVEN BY MJAI EVENTS instead of its own random wall. Only our seat's
  hand is known; every public fact (rivers, melds, dora, riichi, points,
  wall count, river-event record) is maintained exactly the way the
  engine maintains it during self-play, so `encode_state()` and the
  engine's legal-action generators (`get_legal_actions` /
  `get_interrupt_actions`) work unchanged. This is what guarantees the
  agent sees the same observation distribution it was trained on — and
  it is verified by `tests/test_mjai_bridge.py`, which replays engine
  self-play as an MJAI stream and asserts tensor + legal-set equality at
  every decision.
* `MjaiDnnBot` — feeds events to the shadow table, decides when a
  reaction is due (own draw / other's discard / robbable kakan / after
  own chi-pon), runs the policy and converts the engine action XML back
  into an MJAI reaction (`dahai`, `reach`+`reach_dahai`, `chi`, `pon`,
  `daiminkan`, `ankan`, `kakan`, `hora`, `none`).

Known gaps vs. Majsoul rules (see tools/majsoul_bridge/README.md): the
engine is single-round (no renchan/honba/hanchan placement pressure; honba
is not in the observation), West-round is encoded as South, and 3-player
mode is unsupported. Red fives, abortive draws (kyuushu declared as MJAI
`ryukyoku`), double yakuman, nagashi mangan and triple-ron draw follow
Majsoul since #6; checkpoints trained before #6 see red fives as plain 5s.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from src.agents.dnn.encoder import encode_state, legal_mask
from src.tasks.mahjong.table import ACTION_RE, PyMahjongTable, sort_key

# ----------------------------------------------------------------------
# Tile conversion  (mjai: 1m..9m 1p.. 1s.. E S W N P F C, 5mr/5pr/5sr red)
# ----------------------------------------------------------------------
_MJAI_HONOR_TO_ENGINE = {"E": "1z", "S": "2z", "W": "3z", "N": "4z",
                         "P": "5z", "F": "6z", "C": "7z"}
_ENGINE_HONOR_TO_MJAI = {v: k for k, v in _MJAI_HONOR_TO_ENGINE.items()}


def mjai_to_engine(tile: str) -> str:
    """'5mr' -> '5m' (plain spelling; redness is tracked separately),
    'E' -> '1z', '3p' -> '3p'."""
    if tile in _MJAI_HONOR_TO_ENGINE:
        return _MJAI_HONOR_TO_ENGINE[tile]
    if len(tile) == 3 and tile[2] == "r":
        return tile[:2]
    return tile


def mjai_to_engine_spelled(tile: str) -> str:
    """Like mjai_to_engine but keeps the red five as the engine's '0x'."""
    if len(tile) == 3 and tile[2] == "r":
        return "0" + tile[1]
    return mjai_to_engine(tile)


def engine_to_mjai(tile: str, red: bool = False) -> str:
    tile = tile.replace("*", "")
    if tile in _ENGINE_HONOR_TO_MJAI:
        return _ENGINE_HONOR_TO_MJAI[tile]
    if tile[0] == "0":                                  # engine red spelling
        return "5" + tile[1] + "r"
    return tile + ("r" if red and tile[0] == "5" else "")


def is_red(tile: str) -> bool:
    return len(tile) == 3 and tile.endswith("r")


WIND_IDX = {"E": 0, "S": 1, "W": 2, "N": 3}

# MahjongCopilot's 46-slot mask order (common/mj_helper.MJAI_MASK_LIST);
# the optional `meta` we attach lets its GUI show our action probabilities.
MJAI_MASK_LIST = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
    "5mr", "5pr", "5sr",
    "reach", "chi_low", "chi_mid", "chi_high", "pon", "kan_select",
    "hora", "ryukyoku", "none",
]
_MASK_INDEX = {k: i for i, k in enumerate(MJAI_MASK_LIST)}


# ----------------------------------------------------------------------
# Shadow table
# ----------------------------------------------------------------------
class ShadowTable(PyMahjongTable):
    """Engine state reconstructed from MJAI events for one observer seat.

    Opponents' hands are unknown and left empty; nothing the observer's
    encoder or legal-action code reads depends on them.
    """

    def __init__(self, me: int):
        super().__init__()
        self.text_obs = False
        self.me = me
        self._rinshan_pending = [False] * 4
        # MJAI spelling of our hand tiles (keeps red-five identity so the
        # reaction names the physical tile Majsoul expects).
        self.my_tiles_mjai: List[str] = []
        self.last_tile_mjai: Optional[str] = None      # last discard / kakan tile
        self.my_drawn_mjai: Optional[str] = None

    # ---- helpers -----------------------------------------------------
    def _take_mjai(self, tile_mjai: str) -> int:
        """Remove one physical tile from our hand (both spellings);
        returns 1 if the copy taken was a red five."""
        eng = mjai_to_engine(tile_mjai)
        self.hands[self.me].remove(eng)
        if tile_mjai in self.my_tiles_mjai:
            taken = tile_mjai
        else:                                       # spelling mismatch: take any copy
            taken = next(t for t in self.my_tiles_mjai if mjai_to_engine(t) == eng)
        self.my_tiles_mjai.remove(taken)
        if is_red(taken):
            self.red[self.me][taken[1]] -= 1
            return 1
        return 0

    def physical(self, engine_tile: str) -> str:
        """Physical tile of ours for an engine action tile: '0x' names the
        red five, '5x' the plain copy (falling back to whichever we hold)."""
        if engine_tile[0] == "0":
            red = "5" + engine_tile[1] + "r"
            return red if red in self.my_tiles_mjai else "5" + engine_tile[1]
        plain = engine_to_mjai(engine_tile)
        if plain in self.my_tiles_mjai:
            return plain
        red = plain + "r"
        if red in self.my_tiles_mjai:
            return red
        return plain

    def _open_interrupt_window(self, tile: str, discarder: int) -> None:
        # RCR 3.13.2 bookkeeping (engine.do_discard): who could have ronned
        self._ron_chance = {self.me} if (self.me != discarder
                                         and tile in self._waits(self.me)) else set()

    def _close_interrupt_window(self) -> None:
        self._confirm_riichi()
        self._apply_missed_ron()

    # ---- events ------------------------------------------------------
    def start_kyoku(self, msg: dict) -> None:
        super().reset()
        self.text_obs = False
        self.dealer = int(msg["oya"])
        self.round_wind_idx = min(WIND_IDX.get(msg.get("bakaze", "E"), 0), 2)   # West kept (no more folding)
        from src.tasks.mahjong.table import WIND_CONST
        self.round_wind = WIND_CONST[self.round_wind_idx]
        self.round_number = self.dealer + 1
        self.turn = self.dealer
        self.points = [int(s) for s in msg["scores"]]
        self.kyotaku = int(msg.get("kyotaku", 0)) * 1000
        self.dora_indicators = [mjai_to_engine(msg["dora_marker"])]
        self.ura_indicators = []
        # 136 - 14 dead - 52 dealt = 70 live draws; each `tsumo` pops one
        self.wall = ["?"] * 70
        self.dead_wall = ["?"] * 14
        self._rinshan_idx = 0
        self.hands = {i: [] for i in range(4)}
        tehai = msg["tehais"][self.me]
        self.my_tiles_mjai = list(tehai)
        self.hands[self.me] = sorted((mjai_to_engine(t) for t in tehai), key=sort_key)
        self.red = {i: {"m": 0, "p": 0, "s": 0} for i in range(4)}
        for t in tehai:
            if is_red(t):
                self.red[self.me][t[1]] += 1
        self.last_drawn_red = [False] * 4
        self.last_discard_red = False
        self.last_drawn = [None] * 4
        self.last_discard = None
        self.last_discarder = None
        self._rinshan_pending = [False] * 4
        self.last_tile_mjai = None
        self.my_drawn_mjai = None
        self._waits_cache.clear()
        self.kyoku_meta = {"bakaze": msg.get("bakaze"), "kyoku": msg.get("kyoku"),
                           "honba": msg.get("honba"), "kyotaku": msg.get("kyotaku")}

    def tsumo(self, actor: int, pai: str) -> None:
        self._close_interrupt_window()
        if self.wall:
            self.wall.pop()
        self.turn = actor
        if actor == self.me:
            eng = mjai_to_engine(pai)
            self.hands[self.me].append(eng)
            self.hands[self.me].sort(key=sort_key)
            self.my_tiles_mjai.append(pai)
            self.my_drawn_mjai = pai
            self.last_drawn[self.me] = eng
            self.last_drawn_red[self.me] = is_red(pai)
            if is_red(pai):
                self.red[self.me][pai[1]] += 1
        else:
            self.last_drawn[actor] = "?"
        self.rinshan[actor] = self._rinshan_pending[actor]
        self._rinshan_pending[actor] = False
        self.temp_furiten[actor] = False

    def reach(self, actor: int) -> None:
        # declaration: the following dahai is the riichi tile
        self.riichi[actor] = True
        self.riichi_pending = actor
        self.daburu[actor] = (self.discard_count[actor] == 0 and not self.any_call)

    def reach_accepted(self, actor: int) -> None:
        if self.riichi_pending == actor:
            self._confirm_riichi()

    def dahai(self, actor: int, pai: str, tsumogiri: bool) -> None:
        tile = mjai_to_engine(pai)
        spelled = mjai_to_engine_spelled(pai)           # '0x' for a red five
        riichi_mark = self.riichi_pending == actor
        self.river_events[actor].append(
            [spelled, bool(tsumogiri), riichi_mark, False, self.discard_count[actor]])
        if riichi_mark:
            self.riichi_turn[actor] = self.discard_count[actor]
        if actor == self.me:
            self._take_mjai(pai)
            self.my_drawn_mjai = None
        self.discards[actor].append(spelled + ("*" if riichi_mark else ""))
        self.furiten_river[actor].append(tile)
        self.last_discard = tile
        self.last_discard_red = is_red(pai)
        self.last_discarder = actor
        self.last_tile_mjai = pai
        self.last_drawn[actor] = None
        self.rinshan[actor] = False
        self.ippatsu[actor] = False
        self.discard_count[actor] += 1
        self.kuikae = None
        if riichi_mark:
            self.ippatsu[actor] = True
        self._open_interrupt_window(tile, actor)

    def _claim_common(self, actor: int, target: int) -> None:
        self._close_interrupt_window()
        if self.discards[target]:
            self.discards[target].pop()
        if self.river_events[target]:
            self.river_events[target][-1][3] = True
        self.turn = actor
        self.last_discard = None
        self.last_drawn[actor] = None
        self.rinshan[actor] = False
        self.any_call = True
        self.ippatsu = [False] * 4
        self.temp_furiten[actor] = False

    def _consume(self, actor: int, consumed: List[str]) -> int:
        """Take the consumed tiles from our hand (if ours); return red count."""
        if actor == self.me:
            return sum(self._take_mjai(c) for c in consumed)
        return sum(1 for c in consumed if is_red(c))

    def chi(self, actor: int, target: int, pai: str, consumed: List[str]) -> None:
        tile = mjai_to_engine(pai)
        used = [mjai_to_engine(c) for c in consumed]
        reds = self._consume(actor, consumed) + int(is_red(pai))
        if actor == self.me:
            self.kuikae = (actor, self._kuikae_tiles(tile, used))
        self.melds[actor].append({"type": "chi", "tiles": sorted(used + [tile], key=sort_key),
                                  "opened": True, "from": target, "red": reds})
        self._claim_common(actor, target)

    def pon(self, actor: int, target: int, pai: str, consumed: List[str]) -> None:
        tile = mjai_to_engine(pai)
        reds = self._consume(actor, consumed) + int(is_red(pai))
        if actor == self.me:
            self.kuikae = (actor, self._kuikae_tiles(tile, []))
        self.melds[actor].append({"type": "pon", "tiles": [tile] * 3,
                                  "opened": True, "from": target, "red": reds})
        self._record_pao(actor, tile, target)
        self._claim_common(actor, target)

    def daiminkan(self, actor: int, target: int, pai: str, consumed: List[str]) -> None:
        tile = mjai_to_engine(pai)
        reds = self._consume(actor, consumed) + int(is_red(pai))
        self.melds[actor].append({"type": "kan", "tiles": [tile] * 4,
                                  "opened": True, "from": target, "red": reds})
        self._record_pao(actor, tile, target)
        self._claim_common(actor, target)
        self._after_kan_event(actor)

    def ankan(self, actor: int, consumed: List[str]) -> None:
        self._close_interrupt_window()
        tile = mjai_to_engine(consumed[0])
        reds = self._consume(actor, consumed)
        if actor == self.me:
            self.my_drawn_mjai = None
        self.melds[actor].append({"type": "ankan", "tiles": [tile] * 4, "opened": False,
                                  "red": reds})
        self.last_drawn[actor] = None
        self._after_kan_event(actor)

    def kakan(self, actor: int, pai: str) -> None:
        self._close_interrupt_window()
        tile = mjai_to_engine(pai)
        reds = int(is_red(pai))
        if actor == self.me:
            reds = self._take_mjai(pai)
            self.my_drawn_mjai = None
        pon = next((m for m in self.melds[actor]
                    if m["type"] == "pon" and m["tiles"][0] == tile), None)
        if pon is not None:
            pon["type"] = "shouminkan"
            pon["tiles"] = [tile] * 4
            pon["red"] = pon.get("red", 0) + reds
        self.last_drawn[actor] = None
        self.last_tile_mjai = pai
        self._after_kan_event(actor)

    def _after_kan_event(self, actor: int) -> None:
        # engine._after_kan minus the implicit rinshan draw: MJAI sends the
        # replacement draw as its own `tsumo` event (wall count: the live
        # wall's tail tile moves to the dead wall, so that tsumo still pops).
        self.kan_count += 1
        self.any_call = True
        self.ippatsu = [False] * 4
        self._rinshan_pending[actor] = True
        self.turn = actor

    def dora(self, marker: str) -> None:
        self.dora_indicators.append(mjai_to_engine(marker))

    def snapshot(self) -> dict:
        """Full observer-side game state (JSON-serialisable) for the record."""
        from src.tasks.mahjong.shanten import dora_from_indicator
        me = self.me
        return {
            "seat": me, "dealer": self.dealer, "round_wind": self.round_wind_idx,
            "kyoku": getattr(self, "kyoku_meta", {}),
            "turn": self.turn, "wall": len(self.wall),
            "hand": list(self.my_tiles_mjai), "drawn": self.my_drawn_mjai,
            "hand_engine": list(self.hands[me]),
            "melds": {p: [{"type": m["type"], "tiles": list(m["tiles"]), "from": m.get("from")}
                          for m in self.melds[p]] for p in range(4)},
            "rivers": {p: list(self.discards[p]) for p in range(4)},
            "river_events": {p: [list(e) for e in self.river_events[p]] for p in range(4)},
            "dora_indicators": list(self.dora_indicators),
            "dora": [dora_from_indicator(i) for i in self.dora_indicators],
            "riichi": [bool(x) for x in self.riichi], "riichi_turn": list(self.riichi_turn),
            "points": list(self.points), "kyotaku": self.kyotaku,
            "last_discard": self.last_discard, "last_discarder": self.last_discarder,
            "last_tile_mjai": self.last_tile_mjai,
            "furiten": bool(self._is_furiten(me)) if self.hands[me] else False,
            "kan_count": self.kan_count, "discard_count": list(self.discard_count),
        }

    # ---- chankan legality --------------------------------------------
    def chankan_actions(self, actor: int, pai: str, ankan: bool = False) -> List[str]:
        """Legal interrupt actions against `actor`'s added kan (or, for a
        国士无双 hand, concealed kan) of `pai`, evaluated BEFORE the kan
        mutates the table."""
        self.pending_kan = {"player": actor, "tile": mjai_to_engine(pai), "ankan": ankan}
        try:
            return self.get_interrupt_actions(self.me)
        finally:
            self.pending_kan = None


# ----------------------------------------------------------------------
# Policy wrapper
# ----------------------------------------------------------------------
class DnnPolicy:
    """Thin callable around a loaded `MahjongPolicyNet`-family module."""

    def __init__(self, net, device: str = "cpu", temperature: float = 0.0):
        self.net = net.to(device).eval()
        self.device = device
        self.temperature = temperature
        self.variant = getattr(net, "encoder_variant", "v1")

    @torch.no_grad()
    def __call__(self, table, pid: int, actions: List[str]
                 ) -> Tuple[str, Dict[str, float], float]:
        """Returns (chosen action xml, {action: prob}, value)."""
        planes, scalars = encode_state(table, pid, variant=self.variant)
        mask, lookup = legal_mask(actions)
        logits, v = self.net.forward_with_value(
            planes[None].to(self.device), scalars[None].to(self.device),
            mask[None].to(self.device))
        logits = logits[0].float().cpu()
        probs = torch.softmax(logits, 0)
        if self.temperature <= 0:
            idx = int(torch.argmax(logits))
        else:
            idx = int(torch.multinomial(torch.softmax(logits / self.temperature, 0), 1))
        dist = {lookup[i]: float(probs[i]) for i in lookup}
        return lookup[idx], dist, float(v[0])


def load_policy(ckpt_path: str, device: str = "cpu", temperature: float = 0.0) -> DnnPolicy:
    from scripts.run_arena_dnn import load_dnn
    return DnnPolicy(load_dnn(ckpt_path, device), device, temperature)


# ----------------------------------------------------------------------
# MJAI bot
# ----------------------------------------------------------------------
class MjaiDnnBot:
    """MJAI bot: `react(event) -> reaction | None`.

    `policy(table, pid, actions) -> (action_xml, {xml: prob}, value)`; any
    callable with that signature works (tests use a scripted one).
    """

    def __init__(self, policy, seat: Optional[int] = None, name: str = "LLM_Mahjong-DNN"):
        self.policy = policy
        self.name = name
        self.seat = seat
        self.table: Optional[ShadowTable] = None
        self.in_kyoku = False
        self.last_decision: Optional[dict] = None   # debugging / GUI
        self.phase: Optional[str] = None
        # ---- structured record (both modes) ----
        # game_record = {"seat", "kyokus": [{"start": start_kyoku msg, "decisions": [...],
        #                "result": end_kyoku liqi data}], "end_game": ...}
        # each decision: state snapshot BEFORE acting, legal actions, probs, V,
        # the policy's pick, the MJAI reaction we returned, and `executed` =
        # what actually happened at the table (differs from the pick when the
        # human overrides in assist mode, or Majsoul rejected the action).
        self.game_record: Optional[dict] = None
        self._pending: Optional[dict] = None        # decision awaiting execution
        self.on_game_end = None                     # callback(game_record)
        self.n_decisions = 0
        if seat is not None:
            self.start_game(seat)

    # ---- lifecycle ---------------------------------------------------
    def start_game(self, seat: int) -> None:
        self._finish_game()
        self.seat = int(seat)
        self.table = ShadowTable(self.seat)
        self.in_kyoku = False
        self.game_record = {"seat": self.seat, "kyokus": [], "end_game": None}

    def _finish_game(self) -> None:
        rec = self.game_record
        if rec and rec["kyokus"] and self.on_game_end:
            self.on_game_end(rec)
        self.game_record = None

    @property
    def current_kyoku(self) -> Optional[dict]:
        if self.game_record and self.game_record["kyokus"]:
            return self.game_record["kyokus"][-1]
        return None

    # execution reconciliation -------------------------------------------
    _EXEC_TYPES = {"dahai", "reach", "chi", "pon", "daiminkan", "ankan", "kakan"}

    def _settle_pending(self, executed: dict) -> None:
        p = self._pending
        if p is None:
            return
        p["executed"] = executed
        xml = _event_to_action(executed, self.seat)
        p["executed_action"] = xml
        verdict = compare_execution(p["chosen"], xml)
        p["match"] = verdict                    # match | same_kind | override | unknown
        # same_kind = the same tile TYPE, a different physical copy (plain vs
        # red five): not a human decision change, so it is not an override.
        p["override"] = verdict == "override"
        self._pending = None

    def _reconcile(self, msg: dict) -> None:
        """Called for every event BEFORE it is applied: decide what the
        previous decision turned into."""
        p = self._pending
        if p is None:
            return
        t, actor = msg.get("type"), msg.get("actor")
        me = self.seat
        if actor == me and t in self._EXEC_TYPES:
            if t == "reach":
                return                   # the riichi tile follows as dahai
            if t == "dahai" and self.table.riichi_pending == me:
                msg = dict(msg, reach=True)      # this dahai is the riichi tile
            self._settle_pending(_event_summary(msg))
        elif t == "end_kyoku":
            ld = msg.get("liqi_data") or {}
            hules = ld.get("hules") or []
            if any(h.get("seat") == me for h in hules):
                self._settle_pending({"type": "hora"})
            elif p["phase"] == "turn":
                # MahjongCopilot drops the events queued right before the
                # round-end message, so our last discard of a kyoku never
                # reaches us: outcome unknown, NOT a human override.
                p["executed"] = {"type": "end_kyoku"}
                p["executed_action"] = None
                p["override"] = False
                self._pending = None
            else:
                self._settle_pending({"type": "none"})
        elif p["phase"] in ("claim", "chankan") and (actor != me or t == "tsumo"):
            # someone else's event (or our own draw) closed the window: we passed
            self._settle_pending({"type": "none"})

    def react_batch(self, msgs: List[dict]) -> Optional[dict]:
        out = None
        for i, m in enumerate(msgs):
            m = dict(m)
            if i < len(msgs) - 1:
                m["can_act"] = False
            out = self.react(m)
        return out

    def react(self, msg: dict) -> Optional[dict]:
        t = msg.get("type")
        can_act = bool(msg.get("can_act", True))
        me = self.seat

        if t == "start_game":
            self.start_game(msg.get("id", me if me is not None else 0))
            return None
        if self.table is None:
            raise RuntimeError("react() before start_game")
        tb = self.table
        self._reconcile(msg)

        if t == "start_kyoku":
            tb.start_kyoku(msg)
            self.in_kyoku = True
            if self.game_record is not None:
                self.game_record["kyokus"].append({"start": msg, "decisions": [], "result": None})
            return None
        if t == "end_kyoku":
            self.in_kyoku = False
            if self.current_kyoku is not None:
                self.current_kyoku["result"] = {"liqi_name": msg.get("liqi_name"),
                                                "liqi_data": msg.get("liqi_data")}
            return None
        if t == "end_game":
            self.in_kyoku = False
            if self.game_record is not None:
                self.game_record["end_game"] = msg.get("liqi_data")
            self._finish_game()
            return None
        if t in ("hora", "ryukyoku"):
            self.in_kyoku = False
            return None
        if not self.in_kyoku:
            return None

        actor = msg.get("actor")
        if t == "dora":
            tb.dora(msg["dora_marker"])
            return None
        if t == "tsumo":
            tb.tsumo(actor, msg["pai"])
            if actor == me and can_act:
                return self._decide_turn()
            return None
        if t == "reach":
            tb.reach(actor)
            return None
        if t == "reach_accepted":
            tb.reach_accepted(actor)
            return None
        if t == "dahai":
            tb.dahai(actor, msg["pai"], msg.get("tsumogiri", False))
            if actor != me and can_act:
                return self._decide_interrupt()
            return None
        if t == "chi":
            tb.chi(actor, msg["target"], msg["pai"], msg["consumed"])
            return self._decide_turn() if (actor == me and can_act) else None
        if t == "pon":
            tb.pon(actor, msg["target"], msg["pai"], msg["consumed"])
            return self._decide_turn() if (actor == me and can_act) else None
        if t == "daiminkan":
            tb.daiminkan(actor, msg["target"], msg["pai"], msg["consumed"])
            return None                                  # rinshan tsumo follows
        if t == "ankan":
            reaction = None
            if actor != me and can_act:                  # Majsoul: 国士 may rob an ankan
                reaction = self._decide_chankan(actor, msg["consumed"][0], ankan=True)
            tb.ankan(actor, msg["consumed"])
            return reaction
        if t == "kakan":
            reaction = None
            if actor != me and can_act:
                reaction = self._decide_chankan(actor, msg["pai"])
            tb.kakan(actor, msg["pai"])
            return reaction
        if t == "nukidora":
            raise RuntimeError("3-player mode is not supported")
        return None

    # ---- decisions ---------------------------------------------------
    def _run_policy(self, actions: List[str], phase: str) -> Tuple[str, dict]:
        self.phase = phase                      # visible to the policy / tests
        state = self.table.snapshot()
        chosen, dist, value = self.policy(self.table, self.seat, actions)
        self.n_decisions += 1
        self.last_decision = {"n": self.n_decisions, "phase": phase, "actions": actions,
                              "chosen": chosen, "probs": dist, "value": value,
                              "state": state, "reaction": None,
                              "executed": None, "executed_action": None, "override": None}
        if self._pending is not None:           # previous one never resolved
            self._pending["executed"] = {"type": "unknown"}
        self._pending = self.last_decision
        if self.current_kyoku is not None:
            self.current_kyoku["decisions"].append(self.last_decision)
        return chosen, dist

    def _finish_reaction(self, r: Optional[dict]) -> Optional[dict]:
        if self.last_decision is not None:
            self.last_decision["reaction"] = {k: v for k, v in (r or {}).items() if k != "meta"}
        return r

    def _decide_turn(self) -> Optional[dict]:
        tb = self.table
        actions = tb.get_legal_actions(self.seat)
        if not actions:
            return None
        chosen, dist = self._run_policy(actions, "turn")
        m = ACTION_RE.search(chosen)
        a_type, tile = m.group(1), m.group(2)
        me = self.seat
        if a_type == "tsumo":
            r = {"type": "hora", "actor": me, "target": me, "pai": tb.my_drawn_mjai}
        elif a_type == "kyuushu":                       # 九种九牌 declaration
            r = {"type": "ryukyoku", "actor": me}
        elif a_type == "kan":
            is_ankan = tb.hands[me].count(tile) == 4
            if is_ankan:
                consumed = self._physical_copies(tile, 4)
                r = {"type": "ankan", "actor": me, "consumed": consumed}
            else:
                pai = tb.physical(tile)
                r = {"type": "kakan", "actor": me, "pai": pai,
                     "consumed": [engine_to_mjai(tile)] * 3}
                # the pon may hold the red copy; Majsoul only needs the kakan tile
        elif a_type == "riichi":
            pai = tb.physical(tile)
            r = {"type": "reach", "actor": me,
                 "reach_dahai": {"type": "dahai", "actor": me, "pai": pai,
                                 "tsumogiri": pai == tb.my_drawn_mjai}}
        else:                                           # discard
            pai = tb.physical(tile)
            r = {"type": "dahai", "actor": me, "pai": pai,
                 "tsumogiri": pai == tb.my_drawn_mjai}
        r["meta"] = self._meta(dist)
        return self._finish_reaction(r)

    def _decide_interrupt(self) -> Optional[dict]:
        tb = self.table
        actions = tb.get_interrupt_actions(self.seat)
        if len(actions) <= 1:
            return None                                 # nothing but skip
        chosen, dist = self._run_policy(actions, "claim")
        m = ACTION_RE.search(chosen)
        a_type, tile, with_ = m.group(1), m.group(2), m.group(3)
        me, target, pai = self.seat, tb.last_discarder, tb.last_tile_mjai
        if a_type == "skip":
            r = {"type": "none"}
        elif a_type == "ron":
            r = {"type": "hora", "actor": me, "target": target, "pai": pai}
        elif a_type == "pon":
            r = {"type": "pon", "actor": me, "target": target, "pai": pai,
                 "consumed": self._physical_copies(tile, 2)}
        elif a_type == "kan":
            r = {"type": "daiminkan", "actor": me, "target": target, "pai": pai,
                 "consumed": self._physical_copies(tile, 3)}
        elif a_type == "chi":
            consumed = [tb.physical(t) for t in with_.split()]
            r = {"type": "chi", "actor": me, "target": target, "pai": pai,
                 "consumed": consumed}
        else:
            r = {"type": "none"}
        r["meta"] = self._meta(dist)
        return self._finish_reaction(r)

    def _decide_chankan(self, actor: int, pai: str, ankan: bool = False) -> Optional[dict]:
        actions = self.table.chankan_actions(actor, pai, ankan=ankan)
        if len(actions) <= 1:
            return None
        chosen, dist = self._run_policy(actions, "chankan")
        if ACTION_RE.search(chosen).group(1) == "ron":
            r = {"type": "hora", "actor": self.seat, "target": actor, "pai": pai}
        else:
            r = {"type": "none"}
        r["meta"] = self._meta(dist)
        return self._finish_reaction(r)

    def _physical_copies(self, engine_tile: str, n: int) -> List[str]:
        """n physical copies of `engine_tile` from our hand, plain first."""
        plain = engine_to_mjai(engine_tile)
        have = [t for t in self.table.my_tiles_mjai if mjai_to_engine(t) == engine_tile]
        have.sort(key=lambda t: (t != plain, t))       # plain before red
        if len(have) < n:
            have += [plain] * (n - len(have))
        return have[:n]

    # ---- MahjongCopilot GUI meta (mask_bits + q_values) ---------------
    def _meta(self, dist: Dict[str, float]) -> dict:
        slots: Dict[int, float] = {}
        for xml, p in dist.items():
            m = ACTION_RE.search(xml)
            a_type, tile, with_ = m.group(1), m.group(2), m.group(3)
            key = None
            if a_type == "discard":
                key = self.table.physical(tile) if tile else None
            elif a_type == "riichi":
                key = "reach"
            elif a_type == "chi":
                called = int(tile[0])
                lo = min(int(t[0]) for t in with_.split())
                key = "chi_low" if called < lo else ("chi_high" if called > lo + 1 else "chi_mid")
            elif a_type == "pon":
                key = "pon"
            elif a_type == "kan":
                key = "kan_select"
            elif a_type in ("ron", "tsumo"):
                key = "hora"
            elif a_type == "skip":
                key = "none"
            elif a_type == "kyuushu":
                key = "ryukyoku"
            if key is None or key not in _MASK_INDEX:
                continue
            i = _MASK_INDEX[key]
            slots[i] = slots.get(i, 0.0) + p
        mask_bits = 0
        q_values = []
        for i in sorted(slots):
            mask_bits |= 1 << i
            q_values.append(math.log(max(slots[i], 1e-9)))
        return {"q_values": q_values, "mask_bits": mask_bits}


# ----------------------------------------------------------------------
# Record helpers: what the table actually did, as an engine action
# ----------------------------------------------------------------------
def _event_summary(msg: dict) -> dict:
    keep = ("type", "pai", "consumed", "tsumogiri", "target", "reach")
    return {k: msg[k] for k in keep if k in msg}


def _event_to_action(ev: dict, me: int) -> Optional[str]:
    t = ev.get("type")
    if t == "none":
        return '<action type="skip" />'
    if t == "hora":
        return None                              # tsumo/ron indistinguishable here; fine
    if t == "dahai":
        kind = "riichi" if ev.get("reach") else "discard"
        # spelled: a red five is the engine's '0x' — folding it to '5x' here
        # made "we played the red 5p" read as an override of itself.
        return f'<action type="{kind}" tile="{mjai_to_engine_spelled(ev["pai"])}" />'
    if t == "chi":
        a, b = [mjai_to_engine_spelled(c) for c in ev["consumed"]]
        return f'<action type="chi" tile="{mjai_to_engine_spelled(ev["pai"])}" with="{a} {b}" />'
    if t == "pon":
        return f'<action type="pon" tile="{mjai_to_engine_spelled(ev["pai"])}" />'
    if t in ("daiminkan", "kakan"):
        return f'<action type="kan" tile="{mjai_to_engine_spelled(ev["pai"])}" />'
    if t == "ankan":
        return f'<action type="kan" tile="{mjai_to_engine_spelled(ev["consumed"][0])}" />'
    return None


def _norm_action(xml: str):
    m = ACTION_RE.search(xml or "")
    if not m:
        return None
    return (m.group(1), m.group(2), tuple(sorted((m.group(3) or "").split())))


def _plain(tile: Optional[str]) -> Optional[str]:
    """Tile kind, red five folded onto the plain five ('0m' -> '5m')."""
    return None if tile is None else ("5" + tile[1] if tile[0] == "0" else tile)


_SAME_KIND_TYPES = {"discard", "riichi", "pon", "kan", "chi"}


def compare_execution(chosen: str, executed_action: Optional[str]) -> str:
    """How the table's actual action relates to the policy's pick:

    "match"     — the same action;
    "same_kind" — same action type and same tile KIND, different physical copy
                  (plain vs red five). Counting this as a human override was
                  misleading, so it is reported separately;
    "override"  — a different action (assist mode: the human played something
                  else; or Majsoul refused ours and the turn timed out);
    "unknown"   — the outcome could not be observed (hora, kyoku end).
    """
    a, b = _norm_action(chosen), _norm_action(executed_action)
    if b is None or a is None:
        return "unknown"
    if a == b:
        return "match"
    if (a[0] == b[0] and a[0] in _SAME_KIND_TYPES
            and _plain(a[1]) == _plain(b[1])
            and tuple(sorted(_plain(t) for t in a[2])) == tuple(sorted(_plain(t) for t in b[2]))):
        return "same_kind"
    return "override"
