"""Engine self-play -> MJAI event stream (observer perspective).

Plays one `PyMahjongTable` round with per-seat policies and emits the
MJAI events MahjongCopilot would produce for `observer` (other hands and
draws are '?'), in the same order and with the same conventions
(`reach_accepted` deferred to the next action, `dora` before the rinshan
`tsumo`, `kakan` announced before the chankan window).

Used by tests/test_mjai_bridge.py to prove the shadow table reproduces
the engine state bit-for-bit; also handy for dumping MJAI logs of our
self-play for third-party analysers.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from src.agents.dnn.mjai_bridge import engine_to_mjai
from src.tasks.mahjong.claims import _resolve_claims
from src.tasks.mahjong.table import ACTION_RE, PyMahjongTable

Policy = Callable[[PyMahjongTable, int, List[str]], str]   # -> action xml
Sink = Callable[[dict], None]

_BAKAZE = ["E", "S", "W", "N"]


def _meld_event(table, pid: int, before: int, discarder: int, called: str) -> Optional[dict]:
    melds = table.melds[pid]
    if len(melds) == before:
        return None
    m = melds[-1]
    tiles = [engine_to_mjai(t) for t in m["tiles"]]
    consumed = list(tiles)
    consumed.remove(engine_to_mjai(called))
    kind = {"chi": "chi", "pon": "pon", "kan": "daiminkan"}[m["type"]]
    return {"type": kind, "actor": pid, "target": discarder,
            "pai": engine_to_mjai(called), "consumed": consumed}


def play_game_mjai(table: PyMahjongTable, policies: Dict[int, Policy],
                   observer: int, sink: Sink, max_steps: int = 600) -> None:
    """Drive `table` (already reset) to completion, emitting MJAI events."""
    me = observer
    pending_reach_acc: Optional[int] = None

    def emit(ev: dict):
        sink(ev)

    def flush_reach():
        nonlocal pending_reach_acc
        if pending_reach_acc is not None:
            emit({"type": "reach_accepted", "actor": pending_reach_acc})
            pending_reach_acc = None

    def emit_tsumo(pid: int):
        emit({"type": "tsumo", "actor": pid,
              "pai": engine_to_mjai(table.last_drawn[pid]) if pid == me else "?"})

    def emit_new_dora(n_before: int):
        for ind in table.dora_indicators[n_before:]:
            emit({"type": "dora", "dora_marker": engine_to_mjai(ind)})

    # ---- start_kyoku ---------------------------------------------------
    tehais = [["?"] * 13 for _ in range(4)]
    my_hand = list(table.hands[me])
    if me == table.dealer:
        my_hand.remove(table.last_drawn[me])
    tehais[me] = [engine_to_mjai(t) for t in my_hand]
    emit({"type": "start_kyoku", "bakaze": _BAKAZE[table.round_wind_idx],
          "dora_marker": engine_to_mjai(table.dora_indicators[0]),
          "honba": 0, "kyoku": table.dealer + 1, "kyotaku": table.kyotaku // 1000,
          "oya": table.dealer, "scores": list(table.points), "tehais": tehais})
    emit_tsumo(table.dealer)

    guard = 0
    while not table.finished and guard < max_steps:
        guard += 1
        pid = table.turn
        actions = table.get_legal_actions(pid)
        if not actions:
            break
        n_dora = len(table.dora_indicators)
        n_melds = len(table.melds[pid])
        action = policies[pid](table, pid, actions)
        _, _, done, info = table.step(pid, action)
        if done:
            emit({"type": "hora", "actor": pid, "target": pid})
            break
        if info.get("chankan"):
            emit({"type": "kakan", "actor": pid, "pai": engine_to_mjai(info["chankan"]),
                  "consumed": [engine_to_mjai(info["chankan"])] * 3})
        elif info.get("discarded"):
            tile, tsumogiri, rdecl, _, _ = table.river_events[pid][-1]
            if rdecl:
                emit({"type": "reach", "actor": pid})
                pending_reach_acc = pid
            # Majsoul timing: daiminkan / kakan dora is turned over with the
            # kan player's discard (engine: pending_dora_reveal). Emitted
            # BEFORE the dahai so the observer's call decision sees it, as
            # the engine does; a live Majsoul stream sends it after the
            # dahai, so the shadow there learns it one decision late.
            emit_new_dora(n_dora)
            n_dora = len(table.dora_indicators)   # a call on this discard must not re-emit it
            emit({"type": "dahai", "actor": pid, "pai": engine_to_mjai(tile),
                  "tsumogiri": bool(tsumogiri)})
        elif len(table.melds[pid]) > n_melds:            # ankan: dora, rinshan tsumo
            m = table.melds[pid][-1]
            emit({"type": "ankan", "actor": pid,
                  "consumed": [engine_to_mjai(t) for t in m["tiles"]]})
            emit_new_dora(n_dora)
            emit_tsumo(pid)
            continue
        else:
            continue

        # ---- interrupt window ----
        candidates = []
        for offset in range(1, 4):
            other = (pid + offset) % 4
            options = table.get_interrupt_actions(other)
            if len(options) == 1:
                continue
            a_str = policies[other](table, other, options)
            m = ACTION_RE.search(a_str)
            candidates.append({"player_id": other, "parsed": a_str,
                               "type": m.group(1) if m else None, "reward": 0.0})
        melds_before = {p: len(table.melds[p]) for p in range(4)}
        called = table.pending_kan["tile"] if table.pending_kan else table.last_discard
        executed, done = _resolve_claims(table, candidates)
        if done:
            emit({"type": "hora", "actor": executed[0]["player_id"], "target": pid})
            break
        if executed:
            claimant = executed[0]["player_id"]
            ev = _meld_event(table, claimant, melds_before[claimant], pid, called)
            flush_reach()
            emit(ev)
            if ev["type"] == "daiminkan":
                emit_new_dora(n_dora)
                emit_tsumo(claimant)
            continue
        if table.pending_kan:
            table.resolve_pending_kan()
            emit_new_dora(n_dora)
            emit_tsumo(pid)
            continue
        _, r_done = table.advance_turn()
        flush_reach()
        if r_done:
            emit({"type": "ryukyoku"})
            break
        emit_tsumo(table.turn)
    emit({"type": "end_kyoku"})
