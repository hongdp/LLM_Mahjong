import random
import re
from typing import Dict, List, Optional

from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld
from mahjong.constants import EAST, SOUTH, WEST, NORTH

from src.tasks.mahjong.wrapper import MahjongEngineAPI
from src.tasks.mahjong.shanten import TileEfficiency, pad_for_melds

WINDS_ZH = ["东", "南", "西", "北"]
WIND_CONST = [EAST, SOUTH, WEST, NORTH]
SUIT_ORDER = {'p': 0, 's': 1, 'm': 2, 'z': 3}
SUIT_BASE_34 = {'m': 0, 'p': 9, 's': 18, 'z': 27}

ACTION_RE = re.compile(
    r'<action\s+type="([^"]+)"(?:\s+tile="([^"]+)")?(?:\s+with="([^"]+)")?\s*/>'
)


def sort_key(tile: str):
    return (SUIT_ORDER.get(tile[-1], 4), int(tile[:-1]))


def str_to_34(tile: str) -> int:
    return SUIT_BASE_34[tile[-1]] + int(tile[:-1]) - 1


def str_from_34(idx: int) -> str:
    if idx < 9:
        return f"{idx + 1}m"
    if idx < 18:
        return f"{idx - 9 + 1}p"
    if idx < 27:
        return f"{idx - 18 + 1}s"
    return f"{idx - 27 + 1}z"


class PyMahjongTable(MahjongEngineAPI):
    """
    Single-round (East 1) four-player riichi mahjong table.

    Rule fidelity: 136-tile wall with a 14-tile dead wall, real dora
    indicators (plus kan dora), chi/pon/daiminkan/ankan/shouminkan with
    structured meld tracking, riichi (tenpai + closed + 1000pt deposit,
    hand locked to tsumogiri), yaku-validated ron/tsumo via the `mahjong`
    HandCalculator, furiten, ryuukyoku tenpai payments, and real point
    settlement. Match structure (renchan / multi-round hanchan / uma) is
    intentionally out of scope: one episode == one round.

    Rewards: per-step rewards only carry format/legality penalties.
    All game-outcome rewards land in `final_rewards` (point deltas x
    REWARD_SCALE, plus an extra deal-in penalty), which the orchestrator
    distributes to every player's trajectory at game end.
    """

    REWARD_SCALE = 0.001
    # No flat deal-in surcharge: the real point transfer plus the placement
    # bonus already price the loss — an extra flat penalty would double-count
    # and push the policy toward over-folding relative to true point EV.
    HOUJUU_EXTRA = 0.0
    ILLEGAL_PENALTY = -5.0
    FORMAT_PENALTY = -10.0
    MAX_MELDS = 4
    # Per-round placement bonus (design doc 4.2 Uma, single-round scale):
    # sharpens the zero-sum ranking pressure that motivates defense.
    RANK_BONUS = [2.0, 0.5, -0.5, -2.0]

    _efficiency = TileEfficiency()
    _calculator = HandCalculator()

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------
    # Setup / state
    # ------------------------------------------------------------------
    def reset(self) -> Dict[int, str]:
        self.turn = 0
        self.dealer = 0
        self.round_wind = EAST
        self.points = [25000, 25000, 25000, 25000]
        self.kyotaku = 0
        self.discards = {i: [] for i in range(4)}
        self.melds = {i: [] for i in range(4)}  # {"type","tiles","opened"}
        self.riichi = [False, False, False, False]
        self.last_discard: Optional[str] = None
        self.last_discarder: Optional[int] = None
        self.last_drawn: List[Optional[str]] = [None, None, None, None]
        self.finished = False
        self.final_rewards: Optional[List[float]] = None
        self.result_summary = ""

        all_tiles = [f"{i}{s}" for s in "mps" for i in range(1, 10)]
        all_tiles += [f"{i}z" for i in range(1, 8)]
        self.wall = all_tiles * 4
        random.shuffle(self.wall)
        self.dead_wall = [self.wall.pop() for _ in range(14)]
        self._next_dora_slot = 5
        self.dora_indicators = [self.dead_wall[4]]

        self.hands = {}
        for pid in range(4):
            self.hands[pid] = sorted(
                [self.wall.pop() for _ in range(13)], key=sort_key
            )
        first = self.wall.pop()
        self.hands[0].append(first)
        self.hands[0].sort(key=sort_key)
        self.last_drawn[0] = first
        return {i: self._format_state(i) for i in range(4)}

    def _meld_str(self, meld: dict) -> str:
        return f"{meld['type']}({' '.join(meld['tiles'])})"

    def _format_state(self, player_id: int) -> str:
        dora = ' '.join(self.dora_indicators)
        state = (
            f"场况 (Global)： 场风: 东, 局数: 东1局, 宝牌指示牌: {dora}, "
            f"供托: {self.kyotaku // 1000}, 剩余牌数: {len(self.wall)}\n"
        )
        own_melds = ' '.join(self._meld_str(m) for m in self.melds[player_id]) or '无'
        riichi_tag = ", 已立直" if self.riichi[player_id] else ""
        state += (
            f"私有 (Private)： 自风: {WINDS_ZH[(player_id - self.dealer) % 4]}, "
            f"点数: {self.points[player_id]}, "
            f"手牌: {' '.join(self.hands[player_id])}, 副露: {own_melds}{riichi_tag}\n"
        )
        state += "公共 (Public)：\n"
        for i in range(4):
            if i == player_id:
                continue
            melds = ' '.join(self._meld_str(m) for m in self.melds[i]) or '无'
            river = ' '.join(self.discards[i]) or '无'
            r_tag = ", 已立直" if self.riichi[i] else ""
            state += (
                f"  玩家{i} ({WINDS_ZH[(i - self.dealer) % 4]}): "
                f"点数: {self.points[i]}, 牌河: {river}, 副露: {melds}{r_tag}\n"
            )
        return state

    # ------------------------------------------------------------------
    # Shanten / win evaluation helpers
    # ------------------------------------------------------------------
    def _shanten(self, tiles: List[str], num_melds: int) -> int:
        """Shanten of a (possibly melded) hand via dummy-triplet padding."""
        # Clamp so padding can never exceed a legal 14-tile hand.
        num_melds = max(0, min(num_melds, (14 - len(tiles)) // 3))
        try:
            return self._efficiency.calculate_shanten(
                pad_for_melds(tiles, num_melds)
            )
        except ValueError:
            # Degenerate tile counts (broken mid-action states) are never
            # winning/tenpai — report "far from tenpai" instead of crashing.
            return 8

    def _alloc_136(self, tiles: List[str], counter: dict) -> List[int]:
        ids = []
        for t in tiles:
            i34 = str_to_34(t)
            copy = min(counter.get(i34, 0), 3)
            ids.append(i34 * 4 + copy)
            counter[i34] = copy + 1
        return ids

    _MELD_TYPE_MAP = {
        "chi": (Meld.CHI, True),
        "pon": (Meld.PON, True),
        "kan": (Meld.KAN, True),
        "ankan": (Meld.KAN, False),
        "shouminkan": (Meld.SHOUMINKAN, True),
    }

    def _win_result(self, player_id: int, win_tile: str, is_tsumo: bool):
        """Returns a HandResponse if (hand + win_tile) is a legal, yaku-
        bearing win for this player, else None."""
        hand = list(self.hands[player_id])
        if is_tsumo:
            if win_tile not in hand:
                return None
            concealed = hand
        else:
            concealed = hand + [win_tile]

        if self._shanten(concealed, len(self.melds[player_id])) != -1:
            return None

        counter: dict = {}
        rest = list(concealed)
        rest.remove(win_tile)
        tile_ids = self._alloc_136(rest, counter)
        win_id = self._alloc_136([win_tile], counter)[0]
        tile_ids.append(win_id)
        meld_objs = []
        for meld in self.melds[player_id]:
            mtype, opened = self._MELD_TYPE_MAP[meld["type"]]
            m_ids = self._alloc_136(meld["tiles"], counter)
            tile_ids.extend(m_ids)
            meld_objs.append(Meld(meld_type=mtype, tiles=m_ids, opened=opened))

        dora_ids = self._alloc_136(self.dora_indicators, {})
        config = HandConfig(
            is_tsumo=is_tsumo,
            is_riichi=self.riichi[player_id],
            player_wind=WIND_CONST[(player_id - self.dealer) % 4],
            round_wind=self.round_wind,
            options=OptionalRules(has_open_tanyao=True),
        )
        result = self._calculator.estimate_hand_value(
            sorted(tile_ids), win_id,
            melds=meld_objs or None,
            dora_indicators=dora_ids,
            config=config,
        )
        return None if result.error else result

    def _waits(self, player_id: int) -> List[str]:
        """Tiles that would complete this player's 13-tile-state hand."""
        hand = self.hands[player_id]
        n_melds = len(self.melds[player_id])
        waits = []
        for i34 in range(34):
            t = str_from_34(i34)
            if hand.count(t) >= 4:
                continue
            if self._shanten(hand + [t], n_melds) == -1:
                waits.append(t)
        return waits

    def _is_furiten(self, player_id: int) -> bool:
        own_river = {t.replace('*', '') for t in self.discards[player_id]}
        return any(w in own_river for w in self._waits(player_id))

    def _can_ron(self, player_id: int, tile: str) -> bool:
        if self._is_furiten(player_id):
            return False
        return self._win_result(player_id, tile, is_tsumo=False) is not None

    def _is_closed(self, player_id: int) -> bool:
        return all(m["type"] == "ankan" for m in self.melds[player_id])

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------
    def get_legal_actions(self, player_id: int) -> List[str]:
        hand = self.hands[player_id]
        drawn = self.last_drawn[player_id]

        # Riichi lock: tsumogiri (or tsumo) only.
        if self.riichi[player_id]:
            actions = []
            if drawn and self._win_result(player_id, drawn, is_tsumo=True):
                actions.append('<action type="tsumo" />')
            if drawn:
                actions.append(f'<action type="discard" tile="{drawn}" />')
            return actions or [
                f'<action type="discard" tile="{t}" />'
                for t in sorted(set(hand), key=sort_key)
            ]

        uniq = sorted(set(hand), key=sort_key)
        actions = [f'<action type="discard" tile="{t}" />' for t in uniq]

        n_melds = len(self.melds[player_id])
        # Riichi declaration: closed hand, >=1000 points, tenpai after discard.
        if self._is_closed(player_id) and self.points[player_id] >= 1000 and drawn:
            for t in uniq:
                rest = list(hand)
                rest.remove(t)
                if self._shanten(rest, n_melds) == 0:
                    actions.append(f'<action type="riichi" tile="{t}" />')

        # Kan from own turn: ankan (4 in hand) / shouminkan (4th tile of own pon).
        if n_melds < self.MAX_MELDS and drawn:
            for t in uniq:
                if hand.count(t) == 4:
                    actions.append(f'<action type="kan" tile="{t}" />')
            for meld in self.melds[player_id]:
                if meld["type"] == "pon" and meld["tiles"][0] in hand:
                    actions.append(f'<action type="kan" tile="{meld["tiles"][0]}" />')

        if drawn and self._win_result(player_id, drawn, is_tsumo=True):
            actions.append('<action type="tsumo" />')
        return actions

    def get_interrupt_actions(self, player_id: int) -> List[str]:
        if self.finished or not self.last_discard:
            return ['<action type="skip" />']

        tile = self.last_discard.replace('*', '')
        actions = ['<action type="skip" />']

        if self._can_ron(player_id, tile):
            actions.append('<action type="ron" />')

        # Riichi players may only ron or pass.
        if self.riichi[player_id] or len(self.melds[player_id]) >= self.MAX_MELDS:
            return actions

        hand = self.hands[player_id]
        if hand.count(tile) >= 2:
            actions.append(f'<action type="pon" tile="{tile}" />')
        if hand.count(tile) >= 3:
            actions.append(f'<action type="kan" tile="{tile}" />')

        if (self.last_discarder + 1) % 4 == player_id and tile[-1] != 'z':
            for pair in self._chi_pairs(player_id, tile):
                actions.append(
                    f'<action type="chi" tile="{tile}" with="{pair[0]} {pair[1]}" />'
                )
        return actions

    def _chi_pairs(self, player_id: int, tile: str) -> List[List[str]]:
        val, suit = int(tile[:-1]), tile[-1]
        hand = self.hands[player_id]
        pairs = []
        for a, b in [(val - 2, val - 1), (val - 1, val + 1), (val + 1, val + 2)]:
            if 1 <= a <= 9 and 1 <= b <= 9:
                ta, tb = f"{a}{suit}", f"{b}{suit}"
                if ta in hand and tb in hand:
                    pairs.append([ta, tb])
        return pairs

    # ------------------------------------------------------------------
    # Turn phase
    # ------------------------------------------------------------------
    def step(self, player_id: int, action_xml: str):
        match = ACTION_RE.search(action_xml or "")
        rewards = {i: 0.0 for i in range(4)}
        discarded = False

        def do_discard(tile: str, riichi_mark: bool = False):
            nonlocal discarded
            self.hands[player_id].remove(tile)
            self.discards[player_id].append(tile + ('*' if riichi_mark else ''))
            self.last_discard = tile
            self.last_discarder = player_id
            self.last_drawn[player_id] = None
            discarded = True

        def forced_discard():
            # Auto-correct so the game always progresses.
            if not self.hands[player_id]:
                return
            if self.riichi[player_id] and self.last_drawn[player_id] in self.hands[player_id]:
                do_discard(self.last_drawn[player_id])
            else:
                do_discard(random.choice(self.hands[player_id]))

        if not match:
            rewards[player_id] = self.FORMAT_PENALTY
            forced_discard()
        else:
            action_type, tile, _with = match.groups()
            hand = self.hands[player_id]
            drawn = self.last_drawn[player_id]

            if action_type == "discard" and tile in hand and (
                not self.riichi[player_id] or tile == drawn
            ):
                do_discard(tile)

            elif (
                action_type == "riichi"
                and tile in hand
                and not self.riichi[player_id]
                and self._is_closed(player_id)
                and self.points[player_id] >= 1000
                and self._riichi_tenpai(player_id, tile)
            ):
                self.riichi[player_id] = True
                self.points[player_id] -= 1000
                self.kyotaku += 1000
                do_discard(tile, riichi_mark=True)

            elif action_type == "tsumo" and drawn and (
                result := self._win_result(player_id, drawn, is_tsumo=True)
            ):
                self._settle_win(player_id, result, is_tsumo=True, discarder=None)
                return (
                    {i: self._format_state(i) for i in range(4)},
                    rewards, True, {"discarded": False},
                )

            elif action_type == "kan" and tile and self._try_own_kan(player_id, tile):
                # Turn continues: player discards after the rinshan draw.
                pass

            else:
                # Illegal action for the turn phase (incl. false tsumo,
                # "skip" on one's own turn, riichi without tenpai).
                rewards[player_id] = self.ILLEGAL_PENALTY
                forced_discard()

        self.hands[player_id].sort(key=sort_key)
        obs = {i: self._format_state(i) for i in range(4)}
        return obs, rewards, self.finished, {"discarded": discarded}

    def _riichi_tenpai(self, player_id: int, discard_tile: str) -> bool:
        rest = list(self.hands[player_id])
        rest.remove(discard_tile)
        return self._shanten(rest, len(self.melds[player_id])) == 0

    def _try_own_kan(self, player_id: int, tile: str) -> bool:
        if len(self.melds[player_id]) >= self.MAX_MELDS or self.riichi[player_id]:
            return False
        hand = self.hands[player_id]
        if hand.count(tile) == 4:  # ankan
            for _ in range(4):
                hand.remove(tile)
            self.melds[player_id].append(
                {"type": "ankan", "tiles": [tile] * 4, "opened": False}
            )
        else:  # shouminkan: upgrade own pon
            pon = next(
                (m for m in self.melds[player_id]
                 if m["type"] == "pon" and m["tiles"][0] == tile),
                None,
            )
            if pon is None or tile not in hand:
                return False
            hand.remove(tile)
            pon["type"] = "shouminkan"
            pon["tiles"] = [tile] * 4
        self._reveal_kan_dora()
        self._rinshan_draw(player_id)
        return True

    def _reveal_kan_dora(self):
        if self._next_dora_slot < len(self.dead_wall):
            self.dora_indicators.append(self.dead_wall[self._next_dora_slot])
            self._next_dora_slot += 1

    def _rinshan_draw(self, player_id: int):
        if self.dead_wall:
            t = self.dead_wall.pop(0)
            self.hands[player_id].append(t)
            self.hands[player_id].sort(key=sort_key)
            self.last_drawn[player_id] = t

    def advance_turn(self):
        self.turn = (self.turn + 1) % 4
        if not self.wall:
            self._ryuukyoku()
            return {i: self._format_state(i) for i in range(4)}, True
        t = self.wall.pop()
        self.hands[self.turn].append(t)
        self.hands[self.turn].sort(key=sort_key)
        self.last_drawn[self.turn] = t
        return {i: self._format_state(i) for i in range(4)}, False

    # ------------------------------------------------------------------
    # Interrupt phase
    # ------------------------------------------------------------------
    def step_interrupt(self, player_id: int, action_xml: str):
        match = ACTION_RE.search(action_xml or "")
        rewards = {i: 0.0 for i in range(4)}
        obs = lambda: {i: self._format_state(i) for i in range(4)}

        if not match:
            rewards[player_id] = self.FORMAT_PENALTY
            return obs(), rewards, False, {"interrupt": False}

        action_type = match.group(1)
        # SECURITY: the claimed tile is ALWAYS the actual last discard —
        # model-supplied tile attributes are never trusted here.
        tile = self.last_discard.replace('*', '') if self.last_discard else None

        if action_type == "skip" or tile is None:
            return obs(), rewards, False, {"interrupt": False}

        if action_type == "ron":
            if self._can_ron(player_id, tile):
                result = self._win_result(player_id, tile, is_tsumo=False)
                self._settle_win(
                    player_id, result, is_tsumo=False, discarder=self.last_discarder
                )
                return obs(), rewards, True, {"interrupt": True}
            rewards[player_id] = self.ILLEGAL_PENALTY
            return obs(), rewards, False, {"interrupt": False}

        interrupted = False
        hand = self.hands[player_id]
        can_meld = (
            not self.riichi[player_id]
            and len(self.melds[player_id]) < self.MAX_MELDS
        )

        if action_type == "pon" and can_meld and hand.count(tile) >= 2:
            hand.remove(tile)
            hand.remove(tile)
            self.melds[player_id].append(
                {"type": "pon", "tiles": [tile] * 3, "opened": True}
            )
            self._claim_discard(player_id)
            interrupted = True

        elif action_type == "kan" and can_meld and hand.count(tile) >= 3:
            for _ in range(3):
                hand.remove(tile)
            self.melds[player_id].append(
                {"type": "kan", "tiles": [tile] * 4, "opened": True}
            )
            self._claim_discard(player_id)
            self._reveal_kan_dora()
            self._rinshan_draw(player_id)
            interrupted = True

        elif action_type == "chi" and can_meld and tile[-1] != 'z' and (
            (self.last_discarder + 1) % 4 == player_id
        ):
            pairs = self._chi_pairs(player_id, tile)
            if pairs:
                wanted = (match.group(3) or "").split()
                chosen = next((p for p in pairs if sorted(p) == sorted(wanted)), pairs[0])
                for t in chosen:
                    hand.remove(t)
                seq = sorted(chosen + [tile], key=sort_key)
                self.melds[player_id].append(
                    {"type": "chi", "tiles": seq, "opened": True}
                )
                self._claim_discard(player_id)
                interrupted = True

        if not interrupted:
            rewards[player_id] = self.ILLEGAL_PENALTY

        hand.sort(key=sort_key)
        return obs(), rewards, False, {"interrupt": interrupted}

    def _claim_discard(self, player_id: int):
        if self.discards[self.last_discarder]:
            self.discards[self.last_discarder].pop()
        self.turn = player_id
        self.last_discard = None
        self.last_drawn[player_id] = None

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------
    def _settle_win(self, winner: int, result, is_tsumo: bool, discarder):
        cost = result.cost
        if is_tsumo:
            if winner == self.dealer:
                for i in range(4):
                    if i != winner:
                        self.points[i] -= cost['main']
                        self.points[winner] += cost['main']
            else:
                for i in range(4):
                    if i == winner:
                        continue
                    pay = cost['main'] if i == self.dealer else cost['additional']
                    self.points[i] -= pay
                    self.points[winner] += pay
        else:
            self.points[discarder] -= cost['main']
            self.points[winner] += cost['main']

        self.points[winner] += self.kyotaku
        self.kyotaku = 0
        self.finished = True

        yaku = ', '.join(str(y) for y in (result.yaku or []))
        how = "自摸" if is_tsumo else f"荣和(放铳:玩家{discarder})"
        self.result_summary = (
            f"玩家{winner} {how} | {result.han}番{result.fu}符 | {yaku} | "
            f"点数: {self.points}"
        )
        self._compute_final_rewards(houjuu_player=discarder if not is_tsumo else None)

    def _ryuukyoku(self):
        if self.finished:
            return
        tenpai = [
            self._shanten(self.hands[i], len(self.melds[i])) <= 0 for i in range(4)
        ]
        n = sum(tenpai)
        if 0 < n < 4:
            gain = 3000 // n
            loss = 3000 // (4 - n)
            for i in range(4):
                self.points[i] += gain if tenpai[i] else -loss
        self.finished = True
        self.result_summary = (
            f"流局 | 听牌: {[i for i in range(4) if tenpai[i]]} | 点数: {self.points}"
        )
        self._compute_final_rewards(houjuu_player=None)

    def _compute_final_rewards(self, houjuu_player):
        self.final_rewards = [
            (self.points[i] - 25000) * self.REWARD_SCALE for i in range(4)
        ]
        if houjuu_player is not None:
            self.final_rewards[houjuu_player] += self.HOUJUU_EXTRA

        # Placement bonus: players tied on points share the average bonus
        # of the positions they span (an all-25000 draw gives everyone 0).
        order = sorted(range(4), key=lambda i: -self.points[i])
        pos = 0
        while pos < 4:
            tied = [i for i in order if self.points[i] == self.points[order[pos]]]
            shared = sum(self.RANK_BONUS[pos:pos + len(tied)]) / len(tied)
            for i in tied:
                self.final_rewards[i] += shared
            pos += len(tied)
