import re
import random
from typing import Dict, List
from src.tasks.mahjong.wrapper import MahjongEngineAPI
from src.tasks.mahjong.shanten import TileEfficiency

class PyMahjongTable(MahjongEngineAPI):
    """
    A fully functional Python Mahjong Table for Phase 0 Testing.
    Handles real 136-tile deck, dealing, and drawing.
    """
    _efficiency = TileEfficiency()

    def _is_winning_hand(self, tiles: List[str]) -> bool:
        # Win check via shanten == -1. Only valid for meld-free 14-tile hands;
        # melded hands (concealed < 13) can never reach -1 with this engine's
        # tile accounting, so they simply won't be offered ron/tsumo.
        if len(tiles) != 14:
            return False
        return self._efficiency.calculate_shanten(tiles) == -1

    def _sort_key(self, tile: str):
        suit_order = {'p': 0, 's': 1, 'm': 2, 'z': 3}
        return (suit_order.get(tile[-1], 4), int(tile[:-1]))

    def __init__(self):
        self.turn = 0
        self.round_num = 1
        
        # Valid Mahjong tiles
        self.ALL_TILES = [f"{i}m" for i in range(1, 10)] + \
                         [f"{i}p" for i in range(1, 10)] + \
                         [f"{i}s" for i in range(1, 10)] + \
                         [f"{i}z" for i in range(1, 8)]
                         
        self.reset()
        
    def reset(self) -> Dict[int, str]:
        self.turn = 0
        self.discards = {0: [], 1: [], 2: [], 3: []}
        self.fulus = {0: [], 1: [], 2: [], 3: []}
        self.last_discard = None
        self.last_discarder = None
        
        # Create a full deck of 136 tiles (4 of each)
        self.wall = self.ALL_TILES * 4
        random.shuffle(self.wall)
        
        # Deal 13 tiles to each player
        self.hands = {0: [], 1: [], 2: [], 3: []}
        for player_id in range(4):
            for _ in range(13):
                self.hands[player_id].append(self.wall.pop())
            self.hands[player_id].sort(key=self._sort_key)
            
        # East draws their first tile
        self.hands[0].append(self.wall.pop())
        self.hands[0].sort(key=self._sort_key)
        
        return {i: self._format_state(i) for i in range(4)}

    def _format_state(self, player_id: int) -> str:
        """Constructs the Tenhou Pinyin state."""
        winds = ["东", "南", "西", "北"]
        state = f"场况 (Global)： 场风: 东, 局数: {self.round_num}, Dora指示牌: 3p, 供托: 0\n"
        state += f"私有 (Private)： 自风: {winds[player_id]}, 点数: 25000, 手牌: {' '.join(self.hands[player_id])}\n"
        state += "公共 (Public)：\n"
        for i in range(4):
            if i != player_id:
                fulu_str = ' '.join(self.fulus[i]) if hasattr(self, 'fulus') and self.fulus[i] else '无'
                state += f"  玩家{i} ({winds[i]}): 牌河: {' '.join(self.discards[i]) if self.discards[i] else '无'}, 副露: {fulu_str}\n"
        return state

    def step(self, player_id: int, action_xml: str) -> tuple[Dict[int, str], Dict[int, float], bool, Dict]:
        match = re.search(r'<action\s+type="([^"]+)"(?:\s+tile="([^"]+)")?.*?/>', action_xml)
        rewards = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        
        discarded = False

        def _forced_discard():
            # Auto-correct an illegal/unparseable action with a random discard
            # so the game always progresses (a "skip" here would loop forever).
            nonlocal discarded
            if not self.hands[player_id]:
                return
            tile = random.choice(self.hands[player_id])
            self.hands[player_id].remove(tile)
            self.discards[player_id].append(tile)
            self.last_discard = tile
            self.last_discarder = player_id
            discarded = True

        if not match:
            # Hallucination / Bad Format
            rewards[player_id] = -10.0
            _forced_discard()
        else:
            action_type = match.group(1)
            tile = match.group(2)

            if action_type == "discard" and tile in self.hands[player_id]:
                self.hands[player_id].remove(tile)
                self.discards[player_id].append(tile)
                self.last_discard = tile
                self.last_discarder = player_id
                discarded = True
            elif action_type == "riichi" and tile in self.hands[player_id]:
                self.hands[player_id].remove(tile)
                self.discards[player_id].append(tile + "*")
                self.last_discard = tile
                self.last_discarder = player_id
                rewards[player_id] = 1.0
                discarded = True
            elif action_type == "tsumo" and self._is_winning_hand(self.hands[player_id]):
                return {i: self._format_state(i) for i in range(4)}, {player_id: 50.0, **{i: -10.0 for i in range(4) if i != player_id}}, True, {}
            else:
                # Includes false tsumo and "skip" during one's own turn —
                # neither is ever legal in the discard phase.
                rewards[player_id] = -5.0
                _forced_discard()
                
        self.hands[player_id].sort(key=self._sort_key)

        # We do not draw here anymore. We wait for interrupt phase.
        done = len(self.wall) <= 14
        obs = {i: self._format_state(i) for i in range(4)}
        # "discarded" tells the orchestrator to run the interrupt phase —
        # string-matching on the model's action text misses riichi and
        # engine-forced discards.
        return obs, rewards, done, {"discarded": discarded}

    def advance_turn(self) -> tuple[Dict[int, str], bool]:
        self.turn = (self.turn + 1) % 4
        done = False
        if len(self.wall) <= 14:
            done = True
        else:
            self.hands[self.turn].append(self.wall.pop())
            self.hands[self.turn].sort(key=self._sort_key)
        return {i: self._format_state(i) for i in range(4)}, done

    def step_interrupt(self, player_id: int, action_xml: str) -> tuple[Dict[int, str], Dict[int, float], bool, Dict]:
        match = re.search(r'<action\s+type="([^"]+)"(?:\s+tile="([^"]+)")?.*?/>', action_xml)
        rewards = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        
        # info["interrupt"] tells the orchestrator whether this call actually
        # took effect (meld/ron) — a penalized illegal attempt behaves as skip.
        if not match: return {i: self._format_state(i) for i in range(4)}, rewards, False, {"interrupt": False}

        action_type = match.group(1)
        tile = match.group(2) or (self.last_discard.replace("*", "") if self.last_discard else "")

        if action_type == "skip":
            return {i: self._format_state(i) for i in range(4)}, rewards, False, {"interrupt": False}

        if action_type == "ron":
            if self._is_winning_hand(self.hands[player_id] + [tile]):
                return {i: self._format_state(i) for i in range(4)}, {player_id: 50.0, **{i: -10.0 for i in range(4) if i != player_id}}, True, {"interrupt": True}
            # False ron: penalize and treat as skip
            rewards[player_id] = -5.0
            return {i: self._format_state(i) for i in range(4)}, rewards, False, {"interrupt": False}

        if not hasattr(self, 'fulus'): self.fulus = {0:[], 1:[], 2:[], 3:[]}

        interrupted = True
        if action_type == "pon" and self.hands[player_id].count(tile) >= 2:
            self.hands[player_id].remove(tile)
            self.hands[player_id].remove(tile)
            self.fulus[player_id].append(f"pon({tile})")
            self.turn = player_id
            if self.discards[self.last_discarder]: self.discards[self.last_discarder].pop()
            rewards[player_id] = 5.0
            
        elif action_type == "kan" and self.hands[player_id].count(tile) >= 3:
            self.hands[player_id].remove(tile)
            self.hands[player_id].remove(tile)
            self.hands[player_id].remove(tile)
            self.fulus[player_id].append(f"kan({tile})")
            self.turn = player_id
            if self.discards[self.last_discarder]: self.discards[self.last_discarder].pop()
            self.hands[player_id].append(self.wall.pop())
            rewards[player_id] = 8.0
            
        elif action_type == "chi" and tile and tile[-1] != 'z' and self._chi_pair(player_id, tile):
            self.turn = player_id
            self.fulus[player_id].append(f"chi({tile})")
            if self.discards[self.last_discarder]: self.discards[self.last_discarder].pop()
            for t in self._chi_pair(player_id, tile):
                self.hands[player_id].remove(t)
            rewards[player_id] = 3.0

        else:
            # Illegal meld attempt (e.g. chi without the sequence pair,
            # pon without duplicates): penalize and treat as skip.
            rewards[player_id] = -5.0
            interrupted = False

        self.hands[player_id].sort(key=self._sort_key)
        return {i: self._format_state(i) for i in range(4)}, rewards, False, {"interrupt": interrupted}

    def _chi_pair(self, player_id: int, tile: str) -> List[str]:
        """Returns the two hand tiles forming a sequence with `tile`, or []."""
        val, suit = int(tile[0]), tile[1]
        hand = self.hands[player_id]
        for a, b in [(val-2, val-1), (val-1, val+1), (val+1, val+2)]:
            if 1 <= a <= 9 and 1 <= b <= 9 and f"{a}{suit}" in hand and f"{b}{suit}" in hand:
                return [f"{a}{suit}", f"{b}{suit}"]
        return []

    def get_interrupt_actions(self, player_id: int) -> List[str]:
        if not getattr(self, 'last_discard', None):
            return ['<action type="skip" />']

        actions = ['<action type="skip" />']
        tile = self.last_discard.replace("*", "")

        if self._is_winning_hand(self.hands[player_id] + [tile]):
            actions.append('<action type="ron" />')

        if self.hands[player_id].count(tile) >= 2:
            actions.append(f'<action type="pon" tile="{tile}" />')
        if self.hands[player_id].count(tile) >= 3:
            actions.append(f'<action type="kan" tile="{tile}" />')
            
        if (self.last_discarder + 1) % 4 == player_id and tile[-1] != 'z':
            if self._chi_pair(player_id, tile):
                actions.append(f'<action type="chi" tile="{tile}" />')

        return list(set(actions))

    def get_legal_actions(self, player_id: int) -> List[str]:
        actions = [f'<action type="discard" tile="{t}" />' for t in sorted(set(self.hands[player_id]), key=self._sort_key)]
        if len(self.hands[player_id]) == 14:
            actions.extend([f'<action type="riichi" tile="{t}" />' for t in sorted(set(self.hands[player_id]), key=self._sort_key)])
            if self._is_winning_hand(self.hands[player_id]):
                actions.append('<action type="tsumo" />')
        return actions
