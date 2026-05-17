import re
import random
from typing import Dict, List
from src.tasks.mahjong.wrapper import MahjongEngineAPI

class PyMahjongTable(MahjongEngineAPI):
    """
    A fully functional Python Mahjong Table for Phase 0 Testing.
    Handles real 136-tile deck, dealing, and drawing.
    """
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
        
        # Create a full deck of 136 tiles (4 of each)
        self.wall = self.ALL_TILES * 4
        random.shuffle(self.wall)
        
        # Deal 13 tiles to each player
        self.hands = {0: [], 1: [], 2: [], 3: []}
        for player_id in range(4):
            for _ in range(13):
                self.hands[player_id].append(self.wall.pop())
            self.hands[player_id].sort()
            
        # East draws their first tile
        self.hands[0].append(self.wall.pop())
        self.hands[0].sort()
        
        return {i: self._format_state(i) for i in range(4)}

    def _format_state(self, player_id: int) -> str:
        """Constructs the Tenhou Pinyin state."""
        winds = ["东", "南", "西", "北"]
        state = f"场况 (Global)： 场风: 东, 局数: {self.round_num}, Dora指示牌: 3p, 供托: 0\n"
        state += f"私有 (Private)： 自风: {winds[player_id]}, 点数: 25000, 手牌: {' '.join(self.hands[player_id])}\n"
        state += "公共 (Public)：\n"
        for i in range(4):
            if i != player_id:
                state += f"  玩家{i} ({winds[i]}): 牌河: {' '.join(self.discards[i]) if self.discards[i] else '无'}, 副露: 无\n"
        return state

    def step(self, player_id: int, action_xml: str) -> tuple[Dict[int, str], Dict[int, float], bool, Dict]:
        match = re.search(r'<action\s+type="([^"]+)"(?:\s+tile="([^"]+)")?.*?/>', action_xml)
        rewards = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        
        if not match:
            # Hallucination / Bad Format
            rewards[player_id] = -10.0
            # Auto-correct
            tile = random.choice(self.hands[player_id])
            self.hands[player_id].remove(tile)
            self.discards[player_id].append(tile)
        else:
            action_type = match.group(1)
            tile = match.group(2)
            
            if action_type == "discard" and tile in self.hands[player_id]:
                self.hands[player_id].remove(tile)
                self.discards[player_id].append(tile)
            else:
                # Illegal move
                rewards[player_id] = -5.0
                tile = random.choice(self.hands[player_id])
                self.hands[player_id].remove(tile)
                self.discards[player_id].append(tile)
                
        # Sort hand after discarding
        self.hands[player_id].sort()
        
        # Next player's turn
        self.turn = (self.turn + 1) % 4
        
        done = False
        
        # Check if wall is exhausted (leaving 14 tiles dead wall)
        if len(self.wall) <= 14:
            done = True
        else:
            # Next player draws
            self.hands[self.turn].append(self.wall.pop())
            self.hands[self.turn].sort()
            
        obs = {i: self._format_state(i) for i in range(4)}
        return obs, rewards, done, {}

    def get_legal_actions(self, player_id: int) -> List[str]:
        return [f'<action type="discard" tile="{t}" />' for t in set(self.hands[player_id])]
