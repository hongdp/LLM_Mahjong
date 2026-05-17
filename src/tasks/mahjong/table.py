import re
from typing import Dict, List
from src.tasks.mahjong.wrapper import MahjongEngineAPI

class PyMahjongTable(MahjongEngineAPI):
    """
    A specific implementation of MahjongEngineAPI using `pymahjong` (mocked for Phase 0).
    Generates Tenhou Pinyin formatted text states.
    """
    def __init__(self):
        self.turn = 0
        self.round_num = 1
        # Mock internal state for Phase 0 logic tests
        self.hands = {
            0: ["1m", "2m", "3m", "5p", "6p", "7p", "1s", "1s", "2s", "3s", "4s", "5s", "6s"],
            1: ["1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "1m", "2m", "3m", "4m"],
            2: ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "1p", "2p", "3p", "4p"],
            3: ["1z", "2z", "3z", "4z", "5z", "6z", "7z", "1m", "9m", "1p", "9p", "1s", "9s"]
        }
        self.discards = {0: [], 1: [], 2: [], 3: []}
        
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

    def reset(self) -> Dict[int, str]:
        self.turn = 0
        self.discards = {0: [], 1: [], 2: [], 3: []}
        return {i: self._format_state(i) for i in range(4)}

    def step(self, player_id: int, action_xml: str) -> tuple[Dict[int, str], Dict[int, float], bool, Dict]:
        # Basic XML parsing
        match = re.search(r'<action\s+type="([^"]+)"(?:\s+tile="([^"]+)")?.*?/>', action_xml)
        
        rewards = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        
        if not match:
            # Hallucination / Bad Format
            rewards[player_id] = -10.0
        else:
            action_type = match.group(1)
            tile = match.group(2)
            
            if action_type == "discard" and tile in self.hands[player_id]:
                self.hands[player_id].remove(tile)
                self.discards[player_id].append(tile)
                # Mock drawing a tile
                self.hands[player_id].append("5z") 
                self.hands[player_id].sort()
                
                # Turn rotation
                self.turn = (self.turn + 1) % 4
            else:
                # Illegal move
                rewards[player_id] = -5.0
                
        obs = {i: self._format_state(i) for i in range(4)}
        
        # We end the mock game quickly for fast local testing
        done = len(self.discards[0]) > 2
        return obs, rewards, done, {}

    def get_legal_actions(self, player_id: int) -> List[str]:
        return [f'<action type="discard" tile="{t}" />' for t in set(self.hands[player_id])]
