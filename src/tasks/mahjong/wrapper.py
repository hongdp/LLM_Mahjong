from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class MahjongEngineAPI(ABC):
    """
    Abstract wrapper for Mahjong engines to ensure we can swap engines (e.g., pymahjong, mjx)
    without rewriting the RLHF logic.
    """
    
    @abstractmethod
    def reset(self) -> Dict[int, str]:
        """
        Resets the table and returns the initial observation for each player.
        Returns:
            Dict mapping player_id (0-3) to their Tenhou-formatted state string.
        """
        pass

    @abstractmethod
    def step(self, player_id: int, action_xml: str) -> tuple[Dict[int, str], Dict[int, float], bool, Dict]:
        """
        Takes an action for a specific player.
        Args:
            player_id: 0, 1, 2, or 3.
            action_xml: The raw XML action string from the LLM (e.g., '<action type="discard" tile="1s" />')
            
        Returns:
            observations: Dict mapping player_id to their new state string.
            rewards: Dict mapping player_id to their step reward.
            done: Boolean indicating if the game/round is over.
            info: Additional debug info.
        """
        pass

    @abstractmethod
    def get_legal_actions(self, player_id: int) -> List[str]:
        """
        Returns a list of legal actions (in XML format) for a given player.
        Useful for masking or baseline heuristic bots.
        """
        pass
