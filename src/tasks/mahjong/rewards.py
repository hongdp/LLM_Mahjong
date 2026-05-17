import torch
import re
from typing import List
from src.core.base_reward import BaseRewardModel
from src.tasks.mahjong.shanten import TileEfficiency

class MahjongStepReward(BaseRewardModel):
    """
    Implements the Step-level rewards defined in the design doc using real Tile Efficiency math.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_pattern = re.compile(r'<action\s+type="([^"]+)"(?:\s+tile="([^"]+)")?.*?/>')
        self.hand_pattern = re.compile(r'手牌: ([1-9][mpsz] )*[1-9][mpsz]')
        self.te = TileEfficiency()

    def compute_reward(self, prompts: List[str], responses: List[str], **kwargs) -> List[torch.Tensor]:
        rewards = []
        for prompt, response in zip(prompts, responses):
            score = 0.0
            
            # Extract current hand from prompt
            hand_match = self.hand_pattern.search(prompt)
            current_hand = []
            if hand_match:
                current_hand = hand_match.group(0).replace('手牌: ', '').split(' ')

            match = self.action_pattern.search(response)
            if not match:
                score -= 10.0 # Severe hallucination / XML break
            else:
                action_type = match.group(1)
                tile = match.group(2)
                
                if action_type == "discard" and tile and current_hand:
                    if tile not in current_hand:
                        score -= 5.0 # Illegal discard hallucination
                    else:
                        # Evaluate real Ukeire
                        candidates = self.te.evaluate_discards(current_hand)
                        if candidates:
                            # Calculate the max ukeire length among all discards
                            max_ukeire = max([len(u) for u in candidates.values()])
                            chosen_ukeire = len(candidates.get(tile, []))
                            
                            if chosen_ukeire == max_ukeire:
                                score += 2.0 # Perfect efficiency
                            else:
                                # Proportional penalty based on how many tiles of efficiency were lost
                                score -= (max_ukeire - chosen_ukeire) * 0.5
                elif action_type in ["pon", "ron", "riichi"]:
                    score += 1.0 # Base reward for valid game actions
                    
            rewards.append(torch.tensor(score, device=self.device, dtype=torch.float32))
            
        return rewards
