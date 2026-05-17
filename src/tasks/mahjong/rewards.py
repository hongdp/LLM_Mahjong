import torch
import re
from typing import List
from src.core.base_reward import BaseRewardModel

class MahjongStepReward(BaseRewardModel):
    """
    Implements the Step-level rewards defined in the design doc:
    1. Format/Hallucination penalty (R = -10.0)
    2. Ukeire / Shanten dummy heuristic.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Match <action type="..." ... />
        self.action_pattern = re.compile(r'<action\s+type="([^"]+)"(?:\s+[^>]+)?\s*/>')

    def compute_reward(self, prompts: List[str], responses: List[str], **kwargs) -> List[torch.Tensor]:
        rewards = []
        for prompt, response in zip(prompts, responses):
            score = 0.0
            
            # 1. Format/Legality Check
            match = self.action_pattern.search(response)
            if not match:
                score -= 10.0 # Severe hallucination / XML break
            else:
                action_type = match.group(1)
                
                # If XML is valid, provide a small positive heuristic for the action.
                if action_type == "discard":
                    # Mock Ukeire/Shanten check: +2.0 for making a decision
                    score += 2.0
                elif action_type in ["pon", "ron", "riichi"]:
                    score += 5.0
                    
            rewards.append(torch.tensor(score, device=self.device, dtype=torch.float32))
            
        return rewards
