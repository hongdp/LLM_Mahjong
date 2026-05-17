import torch
from typing import List
from src.rewards.base import BaseRewardModel
from src.rewards.registry import register_reward

@register_reward("length_penalty")
class LengthPenaltyReward(BaseRewardModel):
    """
    A simple dummy reward model that penalizes overly long responses.
    Useful for local sanity checks (Phase 0).
    """
    def __init__(self, target_length: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.target_length = target_length

    def compute_reward(self, prompts: List[str], responses: List[str], **kwargs) -> List[torch.Tensor]:
        rewards = []
        for response in responses:
            # Simple heuristic: reward is higher if length is closer to target
            length = len(response)
            penalty = abs(length - self.target_length) * -0.01
            # Return as tensor on the specified device
            rewards.append(torch.tensor(penalty, device=self.device, dtype=torch.float32))
        return rewards

@register_reward("mock_sentiment")
class MockSentimentReward(BaseRewardModel):
    """
    A mock sentiment reward model that rewards the presence of positive words.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.positive_words = ["good", "great", "excellent", "awesome", "perfect"]

    def compute_reward(self, prompts: List[str], responses: List[str], **kwargs) -> List[torch.Tensor]:
        rewards = []
        for response in responses:
            score = 0.0
            lower_resp = response.lower()
            for word in self.positive_words:
                if word in lower_resp:
                    score += 1.0
            rewards.append(torch.tensor(score, device=self.device, dtype=torch.float32))
        return rewards
