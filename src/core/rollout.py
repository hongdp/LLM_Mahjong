from dataclasses import dataclass
from typing import List, Dict, Any
import torch

@dataclass
class TrajectoryStep:
    """Represents a single state-action-reward transition."""
    prompt_text: str
    action_text: str
    reward: float
    is_terminal: bool

class ReplayBuffer:
    """
    Stores full game trajectories (episodes) for all agents.
    Calculates advantages (Return-to-Go) before yielding batches.
    """
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self.episodes: List[List[TrajectoryStep]] = []

    def add_episode(self, episode: List[TrajectoryStep]):
        self.episodes.append(episode)

    def calculate_advantages_and_flatten(self) -> List[Dict[str, Any]]:
        """
        Calculates Return-To-Go for each step in every episode.
        Normalizes the advantages across the entire buffer (GRPO/PPO style).
        Returns a flat list of training samples.
        """
        all_samples = []
        all_returns = []

        # 1. Calculate Returns (discounted sum of future rewards)
        for episode in self.episodes:
            returns = []
            R = 0
            # Iterate backwards
            for step in reversed(episode):
                R = step.reward + self.gamma * R
                returns.insert(0, R)
            all_returns.extend(returns)
            
            # Pair steps with returns
            for step, R in zip(episode, returns):
                all_samples.append({
                    "prompt": step.prompt_text,
                    "action": step.action_text,
                    "return": R
                })

        if not all_returns:
            return []

        # 2. Normalize Returns into Advantages
        returns_tensor = torch.tensor(all_returns, dtype=torch.float32)
        mean = returns_tensor.mean()
        std = returns_tensor.std() + 1e-8
        advantages = (returns_tensor - mean) / std

        # 3. Inject advantages into samples
        for i, sample in enumerate(all_samples):
            sample["advantage"] = advantages[i].item()

        return all_samples

    def clear(self):
        self.episodes = []
