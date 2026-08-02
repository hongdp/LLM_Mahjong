from dataclasses import dataclass
from typing import List, Dict, Any
import torch

@dataclass
class TrajectoryStep:
    """Represents a single state-action-reward transition.

    gen_token_ids / old_logprobs are populated only when the rollout runs
    with capture_logprobs=True (PPO): the exact sampled token ids and the
    behavior policy's raw-logit logprobs for them. Storing ids (not text)
    sidesteps retokenization drift between rollout and update time.
    """
    prompt_text: str
    action_text: str
    reward: float
    is_terminal: bool
    gen_token_ids: list = None
    old_logprobs: list = None
    # Settlement breakdown, filled on the terminal step only (for logs/UI —
    # training keeps consuming the combined `reward`).
    settlement: float = None      # final_rewards[pid] as merged into reward
    final_points: int = None      # points after the game's point transfers
    rank_bonus: float = None      # placement share of the settlement
    game_result: str = None       # engine result_summary string

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
                    "return": R,
                    "gen_token_ids": step.gen_token_ids,
                    "old_logprobs": step.old_logprobs,
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
