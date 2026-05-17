from abc import ABC, abstractmethod
from typing import Any
from src.core.rollout import ReplayBuffer

class BaseTask(ABC):
    """
    Abstract Base Class for RL Tasks with custom Multi-Turn Rollouts.
    """

    @abstractmethod
    def collect_rollouts(self, num_episodes: int, model=None, tokenizer=None) -> ReplayBuffer:
        """
        Runs the environment interactively using the model and returns a populated ReplayBuffer.
        """
        pass
